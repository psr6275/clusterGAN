import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

mnist = input_data.read_data_sets('./data/mnist')
slim = tc.slim

##Finally, I want to share the weight discriminator and encoder
##Implement with slim!


class Discriminator(object):
    def __init__(self,x_dim = 784):
        self.x_dim = x_dim
        self.name = 'mnist/cgan/d_net'

    def __call__(self,x,reuse=True):
        with tf.variable_scope(self.name,reuse = reuse) as vs:
            conv1 = slim.conv2d(x,64,4,2,activation_fn = tf.nn.relu)
            conv2 = slim.conv2d(conv1,128,4,2,activation_fn = tf.nn.relu)
            fc1 = slim.fully_connected(conv2,1024,activation_fn = tf.nn.relu)
            fc2 = slim.fully_connected(fc1, 1, activation_fn = None)
        self.variables = tc.framework.get_variables(vs)
        return fc2
## Discriminator를 BEGAN 처럼 만드는 건 어떨까?
class Generator(object):
    def __init__(self,z_dim = 10,x_dim = 784):
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.name = 'mnist/clus_wgan/g_net'

    def __call__(self,z,reuse = True):
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            bs = tf.shape(z)[0]
            fc1 = slim.fully_connected(z,1024,regularizer = slim.l2_regularizer(2.5e-5),activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm)
            fc2 = slim.fully_connected(fc1,7*7*128,weight_regularizer = slim.l2_regularizer(2.5e-5),activation_fn=tf.nn.relu)
            fc2 = tf.reshape(fc2,tf.stack([bs,7,7,128]))
            fc = slim.batch_norm(fc2)
            conv1 = slim.conv2d(fc,64,4,2,weight_regularizer=slim.l2_regularizer(2.5e-5),activation_fn = tf.nn.relu,normalizer_fn = slim.batch_norm)
            conv2 = slim.conv2d(conv1,1,4,2,weight_regularizer = slim.l2_regularizer(2.5e-5),activation_fn = tf.nn.sigmoid)
            conv2 = tf.reshape(conv2,tf.stack([bs,self.x_dim]))

        self.variables = tc.framework.get_variables(vs)
        return conv2

class Encoder(object):
    def __init__(self,z_dim = 10,dim_gen = 10,x_dim = 784):
        self.z_dim = z_dim
        self.dim_gen = dim_gen
        self.x_dim = x_dim
        self.name = 'mnist/clus_wgan/enc_net'

    def __call__(self,x,reuse=True):
        with tf.variable_scope(self.name,reuse = reuse) as vs:
            bs = tf.shape(x)[0]
            x = tf.reshape(x,[bs,28,28,1])
            conv1 = slim.conv2d(x,64,4,2,weight_regularizer = slim.l2_regularizer(2.5e-5),activation_fn = tf.nn.relu)
            conv2 = slim.conv2d(conv1,128,4,2,weight_regularizer = slim.l2_regularizer(2.5e-5),activation_fn = tf.nn.relu)
            conv2 = tcl.flatten(conv2)
            fc1 = slim.fully_connected(conv2,1024,weight_regularizer = slim.l2_regularizer(2.5e-5),activation_fn = tf.nn.relu)
            fc2 = slim.fully_connected(fc1,self.z_dim,activation_fn = None)
            logits = fc2[:,self.dim_gen:]
            y = tf.nn.softmax(logits)
        self.variables = tc.framework.get_variables(vs)
        return fc2[:,:self.dim_gen],y, logits

### Data related part!

class DataSampler(object):
    def __init__(self):
        self.shape = [28,28,1]

    def train(self,batch_size, label=False):
        if label:
            return mnist.train.next_batch(batch_size)
        else:
            return mnist.train.next_batch(batch_size)
    def test(self,batch_size =None):
        if batch_size < len(mnist.test.labels) and batch_size is not None:
            return mnist.test.next_batches(batch_size)
        else:
            return mnist.test.images, mnist.test.labels
    def validation(self,batch_dize = None):
        if batch_size < len(mnist.validation.labels) and batch_size is not None:
            return mnist.validation.next_batches(batch_size)
        else:
            return mnist.validation.images, mnist.validation.lables
    def data2img(self,data):
        return np.reshape(data,[data.shape[0]]+self.shape)

    def load_all(self):
        X_train = mnist.train.images
        X_val = mnist.validation.images
        X_test = mnist.test.images

        Y_train = mnist.train.labels
        Y_val = mnist.validation.labels
        Y_test = mnist.test.labels

        X = np.concatenate((X_train, X_val, X_test))
        Y = np.concatenate((Y_train, Y_val, Y_test))

        return X, Y.flatten()

class NoiseSampler(object):
    def __init__(self,z_dim=100,mode = 'uniform'):
        self.mode = mode
        self.z_dim = z_dim
        self.K = 10

        if self.mode == 'mix_gauss':
            self.mu_mat = (1.0) * np.eye(self.K, self.z_dim) # initial Mu is one hot!
            self.sig = 0.1

        elif self.mode == 'one_hot':
            self.mu_mat = (1.0) * np.eye(self.K)
            self.sig = 0.10

        elif self.mode == 'pca_kmeans':

            data_x = mnist.train.images
            feature_mean = np.mean(data_x, axis=0)
            data_x -= feature_mean # data centering
            data_embed = PCA(n_components=self.z_dim, random_state=0).fit_transform(data_x)
            data_x += feature_mean #original data
            kmeans = KMeans(n_clusters=self.K, random_state=0)
            kmeans.fit(data_embed)
            self.mu_mat = kmeans.cluster_centers_ #cluster centers on pca data
            shift = np.min(self.mu_mat)
            scale = np.max(self.mu_mat - shift)
            self.mu_mat = (self.mu_mat - shift) / scale ## normalize 0~1
            self.sig = 0.15

    def __call__(self, batch_size, z_dim):
        if self.mode == 'uniform':
            return np.random.uniform(-1.0, 1.0, [batch_size, z_dim])
        elif self.mode == 'normal':
            return 0.15 * np.random.randn(batch_size, z_dim)
        elif self.mode == 'mix_gauss':
            k = np.random.randint(low=0, high=self.K, size=batch_size)
            return self.sig * np.random.randn(batch_size, z_dim) + self.mu_mat[k]
        elif self.mode == 'pca_kmeans':
            k = np.random.randint(low=0, high=self.K, size=batch_size)
            return self.sig * np.random.randn(batch_size, z_dim) + self.mu_mat[k]
        elif self.mode == 'one_hot':
            k = np.random.randint(low=0, high=self.K, size=batch_size) #random part and one hot part concatenate
            return np.hstack((self.sig * np.random.randn(batch_size, z_dim - self.K), self.mu_mat[k]))


#####################################################
#####Cluster GAN###################
###################################
tf.set_random_seed(0)


class clusGAN(object):
    def __init__(self, g_net, d_net, enc_net, x_sampler, z_sampler, data, model, sampler,
                 num_classes, dim_gen, n_cat, batch_size, beta_cycle_gen, beta_cycle_label):
        self.model = model
        self.data = data
        self.sampler = sampler
        self.g_net = g_net
        self.d_net = d_net
        self.enc_net = enc_net
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
        self.num_classes = num_classes
        self.dim_gen = dim_gen
        self.n_cat = n_cat
        self.batch_size = batch_size
        scale = 10.0
        self.beta_cycle_gen = beta_cycle_gen
        self.beta_cycle_label = beta_cycle_label

        self.x_dim = self.d_net.x_dim
        self.z_dim = self.g_net.z_dim

        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.z_gen = self.z[:, 0:self.dim_gen]
        self.z_hot = self.z[:, self.dim_gen:]# one hot encoding part(label and discrete part)

        self.x_ = self.g_net(self.z) # generated input
        self.z_enc_gen, self.z_enc_label, self.z_enc_logits = self.enc_net(self.x_, reuse=False) # latent variables from generated inputs
        self.z_infer_gen, self.z_infer_label, self.z_infer_logits = self.enc_net(self.x) #latent variables from training inputs

        self.d = self.d_net(self.x, reuse=False) #discriminator output from true inputs.
        self.d_ = self.d_net(self.x_) #discriminator output from generated inputs.

        self.g_loss = tf.reduce_mean(self.d_) + \ # reduce discriminator loss of generated inputs
                      self.beta_cycle_gen * tf.reduce_mean(tf.square(self.z_gen - self.z_enc_gen)) + \ # reduce the difference between latent variables
                      self.beta_cycle_label * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.z_enc_logits, labels=self.z_hot)) #reduce the loss between one hot intended and induced for generated inputs

        self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_) #well disentangle true and generated inputs

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.x_ # interpolated inputs
        d_hat = self.d_net(x_hat) #discriminator value of interpolated input

        ddx = tf.gradients(d_hat, x_hat)[0] #
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale) # why the gradient size should be 1? this part is related to gan gp

        self.d_loss = self.d_loss + ddx # instead of clipping updated weights regularize the gradient of discriminators

        self.d_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(self.d_loss, var_list=self.d_net.vars) # for discriminator the model should update discriminator
        self.g_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \ # ignore the discriminator variable
            .minimize(self.g_loss, var_list=[self.g_net.vars, self.enc_net.vars]) #for generator we should train generator and encocder network

        # Reconstruction Nodes
        self.recon_loss = tf.reduce_mean(tf.abs(self.x - self.x_), 1) # add the reconstruction loss
        self.compute_grad = tf.gradients(self.recon_loss, self.z) #

        self.saver = tf.train.Saver()

        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

    def train(self, num_batches=500000):

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

        batch_size = self.batch_size
        plt.ion()
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        print(
            'Training {} on {}, sampler = {}, z = {} dimension, beta_n = {}, beta_c = {}'.
                format(self.model, self.data, self.sampler, self.z_dim, self.beta_cycle_gen, self.beta_cycle_label))

        im_save_dir = 'logs/{}/{}/{}_z{}_cyc{}_gen{}'.format(self.data, self.model, self.sampler, self.z_dim,
                                                             self.beta_cycle_label, self.beta_cycle_gen)
        if not os.path.exists(im_save_dir):
            os.makedirs(im_save_dir)

        for t in range(0, num_batches):
            d_iters = 5

            for _ in range(0, d_iters):
                bx = self.x_sampler.train(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)
                self.sess.run(self.d_adam, feed_dict={self.x: bx, self.z: bz})

            bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)
            self.sess.run(self.g_adam, feed_dict={self.z: bz})

            if (t + 1) % 100 == 0:
                bx = self.x_sampler.train(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)

                d_loss = self.sess.run(
                    self.d_loss, feed_dict={self.x: bx, self.z: bz}
                )
                g_loss = self.sess.run(
                    self.g_loss, feed_dict={self.z: bz}
                )
                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                      (t + 1, time.time() - start_time, d_loss, g_loss))

            if (t + 1) % 5000 == 0:
                bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)
                bx = self.sess.run(self.x_, feed_dict={self.z: bz})
                bx = xs.data2img(bx)
                bx = grid_transform(bx, xs.shape)

                imsave('logs/{}/{}/{}_z{}_cyc{}_gen{}/{}.png'.format(self.data, self.model, self.sampler,
                                                                     self.z_dim, self.beta_cycle_label,
                                                                     self.beta_cycle_gen, (t + 1) / 100), bx)

        self.recon_enc(timestamp, val=True)
        self.save(timestamp)

    def save(self, timestamp):

        checkpoint_dir = 'checkpoint_dir/{}/{}_{}_{}_z{}_cyc{}_gen{}'.format(self.data, timestamp, self.model,
                                                                             self.sampler,
                                                                             self.z_dim, self.beta_cycle_label,
                                                                             self.beta_cycle_gen)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'))

    def load(self, pre_trained=False, timestamp=''):

        if pre_trained == True:
            print('Loading Pre-trained Model...')
            checkpoint_dir = 'pre_trained_models/{}/{}_{}_z{}_cyc{}_gen{}'.format(self.data, self.model, self.sampler,
                                                                                  self.z_dim, self.beta_cycle_label,
                                                                                  self.beta_cycle_gen)
        else:
            if timestamp == '':
                print('Best Timestamp not provided. Abort !')
                checkpoint_dir = ''
            else:
                checkpoint_dir = 'checkpoint_dir/{}/{}_{}_{}_z{}_cyc{}_gen{}'.format(self.data, timestamp, self.model,
                                                                                     self.sampler,
                                                                                     self.z_dim, self.beta_cycle_label,
                                                                                     self.beta_cycle_gen)

        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'))
        print('Restored model weights.')

    def _gen_samples(self, num_images):

        batch_size = self.batch_size
        bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)
        fake_im = self.sess.run(self.x_, feed_dict={self.z: bz})
        for t in range(num_images // batch_size):
            bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)
            im = self.sess.run(self.x_, feed_dict={self.z: bz})
            fake_im = np.vstack((fake_im, im))

        print(' Generated {} images .'.format(fake_im.shape[0]))
        np.save('./Image_samples/{}/{}_{}_K_{}_gen_images.npy'.format(self.data, self.model, self.sampler,
                                                                      self.num_classes), fake_im)

    def gen_from_all_modes(self):

        if self.sampler == 'one_hot':
            batch_size = 1000
            label_index = np.tile(np.arange(self.num_classes), int(np.ceil(batch_size * 1.0 / self.num_classes)))

            bz = self.z_sampler(batch_size, self.z_dim, self.sampler, num_class=self.num_classes,
                                n_cat=self.n_cat, label_index=label_index)
            bx = self.sess.run(self.x_, feed_dict={self.z: bz})

            for m in range(self.num_classes):
                print('Generating samples from mode {} ...'.format(m))
                mode_index = np.where(label_index == m)[0]
                mode_bx = bx[mode_index, :]
                mode_bx = xs.data2img(mode_bx)
                mode_bx = grid_transform(mode_bx, xs.shape)

                imsave('logs/{}/{}/{}_z{}_cyc{}_gen{}/mode{}_samples.png'.format(self.data, self.model, self.sampler,
                                                                                 self.z_dim, self.beta_cycle_label,
                                                                                 self.beta_cycle_gen, m), mode_bx)

    def recon_enc(self, timestamp, val=True):

        if val:
            data_recon, label_recon = self.x_sampler.validation()
        else:
            data_recon, label_recon = self.x_sampler.test()
            # data_recon, label_recon = self.x_sampler.load_all()

        num_pts_to_plot = data_recon.shape[0]
        recon_batch_size = self.batch_size
        latent = np.zeros(shape=(num_pts_to_plot, self.z_dim))

        print('Data Shape = {}, Labels Shape = {}'.format(data_recon.shape, label_recon.shape))
        for b in range(int(np.ceil(num_pts_to_plot * 1.0 / recon_batch_size))):
            if (b + 1) * recon_batch_size > num_pts_to_plot:
                pt_indx = np.arange(b * recon_batch_size, num_pts_to_plot)
            else:
                pt_indx = np.arange(b * recon_batch_size, (b + 1) * recon_batch_size)
            xtrue = data_recon[pt_indx, :]

            zhats_gen, zhats_label = self.sess.run([self.z_infer_gen, self.z_infer_label], feed_dict={self.x: xtrue})

            latent[pt_indx, :] = np.concatenate((zhats_gen, zhats_label), axis=1)

        if self.beta_cycle_gen == 0:
            self._eval_cluster(latent[:, self.dim_gen:], label_recon, timestamp, val)
        else:
            self._eval_cluster(latent, label_recon, timestamp, val)

    def _eval_cluster(self, latent_rep, labels_true, timestamp, val):

        if self.data == 'fashion' and self.num_classes == 5:
            map_labels = {0: 0, 1: 1, 2: 2, 3: 0, 4: 2, 5: 3, 6: 2, 7: 3, 8: 4, 9: 3}
            labels_true = np.array([map_labels[i] for i in labels_true])

        km = KMeans(n_clusters=max(self.num_classes, len(np.unique(labels_true))), random_state=0).fit(latent_rep)
        labels_pred = km.labels_

        purity = metric.compute_purity(labels_pred, labels_true)
        ari = adjusted_rand_score(labels_true, labels_pred)
        nmi = normalized_mutual_info_score(labels_true, labels_pred)

        if val:
            data_split = 'Validation'
        else:
            data_split = 'Test'
            # data_split = 'All'

        print('Data = {}, Model = {}, sampler = {}, z_dim = {}, beta_label = {}, beta_gen = {} '
              .format(self.data, self.model, self.sampler, self.z_dim, self.beta_cycle_label, self.beta_cycle_gen))
        print(' #Points = {}, K = {}, Purity = {},  NMI = {}, ARI = {},  '
              .format(latent_rep.shape[0], self.num_classes, purity, nmi, ari))

        with open('logs/Res_{}_{}.txt'.format(self.data, self.model), 'a+') as f:
            f.write(
                '{}, {} : K = {}, z_dim = {}, beta_label = {}, beta_gen = {}, sampler = {}, Purity = {}, NMI = {}, ARI = {}\n'
                .format(timestamp, data_split, self.num_classes, self.z_dim, self.beta_cycle_label, self.beta_cycle_gen,
                        self.sampler, purity, nmi, ari))
            f.flush()
