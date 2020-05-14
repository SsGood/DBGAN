from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCN(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(GCN, self).__init__(**kwargs)
        """
        inputs:输入
        input_dim:feature的数量，即input的维度？
        feature_nonzero：非0的特征
        adj:邻接矩阵
        dropout：dropout
        """

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']

    def construct(self, inputs = None, hidden = None, reuse = False):
        if inputs == None :
            inputs = self.inputs

            
        with tf.variable_scope('Encoder', reuse=reuse):
            self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                                  output_dim=FLAGS.hidden1,
                                                  adj=self.adj,
                                                  features_nonzero = self.features_nonzero,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  logging=self.logging,
                                                  name='e_dense_1')(inputs)
                                                  
                                                  
            self.noise = gaussian_noise_layer(self.hidden1, 0.1)
            if hidden == None:
                hidden = self.hidden1

            self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging,
                                           name='e_dense_2')(hidden)


            self.z_mean = self.embeddings

            self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                          act=lambda x: x,
                                          logging=self.logging)(self.embeddings)
            return self.z_mean, self.reconstructions




class Generator_z2g(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(Generator_z2g, self).__init__(**kwargs)
        """
        inputs:输入
        input_dim:feature的数量，即input的维度？
        feature_nonzero：非0的特征
        adj:邻接矩阵
        dropout：dropout
        """

        self.inputs = placeholders['real_distribution']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']


    def construct(self, inputs = None, hidden = None, reuse = False):
        if inputs == None:
            inputs = self.inputs
        with tf.device("/gpu:1"):
            with tf.variable_scope('Decoder', reuse=reuse):
            
                self.hidden1 = GraphConvolution(input_dim=FLAGS.hidden2,
                                                      output_dim=FLAGS.hidden1,
                                                      adj=self.adj,
                                                      act=tf.nn.relu,
                                                      dropout=self.dropout,
                                                      logging=self.logging,
                                                      name='GG_dense_1')(inputs)
                if hidden == None:
                    hidden = self.hidden1

                self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                               output_dim=self.input_dim,
                                               adj=self.adj,
                                               act=lambda x: x,
                                               dropout=self.dropout,
                                               logging=self.logging,
                                               name='GG_dense_2')(self.hidden1)


            self.z_mean = self.embeddings
            return self.z_mean


def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        # np.random.seed(1)
        tf.set_random_seed(1)
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


class D_graph(Model):
    def __init__(self, num_features, **kwargs):
        super(D_graph, self).__init__(**kwargs)

        self.act = tf.nn.relu
        self.num_features = num_features

    def construct(self, inputs, reuse = False):
        # input是一张Graph的adj，把每一列当成一个通道，所以input的通道数是num_nodes
        with tf.variable_scope('D_Graph'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # np.random.seed(1)
            #tf.set_random_seed(1)
            with tf.device("/gpu:1"):
                dc_den1 = tf.nn.relu(dense(inputs, self.num_features, 512, name='GD_den1'))#(bs,num_nodes,512)
                dc_den2 = tf.nn.relu(dense(dc_den1, 512, 128, name='GD_den2'))#(bs, num_nodes, 128)
                output = dense(dc_den2, 128, 1, name='GD_output')#(bs,num_nodes,1)
            return output
            
def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise    

class Discriminator(Model):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

        self.act = tf.nn.relu

    def construct(self, inputs, reuse = False):
        # with tf.name_scope('Discriminator'):
        with tf.device("/gpu:1"):
            with tf.variable_scope('Discriminator'):
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                # np.random.seed(1)
                tf.set_random_seed(1)
                dc_den1 = tf.nn.relu(dense(inputs, FLAGS.hidden2, FLAGS.hidden3, name='dc_den1'))
                dc_den2 = tf.nn.relu(dense(dc_den1, FLAGS.hidden3, FLAGS.hidden1, name='dc_den2'))
                output = dense(dc_den2, FLAGS.hidden1, 1, name='dc_output')
                return output