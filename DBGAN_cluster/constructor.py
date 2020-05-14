import tensorflow as tf
import numpy as np
from model import GCN, Generator_z2g, Discriminator, D_graph
from optimizer import OptimizerAE, OptimizerCycle
import scipy.sparse as sp
from input_data import load_data
import inspect
from preprocessing import preprocess_graph, sparse_to_tuple, mask_test_edges, construct_feed_dict


flags = tf.app.flags
FLAGS = flags.FLAGS

def get_placeholder(adj,num_features):
# 给tf.sparse_placeholder喂数据时：
#   1.应该直接填充 (indices, values, shape)
#   2.或者使用 tf.SparseTensorValue

    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'features_dense': tf.placeholder(tf.float32,shape=[adj.shape[0], num_features],
                                            name='real_distribution'),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'real_distribution': tf.placeholder(dtype=tf.float32, shape=[adj.shape[0], FLAGS.hidden2],
                                            name='real_distribution')

    }

    return placeholders


def get_model(model_str, placeholders, num_features, num_nodes, features_nonzero):
    # 计算图构建
    discriminator = Discriminator()
    D_Graph = D_graph(num_features)
    d_real = discriminator.construct(placeholders['real_distribution'])
    GD_real = D_Graph.construct(placeholders['features_dense'])
    model = None
    if model_str == 'arga_ae':
        model = GCN(placeholders, num_features, features_nonzero)

    elif model_str == 'DBGAN':
        model = GCN(placeholders, num_features, features_nonzero)
        model_z2g = Generator_z2g(placeholders, num_features,features_nonzero)

    return d_real, discriminator, model, model_z2g, D_Graph, GD_real


def format_data(data_name):
    # Load data

    adj, features, y_test, tx, ty, test_maks, true_labels = load_data(data_name)


    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    #删除对角线元素
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train
    adj_dense = adj.toarray()

    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]
    features_dense = features.tocoo().toarray()

    features = sparse_to_tuple(features.tocoo())
    #num_features是feature的维度
    num_features = features[2][1]
    #features_nonzero就是非零feature的个数
    features_nonzero = features[1].shape[0]

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    items = [
    adj, num_features, num_nodes, features_nonzero, 
    pos_weight, norm, adj_norm, adj_label, 
    features, true_labels, train_edges, val_edges, 
    val_edges_false, test_edges, test_edges_false, adj_orig,features_dense, adj_dense,features_dense
    ]
    feas = {}

    print('num_features is:',num_features)
    print('num_nodes is:',num_nodes)
    print('features_nonzero is:',features_nonzero)
    print('pos_weight is:',pos_weight)
    print('norm is:',norm)

    for item in items:
        #item_name = [ k for k,v in locals().iteritems() if v == item][0]
        feas[retrieve_name(item)] = item


    return feas

def get_optimizer(model_str, model, model_z2g, D_Graph, discriminator, placeholders, pos_weight, norm, d_real,num_nodes,GD_real):
    if model_str == 'arga_ae':
        output = model.construct()
        embeddings = output[0]
        reconstructions = output[1]
        d_fake = discriminator.construct(embeddings, reuse=True)
        opt = OptimizerAE(preds=reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm,
                          d_real=d_real,
                          d_fake=d_fake)
    elif model_str == 'DBGAN':
        z2g = model_z2g.construct()
        g2z = model.construct()
        embeddings = g2z[0]
        reconstructions = g2z[1]
        d_fake = discriminator.construct(embeddings, reuse=True)
        GD_fake = D_Graph.construct(z2g, reuse = True)
        print('----------------------------',tf.shape(placeholders['features']),'----------------------------')
        opt = OptimizerCycle(preds=reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm,
                          d_real=d_real,
                          d_fake=d_fake,
                            GD_real = GD_real,
                            GD_fake = GD_fake,
                            preds_z2g = placeholders['real_distribution'],
                            labels_z2g = placeholders['real_distribution'],
                            preds_cycle = model_z2g.construct(inputs = embeddings, hidden = None, reuse = True),
                            labels_cycle = placeholders['features_dense'])
    return opt

def update(model, opt, sess, adj_norm, adj_label, features, placeholders, adj, distribution, adj_dense):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['features_dense']: adj_dense})
    z_real_dist = np.random.randn(adj.shape[0], FLAGS.hidden2)
    z_real_dist = distribution.sample(adj.shape[0])
    feed_dict.update({placeholders['real_distribution']: z_real_dist})

    for j in range(5):
        _, reconstruct_loss = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        GG_loss,_ = sess.run([opt.generator_loss_z2g, opt.generator_optimizer_z2g], feed_dict=feed_dict)
    g_loss, _ = sess.run([opt.generator_loss, opt.generator_optimizer], feed_dict=feed_dict)
    d_loss, _ = sess.run([opt.dc_loss, opt.discriminator_optimizer], feed_dict=feed_dict)
    with tf.device("/gpu:3"):
        GD_loss,_ = sess.run([opt.GD_loss, opt.discriminator_optimizer_z2g], feed_dict=feed_dict)
        
        #GD_loss = sess.run(opt.GD_loss, feed_dict=feed_dict)
        #GG_loss = sess.run(opt.generator_loss_z2g, feed_dict=feed_dict)
    #g_loss, _ = sess.run([opt.generator_loss, opt.generator_optimizer], feed_dict=feed_dict)
    g_loss = sess.run(opt.generator_loss, feed_dict=feed_dict)
    d_loss = sess.run(opt.dc_loss, feed_dict=feed_dict)

    emb = sess.run(model.z_mean, feed_dict=feed_dict)
    avg_cost = [reconstruct_loss, d_loss, g_loss, GD_loss, GG_loss]

    return emb, avg_cost


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    print([var_name for var_name, var_val in callers_local_vars if var_val is var][-1])
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][-1]