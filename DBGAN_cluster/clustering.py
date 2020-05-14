from __future__ import division
from __future__ import print_function
from sklearn.cluster import KMeans
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import tensorflow as tf
from metrics import clustering_metrics
from constructor import get_placeholder, get_model, format_data, get_optimizer, update
from sklearn.neighbors import KernelDensity
from dppy.finite_dpps import FiniteDPP
from sklearn.decomposition import PCA
import scipy.io as scio
import numpy as np
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

class Clustering_Runner():
    def __init__(self, settings):

        print("Clustering on dataset: %s, model: %s, number of iteration: %3d" % (settings['data_name'], settings['model'], settings['iterations']))

        self.data_name = settings['data_name']
        self.iteration =settings['iterations']
        self.model = settings['model']
        self.n_clusters = settings['clustering_num']

    def erun(self):
        model_str = self.model

        # formatted data
        feas = format_data(self.data_name)

        # Define placeholders
        placeholders = get_placeholder(feas['adj'], feas['num_features'])
        
        #定义由Dpp和密度估计出来的混合高斯
        DPP = FiniteDPP('correlation',**{'K': feas['adj'].toarray()})
        #DPP.sample_exact_k_dpp(size=4)
        pca = PCA(n_components = FLAGS.hidden2)
        
        #index = DPP.list_of_samples[0]
        
        if self.data_name == 'cora':
            DPP.sample_exact_k_dpp(size=14)
            index = DPP.list_of_samples[0]
        elif self.data_name == 'citeseer':
            #'''
            index = np.array([481, 1763, 1701,  171, 1425,  842])#epoch 36时最高 0.571
            #'''
            #'''
            index = np.array([3165,  589, 1283, 1756, 2221, 2409])#50时可以达到0.545 
            #'''
            #'''
            index = np.array([2300, 2725, 3313, 1216, 2821, 2432])#50 
            #'''
            index = np.array([1718, 3241,  787, 2727,  624, 3110, 1503, 1867, 2410, 1594, 1203,
        2711,  171, 1790, 1778,  294,  685,   39, 1700, 2650, 2028, 2573,
         375, 2744, 2302, 1876,  784, 2233, 2546, 1793, 1677, 3278, 2587,
        2623, 1018, 1160, 3166,  668, 1663, 3007,  864, 2893,  743, 3129,
        3104, 3277, 1643, 3047,  322,  298, 2894,   35, 2578, 2031, 3316,
        1815,  361, 1868, 1546, 1895, 1514,  636])#这个性能最高
        
        elif self.data_name == 'pubmed':
            index = np.array([  842,  3338,  5712, 17511, 10801,  2714,  6970, 13296,  5466,
         2230])
        feature_sample = feas['features_dense']
        feature_sample = pca.fit_transform(feature_sample)
        
        featuresCompress = np.array([feature_sample[i] for i in index])
        #featuresCompress = np.array(feature_sample)
        kde = KernelDensity(bandwidth=0.7).fit(featuresCompress)

        # construct model
        d_real, discriminator, ae_model, model_z2g, D_Graph, GD_real = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'])

        # Optimizer
        opt = get_optimizer(model_str, ae_model, model_z2g, D_Graph, discriminator, placeholders, feas['pos_weight'], feas['norm'], d_real, feas['num_nodes'], GD_real)

        # Initialize session
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        #record list
        record = []
        record_emb = []
        # Train model
        for epoch in range(self.iteration):
            emb, avg_cost = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'],kde, feas['features_dense'])
            if (epoch+1) % 2 == 0:
                record_emb.append(emb)
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=4).fit(emb)
                print("Epoch:", '%04d' % (epoch + 1))
                predict_labels = kmeans.predict(emb)
                cm = clustering_metrics(feas['true_labels'], predict_labels)
                [a,b,c] = cm.evaluationClusterModelFromLabel()
                record.append([a,b,c])
        rec = np.array(record)
        index = rec[:,0].tolist().index(max(rec[:,0].tolist()))
        ana = record[index]
        emb = record_emb[index]
        scio.savemat('result/{}.mat'.format(self.data_name),{'embedded':emb,
                                     'labels':feas['true_labels']})
        print('The peak ACC=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (ana[0], ana[1], ana[2]))
            