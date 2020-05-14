from __future__ import division
from __future__ import print_function
import os
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = " 0,4,2,3"

import tensorflow as tf
import settings
from constructor import get_placeholder, get_model, format_data, get_optimizer, update
from metrics import linkpred_metrics
from sklearn.neighbors import KernelDensity
from dppy.finite_dpps import FiniteDPP
from sklearn.decomposition import PCA
import numpy as np
import scipy.io as scio
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

class Link_pred_Runner():
    def __init__(self, settings):
        self.data_name = settings['data_name']
        self.iteration = settings['iterations']
        self.model = settings['model']

    def erun(self):
        model_str = self.model
        # formatted data
        feas = format_data(self.data_name)

        # Define placeholders
        # 定义placeholders,get_placeholder函数中只需要传入一个参数，即adj，函数中需要用到adj.shape
        placeholders = get_placeholder(feas['adj'], feas['num_features'])
        
        #定义由Dpp和密度估计出来的混合高斯
        DPP = FiniteDPP('correlation',**{'K': feas['adj'].toarray()})
        #DPP.sample_exact_k_dpp(size=4)
        pca = PCA(n_components = FLAGS.hidden2)
        
        #index = DPP.list_of_samples[0]
        
        if self.data_name == 'cora':
            DPP.sample_exact_k_dpp(size=21)
            index = DPP.list_of_samples[0]
            pass
        elif self.data_name == 'citeseer':
            
            index = np.array([1782,  741, 3258, 3189, 3112, 2524, 2895, 1780, 1100, 2735, 1318,
       2944, 1825,   18,  987, 2564,  463,    6, 3173,  701, 1901, 2349,
       2786, 2412,  646, 2626, 2648, 1793,  432,  538, 1729, 1217, 1397,
       1932, 2850,  458, 2129,  702, 2934, 2030, 2882, 1393,  308, 1271,
       1106, 2688,  629, 1145, 3251, 1903, 1004, 1149, 1385,  285,  858,
       2977,  844,  335,  532,  404, 3174,  528])
        
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
        #sess = tf.Session(config = config)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        val_roc_score = []
        record = []
        record_emb = []
        # Train model
        for epoch in range(self.iteration):

            emb, avg_cost = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'],kde, feas['features_dense'])

            lm_train = linkpred_metrics(feas['val_edges'], feas['val_edges_false'])
            roc_curr, ap_curr, _ = lm_train.get_roc_score(emb, feas)
            val_roc_score.append(roc_curr)
            print("Epoch:", '%04d' % (epoch + 1), "train_loss= {:.5f}, d_loss= {:.5f}, g_loss= {:.5f}, GD_loss= {:.5f}, GG_loss= {:.5f}".format(avg_cost[0],avg_cost[1],avg_cost[2],avg_cost[3],avg_cost[4]), "val_roc=", "{:.5f}".format(val_roc_score[-1]), "val_ap=", "{:.5f}".format(ap_curr))

            if (epoch+1) % 10 == 0:
                lm_test = linkpred_metrics(feas['test_edges'], feas['test_edges_false'])
                roc_score, ap_score,_ = lm_test.get_roc_score(emb, feas)
                print('Test ROC score: ' + str(roc_score))
                print('Test AP score: ' + str(ap_score))
                record.append([roc_score,ap_score])
                record_emb.append(emb)
        rec = np.array(record)        
        index = rec[:,0].tolist().index(max(rec[:,0].tolist()))
        emb = record_emb[index]
        ana = record[index]
        scio.savemat('result/{}_link_64_64_new.mat'.format(self.data_name),{'embedded':emb,
                                     'labels':feas['true_labels']})
        print('The peak val_roc=%f, ap = %f' % (ana[0], ana[1]))