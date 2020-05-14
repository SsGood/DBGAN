import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm, d_real, d_fake):
        preds_sub = preds
        labels_sub = labels

        self.real = d_real

        # Discrimminator Loss
        self.dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.real), logits=self.real,name='dclreal'))

        self.dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake,name='dcfake'))
        self.dc_loss = self.dc_loss_fake + self.dc_loss_real

        # Generator loss
        generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake, name='gl'))


        # pos_weight，允许人们通过向上或向下加权相对于负误差的正误差的成本来权衡召回率和精确度
        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.generator_loss = generator_loss + self.cost


        all_variables = tf.trainable_variables()
        dc_var = [var for var in all_variables if 'dc_' in var.name]
        en_var = [var for var in all_variables if 'e_' in var.name]

      
        with tf.variable_scope(tf.get_variable_scope()):
            self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                             beta1=0.9, name='adam1').minimize(self.dc_loss, var_list=dc_var) #minimize(dc_loss_real, var_list=dc_var)

            self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                         beta1=0.9, name='adam2').minimize(self.generator_loss, var_list=en_var)


        # 值得注意的是，这个地方，除了对抗优化之外，
        # 还单纯用cost损失又优化了一遍，
        # 待会儿看训练的时候注意看是在哪部分进行的这部分优化操作
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)


class OptimizerCycle(object):
    def __init__(self, preds, labels, pos_weight, norm, d_real, d_fake, GD_real, GD_fake, preds_z2g, labels_z2g, preds_cycle,labels_cycle):
        
        preds_sub = preds
        labels_sub = labels

        self.real = d_real

        # Discrimminator Loss
        self.dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.real), logits=self.real,name='dclreal'))
        self.dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake,name='dcfake'))
        with tf.device("/gpu:2"):
            self.GD_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(GD_real), logits=GD_real,name='GD_real'))
            self.GD_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(GD_fake), logits=GD_fake,name='GD_fake'))
        
        self.dc_loss = self.dc_loss_fake + self.dc_loss_real
        self.GD_loss = self.GD_loss_fake + self.GD_loss_real

        # Generator loss
        generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake, name='gl'))
        generator_loss_z2g = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(GD_fake), logits=GD_fake, name='G_z2g'))

        # pos_weight，允许人们通过向上或向下加权相对于负误差的正误差的成本来权衡召回率和精确度
        with tf.device("/gpu:2"):
            self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        
        
        cost_cycle = norm * tf.reduce_mean(tf.square(preds_cycle - labels_cycle))

        cost_z2g = norm * tf.reduce_mean(tf.square(preds_z2g-labels_z2g))
        #with tf.device("/gpu:1"):
        #self.cost = 0.0000001 * self.cost + cost_cycle #for citeseer 
        self.cost = 0.01 * self.cost + cost_cycle # for cora
        self.generator_loss = generator_loss + self.cost
        self.generator_loss_z2g = generator_loss_z2g


        all_variables = tf.trainable_variables()
        dc_var = [var for var in all_variables if 'dc_' in var.name]
        en_var = [var for var in all_variables if 'e_' in var.name]
        GG_var = [var for var in all_variables if 'GG' in var.name]
        GD_var = [var for var in all_variables if 'GD' in var.name]

      
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device("/gpu:3"):
                self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                                 beta1=0.9, name='adam1').minimize(self.dc_loss, var_list=dc_var) #minimize(dc_loss_real, var_list=dc_var)

                self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                             beta1=0.9, name='adam2').minimize(self.generator_loss, var_list=en_var)
            
            with tf.device("/gpu:3"):
                self.discriminator_optimizer_z2g = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                                 beta1=0.9, name='adam1').minimize(self.GD_loss, var_list=GD_var)

                self.generator_optimizer_z2g = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                             beta1=0.9, name='adam2').minimize(self.generator_loss_z2g, var_list=GG_var)


        # 值得注意的是，这个地方，除了对抗优化之外，
        # 还单纯用cost损失又优化了一遍，
        # 待会儿看训练的时候注意看是在哪部分进行的这部分优化操作
        with tf.device("/gpu:3"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)
            #self.grads_vars = self.optimizer.compute_gradients(self.cost)
        
        #self.optimizer_z2g = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        #self.opt_op_z2g = self.optimizer.minimize(cost_z2g)
        #self.grads_vars_z2g = self.optimizer.compute_gradients(cost_z2g)
