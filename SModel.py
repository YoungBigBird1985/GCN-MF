from layers import *


flags = tf.app.flags
FLAGS = flags.FLAGS


class SModel(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.user_layers = []
        self.item_layers = []
        self.user_activations = []
        self.itempos_activations = []
        self.itemneg_activations = []
        #
        # self.user_history = None
        # self.item_history = None

        self.user_inputs = None
        self.itempos_inputs = None
        self.itemneg_inputs = None

        self.user_outputs = None
        self.itempos_outputs = None
        self.itemneg_outputs = None
        self.outputs_result = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

        self.train_op = None
        self.test_op = None

        self.weight_loss = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build_history()
            self._build()

        # Build sequential layer model
        self.user_activations.append(self.user_inputs)
        for layer in self.user_layers:
            hidden = layer(self.user_activations[-1])
            self.user_activations.append(hidden)
        self.user_outputs = self.user_activations[-1]

        self.itempos_activations.append(self.itempos_inputs)
        for layer in self.item_layers:
            hidden = layer(self.itempos_activations[-1])
            self.itempos_activations.append(hidden)
        self.itempos_outputs = self.itempos_activations[-1]

        #
        self.update_history = []




        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = [self.optimizer.minimize(self.loss)]
        self.train_op = []
        with tf.control_dependencies(self.opt_op):
            self.train_op = tf.group(*self.update_history)
        self.test_op = tf.group(*self.update_history)

    def predict(self):
        pass

    def _build_history(self):
        pass


    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)




class SNIMC_MF(SModel): #mixture of dense and gcn
    def __init__(self, placeholders, user_length, item_length, user_input_dim, item_input_dim, **kwargs):
        super(SNIMC_MF, self).__init__(**kwargs)
        self.user_inputs = placeholders['user_AXfeatures']# A*X for the bottom layer, not original feature X
        self.itempos_inputs = placeholders['itempos_AXfeatures']

        self.labels = placeholders['labels']
        self.labelweight = placeholders['label_weights']

        self.user_length = user_length
        self.item_length = item_length
        self.user_input_dim = user_input_dim
        self.item_input_dim = item_input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = FLAGS.embedding_size
        self.placeholders = placeholders
        #self.support = placeholders['support']
        #self.loss_decay = placeholders['loss_decay']

        #self.weight_loss = placeholders['weight_loss']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)



        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.user_layers[0].vars.values():
            self.loss += FLAGS.weight_decay  * tf.nn.l2_loss(var)

        for var in self.item_layers[0].vars.values():
            self.loss += FLAGS.weight_decay  * tf.nn.l2_loss(var)

        self.outputs_result = tf.clip_by_value(tf.matmul(self.user_outputs, self.itempos_outputs, transpose_b=True), 0, 1)
        self.loss += tf.reduce_sum(
            tf.multiply(tf.squared_difference(self.outputs_result, self.labels), self.labelweight))

        #self.outputs_result = tf.reduce_sum(tf.multiply(self.user_outputs, (self.itempos_outputs - self.itemneg_outputs)), 1, keep_dims=True)

        #self.loss += -tf.reduce_mean(tf.log(tf.sigmoid(self.outputs_result)))
        #self.loss += - tf.reduce_mean(tf.multiply(tf.log(tf.sigmoid(self.outputs_result)), self.weight_loss))



    def _accuracy(self):
        self.accuracy = tf.reduce_mean(tf.to_float(self.outputs_result > 0))

    def _build(self):
        self.user_layers.append(InductiveUser(input_dim=self.user_input_dim,
                                 output_dim=FLAGS.user_hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=False,
                                 logging=self.logging))


        self.item_layers.append(InductiveItem(input_dim=self.item_input_dim,
                                 output_dim=FLAGS.item_hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=False,
                                 logging=self.logging))


    def predict(self):
        return tf.nn.softmax(self.outputs)

    def _build_history(self):
        # Create history after each aggregation
        return




class SGCN_MF_Muti(SModel):
    def __init__(self, placeholders, user_length, item_length, user_input_dim, item_input_dim, **kwargs):
        super(SGCN_MF_Muti, self).__init__(**kwargs)
        self.user_inputs = placeholders['user_AXfeatures']# A*X for the bottom layer, not original feature X
        self.itempos_inputs = placeholders['itempos_AXfeatures']

        self.labels = placeholders['labels']
        self.labelweight = placeholders['label_weights']

        self.user_length = user_length
        self.item_length = item_length
        self.user_input_dim = user_input_dim
        self.item_input_dim = item_input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = FLAGS.embedding_size
        self.placeholders = placeholders
        self.support = placeholders['support']
        #self.loss_decay = placeholders['loss_decay']

        #self.weight_loss = placeholders['weight_loss']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)



        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.user_layers[0].vars.values():
            self.loss += FLAGS.weight_decay  * tf.nn.l2_loss(var)

        for var in self.item_layers[0].vars.values():
            self.loss += FLAGS.weight_decay  * tf.nn.l2_loss(var)

        self.outputs_result = tf.clip_by_value(tf.matmul(self.user_outputs, self.itempos_outputs, transpose_b=True), 0, 1)
        self.loss += tf.reduce_sum( tf.multiply(tf.squared_difference(self.outputs_result, self.labels), self.labelweight) )

        #self.outputs_result = tf.reduce_sum(tf.multiply(self.user_outputs, (self.itempos_outputs - self.itemneg_outputs)), 1, keep_dims=True)

        #self.loss += -tf.reduce_mean(tf.log(tf.sigmoid(self.outputs_result)))
        #self.loss += - tf.reduce_mean(tf.multiply(tf.log(tf.sigmoid(self.outputs_result)), self.weight_loss))



    def _accuracy(self):
        self.accuracy = tf.reduce_mean(tf.to_float(self.outputs_result > 0))

    def _build(self):
        self.user_layers.append(InductiveUser(input_dim=self.user_input_dim,
                                 output_dim=FLAGS.user_hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=False,
                                 sparse_inputs=False,
                                 logging=self.logging))

        self.user_layers.append(GraphConvolution_Flag(input_dim=FLAGS.user_hidden1,
                                                      output_dim=self.output_dim,
                                                      placeholders=self.placeholders,
                                                      support=self.support,
                                                      act=tf.nn.relu,
                                                      dropout=False,
                                                      flag='user',
                                                      logging=self.logging))


        self.item_layers.append(InductiveItem(input_dim=self.item_input_dim,
                                 output_dim=FLAGS.item_hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=False,
                                 sparse_inputs=False,
                                 logging=self.logging))

        self.item_layers.append(GraphConvolution_Flag(input_dim=FLAGS.user_hidden1,
                                                      output_dim=self.output_dim,
                                                      placeholders=self.placeholders,
                                                      support=self.support,
                                                      act=tf.nn.relu,
                                                      dropout=False,
                                                      flag='item_pos',
                                                      logging=self.logging))


    def predict(self):
        return tf.nn.softmax(self.outputs)

    def _build_history(self):
        # Create history after each aggregation
        return

class SGCN_MF(SModel): #mixture of dense and gcn
    def __init__(self, placeholders, user_length, item_length, user_input_dim, item_input_dim, **kwargs):
        super(SGCN_MF, self).__init__(**kwargs)
        self.user_inputs = placeholders['user_AXfeatures']# A*X for the bottom layer, not original feature X
        self.itempos_inputs = placeholders['itempos_AXfeatures']

        self.labels = placeholders['labels']
        self.labelweight = placeholders['label_weights']

        self.user_length = user_length
        self.item_length = item_length
        self.user_input_dim = user_input_dim
        self.item_input_dim = item_input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = FLAGS.embedding_size
        self.placeholders = placeholders
        #self.support = placeholders['support']
        #self.loss_decay = placeholders['loss_decay']

        #self.weight_loss = placeholders['weight_loss']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)



        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.user_layers[0].vars.values():
            self.loss += FLAGS.weight_decay  * tf.nn.l2_loss(var)

        for var in self.item_layers[0].vars.values():
            self.loss += FLAGS.weight_decay  * tf.nn.l2_loss(var)

        self.outputs_result = tf.clip_by_value(tf.matmul(self.user_outputs, self.itempos_outputs, transpose_b=True), 0, 1)
        self.loss += tf.reduce_sum(
            tf.multiply(tf.squared_difference(self.outputs_result, self.labels), self.labelweight))

        #self.outputs_result = tf.reduce_sum(tf.multiply(self.user_outputs, (self.itempos_outputs - self.itemneg_outputs)), 1, keep_dims=True)

        #self.loss += -tf.reduce_mean(tf.log(tf.sigmoid(self.outputs_result)))
        #self.loss += - tf.reduce_mean(tf.multiply(tf.log(tf.sigmoid(self.outputs_result)), self.weight_loss))



    def _accuracy(self):
        self.accuracy = tf.reduce_mean(tf.to_float(self.outputs_result > 0))

    def _build(self):
        self.user_layers.append(InductiveUser(input_dim=self.user_input_dim,
                                 output_dim=FLAGS.user_hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=False,
                                 logging=self.logging))


        self.item_layers.append(InductiveItem(input_dim=self.item_input_dim,
                                 output_dim=FLAGS.item_hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=False,
                                 logging=self.logging))


    def predict(self):
        return tf.nn.softmax(self.outputs)

    def _build_history(self):
        # Create history after each aggregation
        return





