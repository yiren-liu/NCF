from models import utils
from dataio import ImpDataset
import math
import tensorflow as tf
from models.evaluation import *
from time import time
from models.BaseModel import *

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights, biases


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer, weights, biases









class CCF(BaseModel):
    def __init__(self, args, num_users, num_items):
        BaseModel.__init__(self, args, num_users, num_items)
        self.layers = eval(args.layers)
        self.lambda_layers = eval(args.reg_layers)
        self.num_factors = args.num_factors

    def build_core_model(self, user_indices, item_indices):

        init_value = self.init_stddev

        emb_user = tf.Variable(tf.truncated_normal([self.num_users, self.num_factors],
                                                   stddev=init_value / math.sqrt(float(self.num_factors)), mean=0),
                               name='user_embedding', dtype=tf.float32)
        emb_item = tf.Variable(tf.truncated_normal([self.num_items, self.num_factors],
                                                   stddev=init_value / math.sqrt(float(self.num_factors)), mean=0),
                               name='item_embedding', dtype=tf.float32)

        emb_user_bias = tf.concat([emb_user, tf.ones((self.num_users, 1), dtype=tf.float32) * 0.1], 1,
                                  name='user_embedding_bias')
        emb_item_bias = tf.concat([tf.ones((self.num_items, 1), dtype=tf.float32) * 0.1, emb_item], 1,
                                  name='item_embedding_bias')

        user_feature = tf.nn.embedding_lookup(emb_user, user_indices, name='user_feature')
        item_feature = tf.nn.embedding_lookup(emb_item, item_indices, name='item_feature')

        user_feature = tf.reshape(user_feature, [-1, user_feature.get_shape().as_list()[1], 1])
        item_feature = tf.reshape(item_feature, [-1, 1, user_feature.get_shape().as_list()[1]])

        latent_matrix = tf.matmul(user_feature, item_feature)
        latent_matrix=tf.reshape(latent_matrix,[-1,latent_matrix.get_shape().as_list()[1],latent_matrix.get_shape().as_list()[2],1])


        model_params = [emb_user, emb_item]

        layer_conv1, weights_conv1, bias_conv1 = \
            new_conv_layer(input=latent_matrix,
                           num_input_channels=num_channels,
                           filter_size=filter_size1,
                           num_filters=num_filters1,
                           use_pooling=True)

        layer_conv2, weights_conv2, bias_conv2 = \
            new_conv_layer(input=layer_conv1,
                           num_input_channels=num_filters1,
                           filter_size=filter_size2,
                           num_filters=num_filters2,
                           use_pooling=True)

        layer_flat, num_features = flatten_layer(layer_conv2)

        layer_fc1, weights_fc1,bias_fc1 = new_fc_layer(input=layer_flat,
                                 num_inputs=num_features,
                                 num_outputs=fc_size,
                                 use_relu=True)

        model_params.extend([weights_conv1, bias_conv1, weights_conv2, bias_conv2, weights_fc1,bias_fc1])






        return layer_fc1, layer_fc1.shape[1]._value, model_params



    def build_model(self, user_indices=None, item_indices=None):

        if not user_indices:
            user_indices = tf.placeholder(tf.int32, [None])
        self.user_indices = user_indices
        if not item_indices:
            item_indices = tf.placeholder(tf.int32, [None])
        self.item_indices = item_indices

        self.ratings = tf.placeholder(tf.float32, [None])

        model_vector, model_len, model_params = self.build_core_model(user_indices, item_indices)

        self.output, self.loss, self.error, self.raw_error, self.train_step = self.build_train_model(model_vector,
                                                                                                     model_len,
                                                                                                     self.ratings,
                                                                                                     model_params)






