# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorlayer.layers import *
import tensorlayer as tl


"""Contains the base class for models."""
class BaseModel(object):
    """Inherit from this class when implementing new models."""

    def create_model(self, **unused_params):
        """Define variables of the model."""
        raise NotImplementedError()

    def run_model(self, unused_model_input, **unused_params):
        """Run model with given input."""
        raise NotImplementedError()

    def get_variables(self):
        """Return all variables used by the model for training."""
        raise NotImplementedError()



class DCGANGenerator(BaseModel):
    def __init__(self):
        self.noise_input_size = 100

    def create_model(self, output_size, **unused_params):
        pass

    def run_model(self, model_input, is_training=True, reuse=None, **unused_params):
        image_size = 64
        s2, s4, s8, s16 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16)
        gf_dim = 64 # Dimension of gen filters in first conv layer. [64]
        c_dim = 1 # n_color 1
        batch_size = 128 # 64
        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        inputs = tf.reshape(model_input, [-1, 100])
        with tf.variable_scope("generator", reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            net_in = InputLayer(inputs, name='g/in')
            net_h0 = DenseLayer(net_in, n_units=gf_dim*8*s16*s16, W_init=w_init,
                                act = tf.identity, name='g/h0/lin')
            net_h0 = ReshapeLayer(net_h0, shape=[-1, s16, s16, gf_dim*8], name='g/h0/reshape')
            net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_training,
                                    gamma_init=gamma_init, name='g/h0/batch_norm')

            net_h1 = DeConv2d(net_h0, gf_dim*4, (5, 5), out_size=(s8, s8), strides=(2, 2),
                            padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h1/decon2d')
            net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_training,
                                    gamma_init=gamma_init, name='g/h1/batch_norm')

            net_h2 = DeConv2d(net_h1, gf_dim*2, (5, 5), out_size=(s4, s4), strides=(2, 2),
                            padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h2/decon2d')
            net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_training,
                                    gamma_init=gamma_init, name='g/h2/batch_norm')

            net_h3 = DeConv2d(net_h2, gf_dim, (5, 5), out_size=(s2, s2), strides=(2, 2),
                            padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h3/decon2d')
            net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_training,
                                    gamma_init=gamma_init, name='g/h3/batch_norm')

            net_h4 = DeConv2d(net_h3, c_dim, (5, 5), out_size=(image_size, image_size), strides=(2, 2),
                            padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h4/decon2d')
            logits = net_h4.outputs
            outputs = tf.nn.tanh(net_h4.outputs)

        return {"output": tf.image.resize_nearest_neighbor(outputs, [50, 50])}

    def get_variables(self):
        print(tl.layers.get_variables_with_name('generator', True, True))
        return tl.layers.get_variables_with_name('generator', True, True)


class DCGANDiscriminator(BaseModel):
    def create_model(self, input_size, **unused_params):
        pass

    def run_model(self, model_input, is_training=True, reuse=None, **unused_params):
        df_dim = 64 # Dimension of discrim filters in first conv layer. [64]
        c_dim = 1 # n_color 3
        batch_size = 128 # 64
        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        model_input = tf.reshape(model_input, [-1, 50, 50, 1])
        inputs = model_input
        inputs = tf.image.resize_nearest_neighbor(model_input, [64, 64])
        with tf.variable_scope("discriminator", reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            net_in = InputLayer(inputs, name='d/in')
            net_h0 = Conv2d(net_in, df_dim, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                            padding='SAME', W_init=w_init, name='d/h0/conv2d')

            net_h1 = Conv2d(net_h0, df_dim*2, (5, 5), (2, 2), act=None,
                            padding='SAME', W_init=w_init, name='d/h1/conv2d')
            net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                                    is_train=is_training, gamma_init=gamma_init, name='d/h1/batch_norm')

            net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
                            padding='SAME', W_init=w_init, name='d/h2/conv2d')
            net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                                    is_train=is_training, gamma_init=gamma_init, name='d/h2/batch_norm')

            net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), act=None,
                            padding='SAME', W_init=w_init, name='d/h3/conv2d')
            net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                                    is_train=is_training, gamma_init=gamma_init, name='d/h3/batch_norm')

            net_h4 = FlattenLayer(net_h3, name='d/h4/flatten')
            net_h4 = DenseLayer(net_h4, n_units=1, act=tf.identity,
                                W_init = w_init, name='d/h4/lin_sigmoid')
            logits = net_h4.outputs
            predictions = tf.nn.sigmoid(net_h4.outputs)

        return {'predictions': predictions, 'logits': logits}

    def get_variables(self):
        return tl.layers.get_variables_with_name('discriminator', True, True)

