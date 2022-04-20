# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
     
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    

def encoder(latent_dim, input_dim):
    encoder_inputs = keras.Input(shape=(input_dim))
    x1 = layers.Dense(48, kernel_initializer='random_normal', bias_initializer='zeros')(encoder_inputs)
    x2 = layers.Dense(28, kernel_initializer='random_normal', bias_initializer='zeros')(x1)
    x3 = layers.Dense(16, kernel_initializer='random_normal', bias_initializer='zeros')(x2)
    z_mean = layers.Dense(latent_dim, kernel_initializer='random_normal', bias_initializer='zeros', name="z_mean")(x3)
    z_log_var = layers.Dense(latent_dim, kernel_initializer='random_normal', bias_initializer='zeros', name="z_log_var")(x3)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model(encoder_inputs, [x1, x2, x3, z_mean, z_log_var, z], name="encoder")

def decoder(latent_dim, output_dim):
    latent_inputs = keras.Input(shape=(latent_dim, ))
    x3 = layers.Dense(16, kernel_initializer='random_normal', bias_initializer='zeros')(latent_inputs)
    x2 = layers.Dense(28, kernel_initializer='random_normal', bias_initializer='zeros')(x3)
    x1 = layers.Dense(48, kernel_initializer='random_normal', bias_initializer='zeros')(x2)
    decoder_outputs = layers.Dense(output_dim, kernel_initializer='random_normal', bias_initializer='zeros')(x1)
    return keras.Model(latent_inputs, [x1, x2, x3, decoder_outputs], name="decoder")
 
class LVAE(keras.Model):
    def __init__(self, latent_dim, fDim, **kwargs):
        super(LVAE, self).__init__(**kwargs)
        self.encoder = encoder(latent_dim, fDim)
        self.decoder = decoder(latent_dim, fDim)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
     
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
 
    def train_step(self, data):
        with tf.GradientTape() as tape:
            x1, x2, x3, z_mean, z_log_var, z = self.encoder(data)
            _x1, _x2, _x3, reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.clip_by_value(tf.nn.sigmoid_cross_entropy_with_logits(x1, _x1), 
                      1e-8,
                      tf.reduce_max(tf.nn.sigmoid_cross_entropy_with_logits(x1, _x1))
                    )
                ) +
                tf.reduce_sum(
                    tf.clip_by_value(tf.nn.sigmoid_cross_entropy_with_logits(x2, _x2), 
                      1e-8,
                      tf.reduce_max(tf.nn.sigmoid_cross_entropy_with_logits(x2, _x2))
                    )
                ) +
                tf.reduce_sum(
                    tf.clip_by_value(tf.nn.sigmoid_cross_entropy_with_logits(x3, _x3), 
                      1e-8,
                      tf.reduce_max(tf.nn.sigmoid_cross_entropy_with_logits(x3, _x3))
                    )
                ) +
                tf.reduce_sum(
                    tf.clip_by_value(tf.nn.sigmoid_cross_entropy_with_logits(data, reconstruction), 
                      1e-8,
                      tf.reduce_max(tf.nn.sigmoid_cross_entropy_with_logits(data, reconstruction))
                    )
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }
        
    def transferESpace(self, data):
        _, _, _, _, _, z = self.encoder.predict(data)
        _, _, _, reconstruction = self.decoder(z)
        r = np.array(
            [
              [distance.euclidean(i, j), distance.cosine(i, j)]
              for i, j in zip(tf.convert_to_tensor(data), reconstruction)])
        
        c = np.concatenate([z, r], axis=1)
        return c