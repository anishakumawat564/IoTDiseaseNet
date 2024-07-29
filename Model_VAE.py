import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import *
from keras.models import *
from keras import backend as K

from Evaluation import evaluation

class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, mask=True, alpha=1, beta=1, gamma=1, dtype=tf.float64):
        super(CustomLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE)
        self.mask = mask
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dtype = dtype

    def call(self, y_true, y_pred):
        def loss_fn(y_true, y_pred, mask):
            y_true = tf.boolean_mask(y_true, mask)
            y_pred = tf.boolean_mask(y_pred, mask)
            return tf.keras.losses.MSE(y_true, y_pred)

        self.mask = tf.not_equal(y_true, 0.)

        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)

        y_pred = tf.multiply(y_pred, tf.cast(self.mask, dtype=self.dtype))
        y_pred_cum = tf.math.cumsum(y_pred, axis=1)
        y_pred_cum = tf.multiply(y_pred_cum, tf.cast(self.mask, dtype=self.dtype))

        y_true_cum = tf.math.cumsum(y_true, axis=1)
        y_true_cum = tf.multiply(y_true_cum, tf.cast(self.mask, dtype=self.dtype))

        loss_value = self.alpha * loss_fn(y_true, y_pred, self.mask) + \
                     self.gamma * loss_fn(y_true_cum, y_pred_cum, self.mask)

        return loss_value

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
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
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
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


def Model_VAE(x_train, y_train, x_test, y_test, sol):
    latent_dim = 2

    encoder_inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(sol[0].astype('int'), 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()



    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    vae = Sequential()
    # model.add(Embedding(n_unique_words, 128, input_length=maxlen))
    vae.add(Dropout(0.5))
    vae.add(Dense(1, activation='sigmoid'))
    vae.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    vae.fit(x_train, y_train.astype('float'), epochs=sol[0], batch_size=128)
    pred = vae.predict(x_test)
    Eval = evaluation(pred.astype('int'), y_test)
    return np.asarray(Eval).ravel()
