import tensorflow as tf
import pandas as pd
import numpy as np
import plotly.express as px
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class PriceAutoencoder(Model):
    """Autoencoder for tabular price vectors (default: six economic features)."""

    def __init__(self, input_dim=6, latent_dim=2, hidden_dims=(32, 16)):
        super().__init__()
        encoder_layers = [layers.Input(shape=(input_dim,))]
        for units in hidden_dims:
            encoder_layers.append(layers.Dense(units, activation='relu'))
        encoder_layers.append(layers.Dense(latent_dim, activation='relu'))
        self.encoder = tf.keras.Sequential(encoder_layers)

        decoder_layers = [layers.Input(shape=(latent_dim,))]
        for units in reversed(hidden_dims):
            decoder_layers.append(layers.Dense(units, activation='relu'))
        decoder_layers.append(layers.Dense(input_dim, activation='linear'))
        self.decoder = tf.keras.Sequential(decoder_layers)

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded