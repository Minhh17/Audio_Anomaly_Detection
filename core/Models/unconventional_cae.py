from tensorflow.keras import layers, Model, regularizers


# encoder model
class Encoder(Model):
    def __init__(self, kernel_size=(3, 3), stride=1, padding=True, name='encoder', **kwargs):    # noqa: E501
        super(Encoder, self).__init__(name=name, **kwargs)

        self.padding = 'same' if padding else 'valid'

        self.layer_1e = layers.Conv2D(128,
                                      (17, 1),
                                      strides=(3, 1),
                                      padding=self.padding,
                                      activation='relu',
                                      kernel_regularizer=regularizers.l2(0.0001))    # noqa: E501

        self.pooling_1e = layers.MaxPooling2D(pool_size=(1, 2))

        self.layer_2e = layers.Conv2D(128,
                                      (3, 3),
                                      strides=1,
                                      padding='same',
                                      activation='relu',
                                      kernel_regularizer=regularizers.l2(0.0001))    # noqa: E501

        self.pooling_2e = layers.MaxPooling2D(pool_size=(2, 2))

    def call(self, input):
        x = self.layer_1e(input)
        x = self.pooling_1e(x)
        x = self.layer_2e(x)
        x = self.pooling_2e(x)
        return x


# decoder model
class Decoder(Model):
    def __init__(self, original_dim=1, kernel_size=(3, 3), stride=1, padding=True,    # noqa: E501
                 name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.padding = 'same' if padding else 'valid'

        self.layer_1d = layers.Conv2DTranspose(128,
                                               (17, 1),
                                               strides=(3, 1),
                                               padding=self.padding,
                                               activation='relu',
                                               kernel_regularizer=regularizers.l2(0.0001))    # noqa:

        self.up_1d = layers.UpSampling2D(size=(1, 2))

        self.layer_2d = layers.Conv2DTranspose(128,
                                               (3, 3),
                                               strides=1,
                                               padding='same',
                                               activation='relu',
                                               kernel_regularizer=regularizers.l2(0.0001))    # noqa: E501

        self.up_2d = layers.UpSampling2D(size=(2, 2))

        self.output_layer = layers.Conv2DTranspose(original_dim,
                                                   (1, 1),
                                                   strides=stride,
                                                   padding=self.padding,
                                                   activation='sigmoid',
                                                   kernel_regularizer=regularizers.l2(0.0001))    # noqa: E501

    def call(self, input):
        x = self.up_2d(input)
        x = self.layer_2d(x)
        x = self.up_1d(x)
        x = self.layer_1d(x)
        output = self.output_layer(x)
        return output


# variation of vae model
# combine both encoder and decoder
class Unconven_CAE(Model):
    def __init__(self, original_dim=1, kernel_size=(3, 3), stride=1, padding=False,    # noqa: E501
                 name='cae', **kwargs):
        super(Unconven_CAE, self).__init__(name='unconven_cae', **kwargs)
        self.encoder = Encoder(kernel_size=kernel_size, stride=stride, padding=padding)    # noqa: E501
        self.decoder = Decoder(original_dim, kernel_size=kernel_size, stride=stride, padding=padding)    # noqa: E501

    def call(self, input):
        z = self.encoder(input)
        output = self.decoder(z)
        return output
