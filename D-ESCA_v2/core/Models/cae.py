from tensorflow.keras import layers, Model, regularizers


# encoder model
class Encoder(Model):
    def __init__(self, kernel_size=(3, 3), intermediate_dim=[16], stride=1, padding=True,    # noqa: E501
                 name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.intermediate_layer_num = len(intermediate_dim)
        self.layer_dict = {}

        self.padding = 'same' if padding else 'valid'

        self.initial_layer = layers.Conv2D(8,
                                           kernel_size,
                                           strides=stride,
                                           padding=self.padding,
                                           activation='relu',
                                           kernel_regularizer=regularizers.l2(0.0001))    # noqa: E501

        for index, num in enumerate(intermediate_dim):
            self.layer_dict["layer_"+str(index+1)] = layers.Conv2D(intermediate_dim[index],    # noqa: E501
                                                                   kernel_size,    # noqa: E501
                                                                   strides=stride,    # noqa: E501
                                                                   padding=self.padding,    # noqa: E501
                                                                   activation='relu',     # noqa: E501
                                                                   kernel_regularizer=regularizers.l2(0.0001))    # noqa: E501
            # self.layer_dict["batch_norm"+str(index+1)] = layers.BatchNormalization()    # noqa: E501
            self.layer_dict["pooling_"+str(index+1)] = layers.MaxPooling2D(pool_size=(2, 2))    # noqa: E501

    def call(self, input):
        x = self.initial_layer(input)
        for i in range(self.intermediate_layer_num):
            x = self.layer_dict["layer_"+str(i+1)](x)
            # x = self.layer_dict["batch_norm"+str(i+1)](x)
            x = self.layer_dict["pooling_"+str(i+1)](x)
        return x


# decoder model
class Decoder(Model):
    def __init__(self, original_dim=1, kernel_size=(3, ), intermediate_dim=[16], stride=1, padding=True,    # noqa: E501
                 name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.intermediate_layer_num = len(intermediate_dim)
        self.layer_dict = {}
        self.padding = 'same' if padding else 'valid'

        self.initial_layer = layers.Conv2DTranspose(intermediate_dim[-1]*2,
                                                    kernel_size,
                                                    strides=stride,
                                                    padding=self.padding,
                                                    activation='relu',
                                                    kernel_regularizer=regularizers.l2(0.0001))    # noqa: E501

        for index, num in enumerate(intermediate_dim):
            self.layer_dict["layer_"+str(index+1)] = layers.Conv2DTranspose(intermediate_dim[index],    # noqa: E501
                                                                            kernel_size,    # noqa: E501
                                                                            strides=stride,    # noqa: E501
                                                                            padding=self.padding,    # noqa: E501
                                                                            activation='relu',    # noqa: E501
                                                                            kernel_regularizer=regularizers.l2(0.0001))    # noqa: E501
            # self.layer_dict["batch_norm"+str(index+1)] = layers.BatchNormalization()    # noqa: E501
            self.layer_dict["upsampling_"+str(index+1)] = layers.UpSampling2D(size=(2, 2))    # noqa: E501

        self.conv_8 = layers.Conv2DTranspose(8,    # noqa: E501
                                             kernel_size,    # noqa: E501
                                             strides=stride,    # noqa: E501
                                             padding=self.padding,    # noqa: E501
                                             activation='relu',    # noqa: E501
                                             kernel_regularizer=regularizers.l2(0.0001))    # noqa: E501

        self.output_layer = layers.Conv2DTranspose(original_dim,
                                                   kernel_size,
                                                   strides=stride,
                                                   padding=self.padding,
                                                   activation='sigmoid',
                                                   kernel_regularizer=regularizers.l2(0.0001))    # noqa: E501

    def call(self, input):
        x = self.initial_layer(input)
        # x = input

        for i in range(self.intermediate_layer_num, 0, -1):
            x = self.layer_dict["layer_"+str(i)](x)
            # x = self.layer_dict["batch_norm"+str(i)](x)
            x = self.layer_dict["upsampling_"+str(i)](x)

        x = self.conv_8(x)
        output = self.output_layer(x)
        return output


# variation of vae model
# combine both encoder and decoder
class ConvolutionalAutoEncoder(Model):
    def __init__(self, original_dim=1, kernel_size=(3, 3), intermediate_dim=[16], stride=1, padding=True,    # noqa: E501
                 name='cae', **kwargs):
        super(ConvolutionalAutoEncoder, self).__init__(name='cae', **kwargs)
        self.encoder = Encoder(intermediate_dim=intermediate_dim, kernel_size=kernel_size, stride=stride, padding=padding)    # noqa: E501
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim, kernel_size=kernel_size, stride=stride, padding=padding)    # noqa: E501

    def call(self, input):
        z = self.encoder(input)
        output = self.decoder(z)
        return output
