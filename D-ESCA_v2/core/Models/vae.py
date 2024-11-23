from tensorflow.keras import layers, Model, regularizers, backend
from tensorflow import shape, exp, fill, math, square, reduce_mean


# sampling layer for reparameterization trick
class SamplingLayer(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_log_var = backend.clip(z_log_var, -100, 10)
        # batch = shape(z_mean)[0]
        # dim = shape(z_mean)[1]
        epsilon = backend.random_normal(shape=shape(z_mean))
        z = z_mean + exp(0.5*z_log_var)*epsilon
        return z


# take the mean of a batch layer
class BatchAverageLayer(layers.Layer):
    # def build(self, input_shape):
    #   self.dummy = tf.fill(input_shape, 1)

    def call(self, input):
        self.dummy = fill(shape(input), 1.0)
        return math.reduce_mean(input, axis=0)*self.dummy


# encoder model
class Encoder(Model):
    def __init__(self, latent_dim=32, intermediate_dim=[128], deep_net=True,
                 name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.intermediate_layer_num = len(intermediate_dim)
        self.layer_dict = {}

        for index, num in enumerate(intermediate_dim):
            self.layer_dict["layer_"+str(index+1)] = layers.Dense(units=num, activation='relu',    # noqa: E501
                                                                  kernel_regularizer=regularizers.l2(0.0001))    # noqa: E501

        self.deep_net = deep_net

        if deep_net:
            self.average_layer = BatchAverageLayer()

        self.mean_layer = layers.Dense(
            units=latent_dim, activation='relu',
            kernel_regularizer=regularizers.l2(0.0001)
        )
        self.variance_layer = layers.Dense(
            units=latent_dim, activation='relu',
            kernel_regularizer=regularizers.l2(0.0001)
        )

        self.sampling_layer = SamplingLayer()

    def call(self, input):
        x = input
        for i in range(self.intermediate_layer_num):
            x = self.layer_dict["layer_"+str(i+1)](x)

        if self.deep_net:
            x = self.average_layer(x)

        z_mean = self.mean_layer(x)
        z_log_var = self.variance_layer(x)

        z = self.sampling_layer((z_mean, z_log_var))

        return z, z_mean, z_log_var


# decoder model
class Decoder(Model):
    def __init__(self, original_dim, intermediate_dim=[128], name='decoder', **kwargs):    # noqa: E501
        super(Decoder, self).__init__(name=name, **kwargs)

        self.intermediate_layer_num = len(intermediate_dim)
        self.layer_dict = {}

        for index, num in enumerate(intermediate_dim):
            self.layer_dict["layer_"+str(index+1)] = layers.Dense(units=num, activation='relu',    # noqa: E501
                                                                  kernel_regularizer=regularizers.l2(0.0001))    # noqa: E501

        self.output_layer = layers.Dense(original_dim, activation='sigmoid',
                                         kernel_regularizer=regularizers.l2(0.0001))    # noqa: E501

    def call(self, input):
        x = input
        for i in range(self.intermediate_layer_num, 0, -1):
            x = self.layer_dict["layer_"+str(i)](x)

        output = self.output_layer(x)
        return output


# variation of vae model
# combine both encoder and decoder
class VariationalAutoEncoder(Model):
    def __init__(self, original_dim, intermediate_dim=[512, 512, 512, 128], latent_dim=32,    # noqa: E501
                 deep_net=False, name='vae', **kwargs):
        super(VariationalAutoEncoder, self).__init__(name='vae', **kwargs)
        self.encoder = Encoder(intermediate_dim=intermediate_dim, latent_dim=latent_dim, deep_net=deep_net)    # noqa: E501
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, input):
        z, z_mean, z_log_var = self.encoder(input)
        output = self.decoder(z)
        # using KL divergence estimation for Gaussian variable
        z_log_var = backend.clip(z_log_var, -100, 10)
        KL_loss = -0.5*reduce_mean(1 + z_log_var - square(z_mean) - exp(z_log_var))    # noqa: E501
        self.add_loss(KL_loss)
        return output
