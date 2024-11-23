from tensorflow.keras import layers, Model, regularizers
from tensorflow import random_uniform_initializer, Variable, shape, reshape, one_hot, matmul, reduce_mean, stop_gradient, reduce_sum, argmin   # noqa: E501


# encoder model
class Encoder(Model):
    def __init__(self, kernel_size=(3, 3), intermediate_dim=[32, 64], latent=64, stride=1, padding=True,    # noqa: E501
                 name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.intermediate_layer_num = len(intermediate_dim)
        self.layer_dict = {}

        self.padding = 'same' if padding else 'valid'

        for index, num in enumerate(intermediate_dim):
            self.layer_dict["layer_"+str(index+1)] = layers.Conv2D(intermediate_dim[index],    # noqa: E501
                                                                   kernel_size,    # noqa: E501
                                                                   strides=stride,    # noqa: E501
                                                                   padding=self.padding,    # noqa: E501
                                                                   activation='relu',     # noqa: E501
                                                                   kernel_regularizer=regularizers.l2(0.0001))    # noqa: E501
            # self.layer_dict["pooling_"+str(index+1)] = layers.MaxPooling2D(pool_size=(2, 2))    # noqa: E501

        self.latent_dim = layers.Conv2D(latent,    # noqa: E501
                                        kernel_size,    # noqa: E501
                                        strides=stride,    # noqa: E501
                                        padding=self.padding,    # noqa: E501
                                        activation='relu',     # noqa: E501
                                        kernel_regularizer=regularizers.l2(0.0001))    # noqa: E501

    def call(self, input):
        x = input
        # print("Number of layers: ", self.intermediate_layer_num)
        for i in range(self.intermediate_layer_num):
            x = self.layer_dict["layer_"+str(i+1)](x)
            # x = self.layer_dict["pooling_"+str(i+1)](x)
        output = self.latent_dim(x)
        return output


# decoder model
class Decoder(Model):
    def __init__(self, original_dim=1, kernel_size=(3, 3), intermediate_dim=[32, 64], stride=1, padding=True,    # noqa: E501
                 name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.intermediate_layer_num = len(intermediate_dim)
        self.layer_dict = {}
        self.padding = 'same' if padding else 'valid'

        for index, num in enumerate(intermediate_dim):
            self.layer_dict["layer_"+str(index+1)] = layers.Conv2DTranspose(intermediate_dim[index],    # noqa: E501
                                                                            kernel_size,    # noqa: E501
                                                                            strides=stride,    # noqa: E501
                                                                            padding=self.padding,    # noqa: E501
                                                                            activation='relu',    # noqa: E501
                                                                            kernel_regularizer=regularizers.l2(0.0001))    # noqa: E501
            # self.layer_dict["upsampling_"+str(index+1)] = layers.UpSampling2D(size=(2, 2))    # noqa: E501

        self.output_layer = layers.Conv2DTranspose(original_dim,
                                                   kernel_size,
                                                   strides=stride,
                                                   padding=self.padding,
                                                   activation='sigmoid',
                                                   kernel_regularizer=regularizers.l2(0.0001))    # noqa: E501

    def call(self, input):
        x = input

        for i in range(self.intermediate_layer_num, 0, -1):
            x = self.layer_dict["layer_"+str(i)](x)
            # x = self.layer_dict["upsampling_"+str(i)](x)

        output = self.output_layer(x)
        return output


class VectorQuantizer(Model):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (
            beta  # This parameter is best kept between [0.25, 2] as per the paper.    # noqa: E501
        )

        # Initialize the embeddings which we will quantize.
        w_init = random_uniform_initializer()
        self.embeddings = Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"    # noqa: E501
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = shape(x)
        flattened = reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = one_hot(encoding_indices, self.num_embeddings)
        quantized = matmul(encodings, self.embeddings, transpose_b=True)
        quantized = reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer.
        # You can learn more about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check    # noqa: E501
        # the original paper to get a handle on the formulation of the loss function.    # noqa: E501
        commitment_loss = self.beta * reduce_mean(
            (stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = reduce_mean((quantized - stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = matmul(flattened_inputs, self.embeddings)
        distances = (
            reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = argmin(distances, axis=1)
        return encoding_indices


# variation of vae model
# combine both encoder and decoder
class VQ_VAE(Model):
    def __init__(self, original_dim=1, kernel_size=(3, 3), stride=1, padding=True,    # noqa: E501
                 num_embeddings=128, intermediate_dim=[32, 64], latent=64, name='vq_vae', **kwargs):    # noqa: E501
        super(VQ_VAE, self).__init__(name='vae', **kwargs)
        self.encoder = Encoder(kernel_size=kernel_size, intermediate_dim=intermediate_dim, stride=stride, padding=padding)    # noqa: E501
        self.vector_quantizer = VectorQuantizer(num_embeddings, latent)
        self.decoder = Decoder(original_dim, kernel_size=kernel_size, intermediate_dim=intermediate_dim, stride=stride, padding=padding)    # noqa: E501

    def call(self, input):
        z = self.encoder(input)
        z = self.vector_quantizer(z)
        output = self.decoder(z)
        return output
