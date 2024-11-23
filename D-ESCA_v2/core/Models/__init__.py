from .cae import ConvolutionalAutoEncoder
from .vae import VariationalAutoEncoder
from .unconventional_cae import Unconven_CAE
from .vq_vae import VQ_VAE


def get_model(cfg):
    '''
        Use parameters from config file to load initiate the right model
    '''
    name = cfg.MODEL.TYPE
    if name == 'cae':
        model = ConvolutionalAutoEncoder()
    elif name == 'vae':
        model = VariationalAutoEncoder(cfg.MODEL.VAE.OG_DIM)
    elif name == 'unconventional_cae':
        model = Unconven_CAE()
    elif name == 'vq_vae':
        model = VQ_VAE()
    else:
        raise ValueError(f'{name} is not yet implemented.')

    return model
