from .vae import VariationalAutoEncoder
from .vq_vae import VQ_VAE


def get_model(cfg):
    '''
        Use parameters from config file to load initiate the right model
    '''
    # name = cfg.MODEL.TYPE
    name = cfg.get('MODEL.TYPE')
    if name == 'vae':
        model = VariationalAutoEncoder(cfg.MODEL.VAE.OG_DIM)
    elif name == 'vq_vae':
        model = VQ_VAE()
        print('VQ-VAE model is loaded.')
    else:
        raise ValueError(f'{name} is not yet implemented.')

    return model
