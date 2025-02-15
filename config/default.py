from yacs.config import CfgNode as CN

_C = CN()

_C.PREPROCESS = CN()
_C.PREPROCESS.TYPE = 'mel'   # gamma or mels
_C.PREPROCESS.AUDIO_LEN = 2   # second
_C.PREPROCESS.SEGMENT_LEN = 2  # second
_C.PREPROCESS.SAMPLE_PER_FILE = 200
_C.PREPROCESS.GAMMA = CN() # define all parameters needed for gamma feature extraction
_C.PREPROCESS.GAMMA.WINDOW_TIME = 0.06*2  # 2s file: winddow_time = 0.06*2 for 32 frame -> nfft = 5292, .0305*2 for 64 frame
_C.PREPROCESS.GAMMA.HOP_TIME = 0.06
_C.PREPROCESS.GAMMA.CHANNELS = 32
_C.PREPROCESS.GAMMA.F_MIN = 100
_C.PREPROCESS.MEL = CN() # define all parameters needed for mel feature extraction
_C.PREPROCESS.MEL.SR = 16000
_C.PREPROCESS.MEL.NFFT = 2048
_C.PREPROCESS.MEL.N_BANDS = 32

_C.DATASET = CN()
_C.DATASET.PATH = CN()
_C.DATASET.PATH.NORMAL = './park_demo/park_source'  # directory of normal training data
_C.DATASET.PATH.TEST = None # directory to test dataset. If this is None then part of normal training data will be used for test
_C.DATASET.PATH.ANOMALY = './park_demo/park_Target3' # directory of anomaly dataset (only use for testing)
_C.DATASET.PATH.TFRECORDS = []
_C.DATASET.RATIO = CN() # specify the ratio of normal data for training and testing. The remaining will be used for validating
_C.DATASET.RATIO.TRAIN = 0.8
_C.DATASET.RATIO.TEST = 0.1 # if DATASET.PATH.TEST is not assigned, test data will be from normal training data with this ratio
_C.DATASET.DATALOADER = CN()
_C.DATASET.DATALOADER.BATCH_SIZE = 128
_C.DATASET.DATALOADER.SHUFFLE = True

# parameters related to training procedure
_C.TRAINING = CN()
_C.TRAINING.LOG_FOLDER = './Results/temp' ############
_C.TRAINING.EPOCH = 3
_C.TRAINING.LEARNING_RATE = 1e-3
_C.TRAINING.PRETRAINED_WEIGHTS = '.Results/base_training_test/saved_model'
_C.TRAINING.SAVE_PATH = './Results'


# paramters specifically for transfer learning
_C.TRANSFER_LEARNING = CN()
_C.TRANSFER_LEARNING.TFRECORDS = ['./data/target/Target3']
_C.TRANSFER_LEARNING.TEST_DIR = None
_C.TRANSFER_LEARNING.EPOCH = 3
_C.TRANSFER_LEARNING.LEARNING_RATE = 1e-3
_C.TRANSFER_LEARNING.BASED_WEIGHTS = './Results/tl-training_results/Target3/saved_model/vq_vae'
_C.TRANSFER_LEARNING.SAVE_PATH = './Results'
_C.TRANSFER_LEARNING.BETA = 1.0
_C.TRANSFER_LEARNING.ANOM_BATCH_SIZE = 128

# parameter for model
_C.MODEL = CN()
_C.MODEL.TYPE = 'vq_vae' # choose from ['cae', 'vae', 'unconventional_cae', 'vq_vae']

_C.POSTPROCESS = CN()
_C.POSTPROCESS.PATH_SAVE_THRESHOLD = '/home/haiminh/Desktop/D-ESCA_v2/Results/Results/'

_C.REALTIME = CN()
_C.REALTIME.TRANSFER_LEARNING = False
_C.REALTIME.LOG_PATH = '/home/haiminh/Desktop/D-ESCA_v2/Results/rt_test'
_C.REALTIME.MANUAL_THRESHOLD = None
_C.REALTIME.RUNTIME = 1000
_C.REALTIME.DEVICE_INDEX_INPUT = 0
_C.REALTIME.SECOND = 2
_C.REALTIME.CHANNELS = 1
_C.REALTIME.SAMPLING_RATE = 44100
_C.REALTIME.IMPORT_FILE = False

_C.RECORD = CN()
# _C.RECORD.DATASET_MODE = True
_C.RECORD.DATASET_PATH = './dataset'
_C.RECORD.ABNOMALY = False
_C.RECORD.DEVICE_INDEX_INPUT = 0
_C.RECORD.SECOND = 2
_C.RECORD.CHANNELS = 1
_C.RECORD.SAMPLING_RATE = 44100

_C.DEVICE = CN()
_C.DEVICE.JETSON = False

_C.MODEL.VQ_VAE = CN()
_C.MODEL.VQ_VAE.OG_DIM = 1
_C.MODEL.VQ_VAE.NUM_EMBEDDINGS = 128
_C.MODEL.VQ_VAE.INTERMEDIATE_DIM = [32, 64]
_C.MODEL.VQ_VAE.LATENT_DIM = 64

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()