import os
import sys
sys.path.append(os.getcwd())
from core.DataLoader import Dataloader
from helper.parser import arg_parser 
from config import update_config, get_cfg_defaults
import tensorflow as tf

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024*3.5)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

if __name__ == '__main__':
    # update config based on default.yaml file
    cfg = get_cfg_defaults()
    config_file = arg_parser('Create Dataloader for further uses.')
    cfg = update_config(cfg, config_file)

    # initiate a feature_extractor with all parameters from cfg
    data_loader = Dataloader(cfg)
    data_loader.create_tfrecord()
    data_loader.accumulate_stat()