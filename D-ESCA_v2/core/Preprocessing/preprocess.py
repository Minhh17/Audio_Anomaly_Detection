import tensorflow as tf
import numpy as np
import os

class Preprocessor():
    def __init__(self, cfg, tl=False) -> None:
        if tl:
            self._load_statistic_tl(cfg.DATASET.PATH.TFRECORDS, cfg.TRANSFER_LEARNING.TFRECORDS)
        else:
            self._load_statistic_npz(cfg.DATASET.PATH.TFRECORDS[0])

    def add_dimentsion(self, input):
        return tf.expand_dims(input, axis=-1)

    def rescale(self, input):
        return tf.clip_by_value(t=(input-self.min)/self.denominator, clip_value_min=0.0, clip_value_max=1.0)

    def _load_statistic_npz(self, path=None):
        with open(os.path.join(path, 'stats.npz'), 'rb') as file:
            holder = np.load(file)
            self.max = holder['max']
            self.min = holder['min']
        self.denominator = self.max - self.min

    def _load_statistic_tl(self, base_records_list, target_records_list):
        # since target list should have only 1 element, we will use this as base value
        self._load_statistic_npz(target_records_list[0])
        max_list = [self.max]
        min_list = [self.min]
        for dir in base_records_list:
            self._load_statistic_npz(dir)
            max_list.append(self.max)
            min_list.append(self.min)
        self.max = np.max(max_list)
        self.min = np.min(min_list)

    def get_max_min(self):
        return self.max, self.min