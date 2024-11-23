import tensorflow as tf
from core.Trainer import ModelTrainer
import os
from os.path import join
from datetime import datetime
from core.Preprocessing import Preprocessor
from core.Postprocessing import Postprocessor

class TL_Trainer(ModelTrainer):
    def __init__(self, cfg, from_config=True, **kwargs):
        # super().__init__(cfg)
        self.model_name = cfg.MODEL.TYPE 
        self.log_dir = cfg.TRANSFER_LEARNING.SAVE_PATH
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.learning_rate = cfg.TRANSFER_LEARNING.LEARNING_RATE
        self.epochs = cfg.TRANSFER_LEARNING.EPOCH
        self._setup_logger()
        self._setup_metrics()
        self.pre_prc = Preprocessor(cfg, tl=True)
        self.post_prc = Postprocessor(cfg)

        self.impl_steps = {
            'train': self.train_step,
            'test': self.test_step,
            'val': self.val_step,
        }
        self.impl_logs = {
            'train': self._write_train_log,
            'test': self._write_test_log,
            'val': self._write_train_log,
        }

        self.theshold_save_path = join(self.log_dir,'save_parameter')
        self.beta = cfg.TRANSFER_LEARNING.BETA if from_config else kwargs['beta']
        self.training_anomaly = kwargs['training_anomaly']
        self.batch_size = cfg.TRANSFER_LEARNING.ANOM_BATCH_SIZE \
            if from_config else kwargs['batch_size']
        self.based_model_path = cfg.TRANSFER_LEARNING.BASED_WEIGHTS \
            if from_config else kwargs['based_model']
        # load the pretrained weights if transfer learning mode is on
        self.load_pretrained_weights(self.based_model_path)
        # tf.concat(temp, axis=0)

    # # override metrics
    # def _setup_logger(self):
    #     # create tensorboard for each data part
    #     log_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    #     self.tensorboard = {
    #         'train': tf.summary.create_file_writer(join(self.log_dir, 'logs', log_name, 'train')),
    #         'val': tf.summary.create_file_writer(join(self.log_dir, 'logs', log_name, 'val')),
    #         'test': tf.summary.create_file_writer(join(self.log_dir, 'logs', log_name, 'test')),
    #     }

    def _setup_metrics(self):
        self.trackers = {
            'total_loss': tf.keras.metrics.Mean(name="total_loss"),
            'reconstruction_loss': tf.keras.metrics.Mean(name="reconstruction_loss"),
            'model_loss': tf.keras.metrics.Mean(name="model_loss"),
            'supervised_loss': tf.keras.metrics.Mean(name="supervised_loss")
        }

    # compute loss in transfer learning
    # @tf.function
    def _compute_loss(self, original, reconstruction):
        # print("----------compute loss 1-------")
        sample_wise_loss = self._reconstruction_loss_sample_wise(original, reconstruction)
        # print("----------sample_wise_loss 1-------")
        anom_loss = self._anomaly_loss()
        # print("----------anom_loss 1-------")
        supervised_loss = self.beta*tf.reduce_mean(
                        tf.sigmoid(anom_loss - sample_wise_loss)
                )
        return {'reconstruction_loss': tf.reduce_mean(sample_wise_loss), \
                'supervised_loss': -1*supervised_loss}, tf.squeeze(sample_wise_loss)

    def _anomaly_loss(self):
        temp = []
        for features, _, _ in self.training_anomaly:
            processed_feature = self.pre_prc.add_dimentsion(self.pre_prc.rescale(features))
            anom_recon = self.model(processed_feature)
            temp.append(tf.reduce_mean((processed_feature - anom_recon)**2, axis=(1, 2, 3), keepdims=True))
        return tf.transpose(tf.concat(temp, axis=0))

    def fit(self, data_dict):
        print(f"Normal_tl {self._get_number_of_samples(data_dict['train'])}")
        print(f"Anomaly tl {self._get_number_of_samples(self.training_anomaly)}")
        
        super().fit(data_dict)