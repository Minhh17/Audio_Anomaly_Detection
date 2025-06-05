import tensorflow as tf
import numpy as np
from core.Models import get_model
from core.Preprocessing import Preprocessor
from core.Postprocessing import Postprocessor
from datetime import datetime
import os
from os.path import join
from os import makedirs
import json
from config.config_manager import ConfigManager 

class ModelTrainer():
    def __init__(self, cfg):
        # important parameters for training
        # self.model_name = cfg.MODEL.TYPE 
        self.model_name = cfg.get('MODEL.TYPE')
        # self.log_dir = cfg.TRAINING.SAVE_PATH
        self.log_dir = cfg.get('TRAINING.SAVE_PATH')
        # Ensure the log directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # self.epochs = cfg.TRAINING.EPOCH
        self.epochs = cfg.get('TRAINING.EPOCH')
        # self.learning_rate = cfg.TRAINING.LEARNING_RATE
        self.learning_rate = cfg.get('TRAINING.LEARNING_RATE')
        # self.theshold_save_path =  cfg.POSTPROCESS.PATH_SAVE_THRESHOLD
        self.theshold_save_path = join(self.log_dir,'save_parameter')
        makedirs(self.log_dir, exist_ok=True)

        # create the model
        self.model = get_model(cfg)
        self._setup_logger()
        self._setup_metrics()

        # a preprocessor and postprocessor
        self.pre_prc = Preprocessor(cfg)
        self.post_prc = Postprocessor(cfg)

        # dict containing all neccessary fuctions
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

    # metrics related functions
    def _setup_metrics(self):
        self.trackers = {
            'total_loss': tf.keras.metrics.Mean(name="total_loss"),
            'reconstruction_loss': tf.keras.metrics.Mean(name="reconstruction_loss"),
            'model_loss': tf.keras.metrics.Mean(name="model_loss"),
            # 'supervised_loss': tf.keras.metrics.Mean(name="supervised_loss")
        }
    def _write_chart_files(sellf, data):
        value_numpy = data[0][1].numpy()
        print(value_numpy)
        with open("/home/haiminh/Desktop/info.txt", "a") as file:
            file.write(f"{value_numpy}\n")

    def _write_to_files(self, data, epoch, step):
        value_numpy = data[0][1].numpy()
        log_data = {
            "epoch": epoch,
            "step": step,
            "loss": float(value_numpy)
        }
        print(f"[PLOT]{json.dumps(log_data)}", flush=True)  # Xuất ra stdout
    
        
    def _get_metrics(self):
        return [(key, value.result()) for key, value in self.trackers.items()]

    def _update_metrics(self, losses_dict):
        if len(list(losses_dict.keys())) != len(list(self.trackers.keys())):
            raise ValueError(f'Not enough parameters for all metrics')
        for key, loss in losses_dict.items():
            self.trackers[key].update_state(loss)

    def _reset_metrics(self):
        for _, metric in self.trackers.items():
            metric.reset_state()

    # logger related function
    def _setup_logger(self):
        # create tensorboard for each data part
        log_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard = {
            'train': tf.summary.create_file_writer(join(self.log_dir, 'logs', log_name, 'train')),
            'val': tf.summary.create_file_writer(join(self.log_dir, 'logs', log_name, 'val')),
            'test': tf.summary.create_file_writer(join(self.log_dir, 'logs', log_name, 'test')),
        }

    def _write_train_log(self, preds, labels, metrics, epoch, datapart='train'):
        with self.tensorboard[datapart].as_default(step=epoch):
            histogram = self.post_prc.plot_loss_hist(preds, datapart)
            tf.summary.image(name='Loss histogram', data=self.post_prc.plot_to_image(histogram))
            # metrics is a list of tuple, which is the result from calling _get_metrics
            for metric in metrics:
                tf.summary.scalar(metric[0], metric[1])

    def _write_test_log(self, preds, labels, metrics, epoch, datapart):
        with self.tensorboard['test'].as_default(step=epoch):
            # histogram plot
            histogram = self.post_prc.plot_loss_hist(preds, 'test')
            tf.summary.image(name='Loss histogram',data=self.post_prc.plot_to_image(histogram))

            # precision-recall plot
            pr_curve = self.post_prc.plot_pr_curve(preds, labels)
            tf.summary.image(name='PR curve',data=self.post_prc.plot_to_image(pr_curve))

            # ROC plot
            roc_curve = self.post_prc.plot_roc_curve(preds, labels)
            tf.summary.image(name='ROC curve',data=self.post_prc.plot_to_image(roc_curve))                   

            # metrics is a list of tuple, which is the result from calling _get_metrics
            for metric in metrics:
                tf.summary.scalar(metric[0], metric[1])

            self.post_prc._log_histogram(preds)
            self.post_prc._log_pr_curve(preds, labels)
            self.post_prc._log_roc_curve(preds, labels)
                

    # some other function for setting up training/testing
    def load_pretrained_weights(self, path):

        # a =  np.random.rand(1,32,32,1)
        # # self.model.build(input_shape = (1,32,32,1))
        # print("-------1-------")
        # latest = tf.train.latest_checkpoint(path)
        # print(latest)
        # self.model.load_weights(path)
        # print("----predict ------")
        # print(self.model.predict(a).shape)
        # self.model.predict()
        self.model = tf.keras.models.load_model(path)
        # print("-------2-------")
        print(self.model.summary())

    def compile(self, optimizer=None):
        opt = tf.keras.optimizers.Adam(self.learning_rate) if not optimizer else optimizer(self.learning_rate)
        self.model.compile(opt)

    # define some function to calculate the losses
    def _compute_loss(self, original, reconstruction):
        # print("----------compute loss 0-------")
        sample_wise_loss = self._reconstruction_loss_sample_wise(original, reconstruction)
        # print("sample_wise_loss : ")
        # print(sample_wise_loss.shape)
        return {'reconstruction_loss': tf.reduce_mean(sample_wise_loss)}, tf.squeeze(sample_wise_loss)

    def _reconstruction_loss_sample_wise(self, original, reconstruction):
        return tf.reduce_mean((original - reconstruction) ** 2, axis=(1, 2, 3), keepdims=True)

    # train_step: update each iteration
    # @tf.function
    def train_step(self, x):
        # training
        total_loss = 0
        with tf.GradientTape() as tape:
            # Outputs from the AE model.
            reconstructions = self.model(x)

            # Calculate the losses.
            loss_dict, sample_loss = self._compute_loss(x, reconstructions)
            loss_dict['model_loss'] = sum(self.model.losses)
            for _ ,loss_value in loss_dict.items(): 
                total_loss += loss_value         

        # Backpropagation.
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        loss_dict['total_loss'] = total_loss
        return loss_dict, sample_loss

    # @tf.function
    def val_step(self, x):
        # Outputs from the AE model
        reconstructions = self.model(x)
        # Calculate the losses.
        loss_dict, sample_loss = self._compute_loss(x, reconstructions)
        loss_dict['model_loss'] = 0
        total_loss = 0
        for _ ,loss_value in loss_dict.items(): 
            total_loss += loss_value
        loss_dict['total_loss'] = total_loss
        return loss_dict, sample_loss

    # @tf.function
    def test_step(self, x):
        '''
            Override this function if you want test_step to behave differently
            Here, test_step behaves like val_step
        '''
        return self.val_step(x)

    # a function to determine the number of samples in tf.Dataset
    def _get_number_of_samples(self, dataset):
        samples_num = 0
        for feature, _, _ in dataset:
            samples_num += feature.shape[0]
        return samples_num

    # the main function to train the model
    def fit(self, data_dict):
        '''
            Training process
            -----------------
            Parameters:
                data_dict: a dict with (key, value) pair containing (data_part, tf.dataset)
        '''
        # data_num_dict = {part: self._get_number_of_samples(data_dict[part]) for part in ['train', 'test', 'val']}

        for epoch in range(self.epochs):
            for part in ['train', 'val', 'test']:
                sample_losses = []
                sample_labels = []

                for feature, label, idx in data_dict[part]:
                    processed_feature = self.pre_prc.rescale(feature)
                    processed_feature = self.pre_prc.add_dimentsion(processed_feature)

                    # Forward pass
                    losses_dict, sample_loss = self.impl_steps[part](processed_feature)

                    # Cập nhật metrics
                    self._update_metrics(losses_dict)
                    metrics_value = self._get_metrics()

                    # Gộp thông tin thành 2 dòng
                    loss_str = " | ".join([f"{loss_name}: {loss_value.numpy():.8f}" for loss_name, loss_value in metrics_value])
                    print(f"[PROG] Epoch {epoch+1}/{self.epochs} | Step {part.capitalize()}", flush=True)
                    print(f"[PROG] {loss_str}", flush=True)
                    print(f"[PROG] ", flush=True)

                    # Lưu sample loss và label
                    sample_losses.append(sample_loss.numpy())
                    sample_labels.append(label.numpy())

                # Ghi log sau khi hoàn thành 1 phần (train/val/test)
                self.impl_logs[part](preds=np.concatenate(sample_losses), 
                                    labels=np.concatenate(sample_labels), 
                                    metrics=metrics_value, 
                                    epoch=epoch, datapart=part)

                # Xử lý threshold sau khi test
                if part == 'test':
                    max_val, min_val = self.pre_prc.get_max_min()
                    self.post_prc.save_threshold(np.concatenate(sample_losses), 
                                                np.concatenate(sample_labels), 
                                                max_val, min_val, 
                                                self.theshold_save_path)

                # Reset metrics sau mỗi phần
                self._reset_metrics()

            # Hiển thị progress tổng sau mỗi epoch
            print(f"[EPOCH] {epoch+1}/{self.epochs}", flush=True)

        # Lưu model sau khi hoàn thành tất cả epoch
        self.model.save(join(self.log_dir, 'saved_model', self.model_name))
        print("[FINISHED]")


