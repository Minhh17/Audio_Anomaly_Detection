import numpy as np
import io
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy
from os.path import join, isdir
from gammatone import plot
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
import json
from os import mkdir, makedirs

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # 👇️ alternatively use str()
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Postprocessor():
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def plot_loss_hist(self, loss, legend):
        figure = plt.figure(figsize=(4,3))
        plt.hist(loss, bins=30)
        plt.xlabel("MSE")
        plt.ylabel("Number of samples")
        plt.title(f'{legend} loss distribution')
        plt.tight_layout()
        return figure

    def plot_roc_curve(self, predictions, true_labels):
        figure = plt.figure(figsize=(4,3))
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        plt.plot(fpr, tpr, marker='.')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.title('ROC Curve')
        plt.grid()
        plt.tight_layout()
        return figure

    def plot_pr_curve(self, predictions, true_labels):
        figure = plt.figure(figsize=(4,3))
        precision, recall, thresholds = precision_recall_curve(true_labels, predictions)
        plt.plot(recall, precision, marker='.')
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Prescision')
        # show the legend
        plt.title('Precision-Recall Curve')
        plt.grid()
        plt.tight_layout()
        return figure

    def save_threshold(self,predictions, true_labels, max, min, path_to_save):
        path = path_to_save
        if not isdir(path):
            makedirs(path)

        precision, recall, thresholds = precision_recall_curve(true_labels, predictions)
        threshold = thresholds[np.argmax(precision+recall)]
        # print("MIN",min)
        # print("MAX",max)
        metrics = {
            # 'auc': auc,
            # 'auc_0.1': auc_1,
            # 'fpr': fpr.tolist(),
            # 'tpr': tpr.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'threshold': threshold,
            'max': max,
            'min': min
        }
        # print(metrics)
        with open(join(path, 'metrics_detail.json'), 'w') as file:
            json.dump(metrics, file, indent=4, cls=NpEncoder)

        print('Done!!')
        return 0


    def plot_to_image(self, figure):
        """
            Converts the matplotlib plot specified by 'figure' to a PNG image and
            returns it. The supplied figure is closed and inaccessible after this call.
        """
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def _plot_mel(self, input_feature, idx):
        duration = self.cfg
        aspect_ratio = duration/scipy.constants.golden
        # may need to return Figure type
        figure, ax = plt.subplots(figsize=(4,3))
        formatter = plot.ERBFormatter(f_min, f/2, unit='Hz', places=0)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_title(idx)
        ax.xaxis.label.set_text('Time (s)')
        ax.yaxis.label.set_text('Freq (hz)')
        im = ax.imshow(input_feature, extent=[0,duration,1,0], aspect=aspect_ratio)
        plt.tight_layout()
        
        return figure
    
    def _plot_gamma(self):
        pass
    def evaluate(self):
        pass