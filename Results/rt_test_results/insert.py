from os import listdir, getpid, scandir, rename, environ, remove, setpgrp, killpg,_exit
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
from os.path import join, isdir, dirname
import time
import os
import sys
sys.path.append(os.getcwd())
import signal
from scipy.io import wavfile
from gammatone import gtgram
import numpy as np
import json
import csv
from datetime import datetime
from argparse import ArgumentParser

import psutil

data = {
    'name': None,
    'pred': None,
    'time': None,
}

# prepare cvs file to log in the information
csv_file = 'temp.csv'
field_names = list(data.keys())
with open(csv_file, 'w') as file:
    csv_writer = csv.DictWriter(file, fieldnames=field_names)
    csv_writer.writeheader()



pred = [
        0.002105263157894735,
        0.002105263157894735,
        0.002105263157894735,
        0.001905263157894735,
        0.0039473684210526316,
        0.0039473684210526316,
        0.003684210526315789,
        0.003684210526315789,
        0.003684210526315789,
        0.003684210526315789,
        0.0034210526315789475,
        0.003157894736842105,
        0.002894736842105263,
        0.002631578947368421,
        0.0023684210526315788,
        0.0021052631578947367,
        0.0018421052631578946,
        0.002789473684210525,
        0.0013157894736842105,
        0.0010526315789473684,
        0.003894736842105263,
        0.003263157894736842,
        0.002631578947368421,]

i = 0
while(i < 21):   
    data[field_names[0]] = i
    data[field_names[1]] = pred[i]
    data[field_names[2]] = datetime.now().strftime("%Y%m%d-%H%M%S")

    with open(csv_file, 'a') as file:
        csv_writer = csv.DictWriter(file, fieldnames=field_names)
        csv_writer.writerow(data)

    # rename(s, join(cfg.REALTIME.LOG_PATH,'temp', f'basefile_{i}.wav')) # move file
    i += 1