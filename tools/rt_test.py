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
import subprocess
from config import update_config, get_cfg_defaults
from helper.parser import arg_parser 
from helper.audio_cleanup import clean_up
import psutil


def testing(cfg = None, eval=None):
    # gate keeping check
    root = dirname(__file__)

    # load threshold and model from file
    # metric_path = cfg.POSTPROCESS.PATH_SAVE_THRESHOLD
    manual_threshold = cfg.REALTIME.MANUAL_THRESHOLD
    rtime = cfg.REALTIME.RUNTIME
    log_dir = cfg.TRANSFER_LEARNING.SAVE_PATH if cfg.REALTIME.TRANSFER_LEARNING  else cfg.TRAINING.SAVE_PATH
    model_name = cfg.MODEL.TYPE
    metric_file = join(log_dir,'save_parameter','metrics_detail.json')
    model = load_model(join(log_dir,'saved_model',model_name))

    with open(metric_file, 'r') as f:
        metric = json.load(f)

    auto_th = metric['threshold']
    MAX = metric['max']
    MIN = metric['min']

    threshold = auto_th if not manual_threshold else float(manual_threshold)
    print(f'Threshold: {threshold}')


    # second load sample files
    # sample_loc = join(root,'test_samples/test') # change this to the recorded file location
    sample_loc = join(cfg.REALTIME.LOG_PATH,'record')

    # some characteristics of gammatone feature
    window_time = cfg.PREPROCESS.GAMMA.WINDOW_TIME 
    channels = cfg.PREPROCESS.GAMMA.CHANNELS
    hop_time = window_time/2
    f_min = cfg.PREPROCESS.GAMMA.F_MIN 
    frame_rate = 44100

    start = time.time()
    i = 0

    print(f'Real-time detection start... model:{model_name}')

    # a dict to store some info
    data = {
        'name': None,
        'pred': None,
        'time': None,
    }

    # prepare cvs file to log in the information
    csv_file = join(cfg.REALTIME.LOG_PATH, 'temp.csv')
    field_names = list(data.keys())
    with open(csv_file, 'w') as file:
        csv_writer = csv.DictWriter(file, fieldnames=field_names)
        csv_writer.writeheader()

    # run another subprocess to read from the csv file and draw graph dynamically
    plotting_graph = join(root, '../helper', 'plotting_graph.py')
    command = ['python3',plotting_graph, '-th', str(threshold), '-csv', csv_file]
    graph = subprocess.Popen(command, preexec_fn=setpgrp)


    # print('------------------1-----------------------')

    try:
        while(True):
            # load and process the audio
            base_file = listdir(sample_loc)
            end = time.time()
            if (end-start) > rtime:
                break
            if not 'basefile.wav' in base_file:
                continue
            try: 
                s = join(sample_loc, 'basefile.wav')
                _, audio = wavfile.read(s)
                gtg = gtgram.gtgram(audio, frame_rate, window_time, hop_time, channels, f_min)    # noqa: E501
                a = np.flipud(20 * np.log10(gtg))
                # rescale
                a = np.clip((a-MIN)/(MAX-MIN), a_min=0, a_max=1)
                a = np.reshape(a, (1 ,a.shape[0], a.shape[1], 1))

                if(a.shape != (1, 32, 32, 1)):
                    # print("Input shape Error:")
                    # print(a.shape)
                    os.remove(join(sample_loc, 'basefile.wav'))
                    continue
                pred = np.mean((a-model.predict(a))**2)
                type = 1 if pred > threshold else 0

                data[field_names[0]] = i
                data[field_names[1]] = pred
                data[field_names[2]] = datetime.now().strftime("%Y%m%d-%H%M%S")

                with open(csv_file, 'a') as file:
                    csv_writer = csv.DictWriter(file, fieldnames=field_names)
                    csv_writer.writerow(data)

                rename(s, join(cfg.REALTIME.LOG_PATH,'temp', f'basefile_{i}.wav')) # move file
                i += 1
                if type == 1:
                    print(f'Detect abnormal at {end - start}s from starting time.')
                else:
                    print('Normal at {end - start}s from starting time..')
                
            except:
                pass

        # print('inferencing end.')
        # wait to input any key
        var = input("Please input any key.")
        killpg(graph.pid, signal.SIGINT)
    except KeyboardInterrupt:

        # print('inferencing end.')
        # wait to input any key
        # var = input("Please input any key.")
        killpg(graph.pid, signal.SIGINT)
        # killpg(monitoring_proc.pid,signal.SIGINT)


    return 0

root = dirname(__file__)
cfg = get_cfg_defaults()
# config_file = arg_parser('Create Dataloader for further uses.')



from argparse import ArgumentParser

parser = ArgumentParser(description='program for running other processes')

# arguments that is needed for every type
parser.add_argument('-f', '--filePath', help='source to split', required=True)
parser.add_argument('-cfg', '--config', help='specify the default .yaml file', required=True)
args = parser.parse_args()

config_file = args.config
dataImportPath = args.filePath


cfg = update_config(cfg, config_file)
used_ram_init =  psutil.virtual_memory().used/1024/1024
monitoring = join(root, '../helper', 'Resource_monitoring.py')
monitor_savepath = join(cfg.REALTIME.LOG_PATH, 'mornitor')


if not os.path.exists(monitor_savepath):
    os.makedirs(monitor_savepath)
pid = getpid()
prediction_list = []
now = datetime.now()
cur_time = now.strftime("%Y%m%d_%H%M%S")
if(cfg.REALTIME.IMPORT_FILE == False):
    record_mic = join(root, '../helper','usbmictest.py')
else:
    record_mic = join(root, '../helper','import_wav_files.py')
# record_mic = join(root, '../tools','create_dataset.py')

try:
    # monitoring_proc = subprocess.Popen(['gnome-terminal', '--disable-factory','--', 'python3', monitoring, '-p', str(pid), '-log', monitor_savepath, '-ri', str(int(used_ram_init)), '-cfg', './config/params.yaml'], 
    #                                 preexec_fn=setpgrp)

    audio_record = subprocess.Popen(['gnome-terminal', '--disable-factory','--', 'python3', record_mic, '-cfg', './config/params.yaml', '-f', dataImportPath],
                                    preexec_fn=setpgrp)
    
    print(audio_record)
   
       # print('------------------2-----------------------')
    


    save_file = testing(cfg= cfg, eval=prediction_list)
    killpg(audio_record.pid, signal.SIGINT)
    # killpg(monitoring_proc.pid,signal.SIGINT)
    print('Cleaning up...')
    clean_up(cur_time)
except KeyboardInterrupt as e:
    print('Get interrupted by keyboard')
    print('Saving the results so far...')
    killpg(audio_record.pid, signal.SIGINT)
    # killpg(monitoring_proc.pid,signal.SIGINT)
    print('Cleaning up...')
    clean_up(cur_time)
    # killpg(monitor.pid, signal.SIGINT)
    try:
        sys.exit(0)
    except SystemExit:
        _exit(0)