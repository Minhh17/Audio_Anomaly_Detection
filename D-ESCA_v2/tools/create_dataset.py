import pyaudio
import wave
import shutil
import os
import sys
import signal

from os import listdir, scandir, rename, environ, remove, setpgrp, killpg,_exit, getpid
sys.path.append(os.getcwd())
from config import update_config, get_cfg_defaults
from helper.parser import arg_parser 
from os.path import join
from datetime import datetime


def recoder(cfg):
    RECORD_SECONDS = cfg.RECORD.SECOND     # Length of time to record (seconds)

    if not cfg.RECORD.ABNOMALY:
        WAVE_OUTPUT_PATH = join(cfg.RECORD.DATASET_PATH,'normal')
    else: 
        WAVE_OUTPUT_PATH = join(cfg.RECORD.DATASET_PATH,'abnomaly')

    if os.path.exists(WAVE_OUTPUT_PATH) == False:
        os.makedirs(WAVE_OUTPUT_PATH)
    iDeviceIndex = cfg.RECORD.DEVICE_INDEX_INPUT        # Index number of recording device

        # Basic Information Settings
    FORMAT = pyaudio.paInt16                    # Audio Format
    CHANNELS = cfg.RECORD.CHANNELS                                # monaural
    RATE = cfg.RECORD.SAMPLING_RATE                                # sample rate
    CHUNK = 2**11                               # Number of data points
    audio = pyaudio.PyAudio()                   #
    
    stream = audio.open(format=FORMAT, channels=CHANNELS,
            rate=RATE, input=True,
            input_device_index = iDeviceIndex, # Index number of recording device
            frames_per_buffer=CHUNK)
    #------------- Recording start ----------------
    try:
        i = 0
        while True:
            print("Start cre Audio file")
            try:
                frames = []
                WAVE_OUTPUT_FILENAME = join(WAVE_OUTPUT_PATH, datetime.now().strftime("%Y%m%d%H%M%S")+".wav")
                # print(WAVE_OUTPUT_FILENAME)
                for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                    data = stream.read(CHUNK)
                    frames.append(data)
                waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                waveFile.setnchannels(CHANNELS)
                waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                waveFile.setframerate(RATE)
                waveFile.writeframes(b''.join(frames))
                waveFile.close()
            except KeyboardInterrupt:
                print("---- Kill recording")
                audio_record_pid = getpid()
                killpg(audio_record_pid, signal.SIGINT)
            except :
                print("os.system() failed")

            print("End cre Audio file")

    except KeyboardInterrupt as e:
        print(e)    



    print ("finished recording")
    
    #-------------- End of Recording ---------------
    
    stream.stop_stream()
    stream.close()
    audio.terminate()

cfg = get_cfg_defaults()
config_file = arg_parser('Create Dataloader for further uses.')
cfg = update_config(cfg, config_file)
recoder(cfg)