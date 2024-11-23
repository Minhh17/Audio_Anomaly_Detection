import pyaudio
import wave
import shutil
import os
import sys
sys.path.append(os.getcwd())
from config.config_manager import ConfigManager
from helper.parser import arg_parser 
from os.path import join
from os import listdir, getpid, scandir, rename, environ, remove, setpgrp, killpg,_exit


cfg = ConfigManager()
config_file = arg_parser('Create Dataloader for further uses.')
cfg.update_from_file(config_file)
def CPfile(output_file, base_file, temp_file):
    if (os.path.exists(base_file) == False):
        shutil.copy2(output_file, temp_file)
        rename(temp_file, base_file)

RECORD_SECONDS = cfg.get('REALTIME.SECOND')                         # Length of time to record (seconds)
                                            # File name to save the audio
WAVE_RECORD_PATH = join(cfg.get('REALTIME.LOG_PATH'),'record')
TEMP_PATH = join(cfg.get('REALTIME.LOG_PATH'),'temp')
if os.path.exists(WAVE_RECORD_PATH) == False:
        os.makedirs(WAVE_RECORD_PATH)
if os.path.exists(TEMP_PATH) == False:
        os.makedirs(TEMP_PATH)       
WAVE_OUTPUT_FILENAME = join(WAVE_RECORD_PATH, 'output.wav')
WAVE_BASE_FILENAME = join(WAVE_RECORD_PATH,'basefile.wav')
WAVE_TEMP_FILENAME = join(WAVE_RECORD_PATH,'temp.wav')

iDeviceIndex = cfg.get('REALTIME.DEVICE_INDEX_INPUT')                   # Index number of recording device

    # Basic Information Settings
FORMAT = pyaudio.paInt16                    # Audio Format
CHANNELS = cfg.get('REALTIME.CHANNELS')                                 # monaural
RATE = cfg.get('REALTIME.SAMPLING_RATE')                                  # sample rate
CHUNK = 2**11                               # Number of data points
audio = pyaudio.PyAudio()                   #
 
stream = audio.open(format=FORMAT, channels=CHANNELS,
        rate=RATE, input=True,
        input_device_index = iDeviceIndex, # Index number of recording device
        frames_per_buffer=CHUNK)
if (os.path.exists(WAVE_BASE_FILENAME) == True):
    os.remove(join(WAVE_BASE_FILENAME))

#------------- Recording start ----------------
try:
    i = 0
    while True:
        print("Start cre Audio file")
        try:
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()
        except:
            print("os.system() failed")
        print("End cre Audio file")
        CPfile(WAVE_OUTPUT_FILENAME, WAVE_BASE_FILENAME, WAVE_TEMP_FILENAME)
        if (os.path.exists(WAVE_OUTPUT_FILENAME) == True):
            os.remove(WAVE_OUTPUT_FILENAME)
except KeyboardInterrupt as e:
    print(e)


print ("finished recording")
 
#-------------- End of Recording ---------------
 
stream.stop_stream()
stream.close()
audio.terminate()
 
