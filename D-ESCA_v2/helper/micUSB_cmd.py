import wave
import datetime
import time
import os
import shutil
import subprocess

def CPfile(fold_file):
    if (os.path.exists("/home/thanhho/SPARC/ESCA/Code/Data/dev_data_fan/Results/realtime/record/basefile.wav") == False):
        shutil.copy2(fold_file, "/home/thanhho/SPARC/ESCA/Code/Data/dev_data_fan/Results/realtime/record/basefile.wav")

if __name__=='__main__':
    channels = 1
    rate = 44100  # Record at 8000 samples per second
    seconds = 2
    try:
        i = 0
        while True:
            print("Start cre Audio file")

            try:
                os.system(f"arecord -D plughw:0,0 -f S16_LE -r {rate} -d {seconds}  /home/thanhho/SPARC/ESCA/Code/Data/dev_data_fan/Results/realtime/record/output.wav > /dev/null 2>&1")
            except:
                print("os.system() failed")
            print("End cre Audio file")

            CPfile("./test_samples/test/output.wav")
            if (os.path.exists("/home/thanhho/SPARC/ESCA/Code/Data/dev_data_fan/Results/realtime/record/output.wav") == True):
                os.remove("/home/thanhho/SPARC/ESCA/Code/Data/dev_data_fan/Results/realtime/record/output.wav")
    except KeyboardInterrupt as e:
        print(e)
