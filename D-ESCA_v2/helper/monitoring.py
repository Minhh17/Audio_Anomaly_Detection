import psutil
import GPUtil
from time import sleep
import json
from argparse import ArgumentParser

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('-log', '--logPath', help='path to the log file')
    args = parser.parse_args()
    RAM = []
    CPU = []
    log_file = args.logPath
    while(True):
        ram = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent()
        print(f"RAM usage: {ram}%")
        print(f"CPU usage: {cpu}%")
        print('-'*50)
        RAM.append(ram)
        CPU.append(cpu)

        # write to a log file
        data = {
            'ram': RAM,
            'cpu': CPU,
        }
        with open(log_file, 'w') as file:
            json.dump(data, file)
        # sleep for 15 seconds
        sleep(10)
