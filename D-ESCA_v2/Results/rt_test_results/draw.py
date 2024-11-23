from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import pandas as pd


def graph_output(i):
    data = pd.read_csv(csv_file)
    y = data['pred']
    x = range(len(y))

    plt.cla()

    ax.bar(x, y, color='lightsteelblue', label='pred')
    ax.set_title('Inferencing results', size=18)
    ax.set_xlabel('Sample number')
    ax.set_ylabel('Mean abs error')

    ax.grid(axis='x', color='blue', lw=0.5, linestyle='--', alpha = 0.2)
    ax.grid(axis='y', color='blue', lw=0.5, linestyle='--', alpha = 0.2)

    ax.axhline(y=threshold, xmin=0.05, xmax=i -0.05,
            color='red',
            lw=2,
            ls='--',
            alpha=0.6,
            label='threshold')
    ax.legend()

    plt.ion()
    #plt.draw()
    plt.pause(1000)


# if __name__=='__main__':
parser = ArgumentParser()
parser.add_argument('-th', '--threshold')
parser.add_argument('-csv', '--csvFile', help='The csv file to plot')
args = parser.parse_args()

globals()['threshold'] = float(args.threshold)
globals()['csv_file'] = args.csvFile

fig, ax = plt.subplots(figsize=(8,4))
globals()['ax'] = ax

anim = FuncAnimation(fig, graph_output, interval=1000)
plt.show()
