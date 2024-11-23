from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import pandas as pd

print('------------------test app-----------------------')

def graph_output(i):
    data = pd.read_csv(csv_file)
    y = data['pred']
    x = range(len(y))

    # Clear and re-draw the bars without resetting the axes
    ax.clear()

    # Plot bars and threshold line
    ax.bar(x, y, color='lightsteelblue', label='pred')
    ax.set_title('Inferencing results', size=18)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Mean abs error')

    # Add grid and threshold line
    ax.grid(axis='x', color='blue', lw=0.5, linestyle='--', alpha=0.2)
    ax.grid(axis='y', color='blue', lw=0.5, linestyle='--', alpha=0.2)
    ax.axhline(y=threshold, color='red', lw=2, ls='--', alpha=0.6, label=f'threshold: {threshold}')
    ax.legend()

    plt.tight_layout()  # For better spacing

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-th', '--threshold', required=True, help='Threshold for anomaly detection')
    parser.add_argument('-csv', '--csvFile', required=True, help='The CSV file to plot')
    args = parser.parse_args()

    threshold = float(args.threshold)
    csv_file = args.csvFile

    fig, ax = plt.subplots(figsize=(8, 4))

    # Set up FuncAnimation with real-time interval
    anim = FuncAnimation(fig, graph_output, interval=1000)  # Update every 1000 ms (1 second)
    plt.show()
