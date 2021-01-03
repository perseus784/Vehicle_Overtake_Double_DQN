import os
import matplotlib.pyplot as plt
import csv
from config import *


def get_data(filename):
    csv_reader = csv.reader(open(os.path.join(log_save_path, filename), 'r'), delimiter=',')
    index, values = [], []
    for x in csv_reader:
        if x:
            index.append(int(x[0]))
            values.append(float(x[1]))
    return index, values

fig, axs = plt.subplots(1, 2)

lo_counter, loss = get_data('loss.csv')
episodes, rewards = get_data('episodic_reward.csv')

axs[0].plot(loss)
axs[1].plot(rewards)
plt.show()
