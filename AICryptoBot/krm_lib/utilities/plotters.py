# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 08:52:28 2023

@author: JohnMurphy
"""

# MAIN
def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-50):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 50 scores')
    plt.savefig(figure_file)
