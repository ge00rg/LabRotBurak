import pickle
import matplotlib.pyplot as plt
import numpy as np

def fsave(figure, filename):
    pickle.dump(figure, open(filename+'.plt', 'wb'))


def fshow(filename):
    fig = pickle.load(open(filename, 'rb'))
    plt.show()