import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST #NB: må installere i riktig mappe ved å skrive "pip install python-mnist" i terminal/command prompt

#load data
mndata = MNIST('./dir_with_mnist_data_files')
images, labels = mndata.load_training()


