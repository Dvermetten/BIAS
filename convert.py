#imports
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
import json
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
import autokeras as ak
#load data

for samples in [30,50,100,200,500,600]:
    model = tf.keras.models.load_model(f"BIAS/models/opt_cnn_model-{samples}")
    model.save(f"BIAS/models/opt_cnn_model-{samples}.keras")