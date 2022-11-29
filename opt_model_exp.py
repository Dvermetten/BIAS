import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
import autokeras as ak
#load data
from BIAS.SB_Test_runner import get_scens_per_dim, get_simulated_data

rep = 20000
n_samples = 100

for n_samples in [100,200,500]:

    scenes = get_scens_per_dim()
    per_label = {"unif":0, "centre":0, "bounds":0, "gaps/clusters":0, "disc":0}
    X = []
    y = []
    for scene in scenes:
        label = scene[0]
        kwargs = scene[1]
        if (label == "unif"):
            rep1 = 4 * rep
        elif (label in ["trunc_unif", "cauchy", "norm"]):
            rep1 = int(rep / 32)
        elif (label in ["bound_thing","inv_norm", "inv_cauchy"]):
            rep1 = int(rep / 48)
        elif (label in ["clusters","gaps", "part_unif"]):
            rep1 = int(rep / 67)
        elif (label in ["spikes", "shifted_spikes"]):
            rep1 = int(rep / 42)
        data = get_simulated_data(label, rep=rep1, n_samples = n_samples, kwargs=kwargs)
        for r in range(rep1):
            X.append(np.sort(data[:,r]))
        if (label in ["trunc_unif", "cauchy", "norm"]):
            label = "centre"
        elif (label in ["bound_thing","inv_norm", "inv_cauchy"]):
            label = "bounds"
        elif (label in ["gaps", "part_unif", "clusters"]):
            label = "gaps/clusters"
        elif (label in ["spikes", "shifted_spikes"]):
            label = "disc"
        per_label[label] += rep1
        y.extend([label]*rep1)

    print(per_label)
    X = np.array(X)
    int_y, targetnames= pd.factorize(y)

    cat_y = to_categorical(int_y)


    X_train, X_test, y_train, y_test = train_test_split(X, cat_y, test_size=0.2, random_state=42, stratify=int_y)

    clf = ak.ImageClassifier(
        max_trials=100,
        overwrite=True,
    )
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    # Search for the best model with EarlyStopping.
    cbs = [
        tf.keras.callbacks.EarlyStopping(patience=5),
    ]

    clf.fit(
        x=X_train,
        y=y_train,
        epochs=50,
        callbacks=cbs
    )
    # Evaluate on the testing data.
    print(
        "Accuracy: {accuracy}".format(
            accuracy=clf.evaluate(x=X_test, y=y_test)
        )
    )

    model = clf.export_model()
    model.summary()
    model.save(f"opt_cnn_model-{n_samples}.h5")
    tf.keras.utils.plot_model(model, to_file=f"opt_cnn_model-{n_samples}.png")

    class newmodel(MLPClassifier):
        def __init__(self, model):
            self.model = model
        def predict(self, X):
            y = self.model.predict(X)
            return np.argmax(y, axis=1)

    model1 = newmodel(model)
    fig, ax = plt.subplots(figsize=(14, 14))
    plot_confusion_matrix(model1, X_test, np.argmax(y_test, axis=1), normalize='true', xticks_rotation = 'vertical', display_labels = targetnames, ax=ax) 
    plt.savefig("opt_cnn_model-{n_samples}-confusion.png")
