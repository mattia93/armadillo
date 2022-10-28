from constants import *
import numpy as np
import pickle
import sklearn.metrics
from sklearn import metrics
from IPython.display import Image as ipy_disp_img
import matplotlib.pyplot as plt

# general functions
def print_keras_model(model, model_name: str):
    open(f"{model_name}.png", "a+")
    tf.keras.utils.plot_model(
        model, show_shapes=True, show_layer_names=True, to_file=f"{model_name}.png"
    )
    return ipy_disp_img(retina=False, filename=f"{model_name}.png")

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp)

def load_object(filename):
    with open(filename, 'rb') as inp:
        return pickle.load(inp)




# visualization functions
def plot_attack(true_image, attacked_image, model, function):

    zero = 'cyan'
    one = 'red'
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize= (15, 10), constrained_layout=True)

    ax[0,0].imshow(true_image, cmap='gray')
    ax[0,0].set_title("True Image")
    ax[1,0].imshow(attacked_image, cmap='gray')
    ax[1,0].set_title("Attacked Image")

    true_predictions = model.predict(np.expand_dims(true_image, axis=0))
    attacked_predictions = model.predict(np.expand_dims(attacked_image, axis=0))
    # true bars
    true_colors = [zero if x != np.amax(true_predictions[0][:]) else one for x in true_predictions[0][:]]
    ax[0,1].bar(range(true_predictions.shape[1]), true_predictions[0][:], color=true_colors)
    for i, v in enumerate(true_predictions[0][:]):
        ax[0,1].text(range(true_predictions.shape[1])[i] - 0.25, v + 0.01, str("{:.3f}".format(v)))
    plt.sca(ax[0,1])
    plt.axis([-1, 10, 0, 1.1])
    plt.xticks(range(true_predictions.shape[1]), list(CLASS_NAMES.values()), rotation=45)
    # ax[0,1].set_ylim([0.0, 1.0])
    ax[0,1].set_title("True Image Prediction")

    # attacked bars
    attacked_colors = [zero if x != np.amax(attacked_predictions[0][:]) else one for x in attacked_predictions[0][:]]
    ax[1,1].bar(range(attacked_predictions.shape[1]), attacked_predictions[0][:], color=attacked_colors)
    for i, v in enumerate(attacked_predictions[0][:]):
        ax[1,1].text(range(attacked_predictions.shape[1])[i] - 0.25, v + 0.01, str("{:.3f}".format(v)))
    plt.sca(ax[1,1])
    plt.axis([-1, 10, 0, 1.1])
    plt.xticks(range(attacked_predictions.shape[1]), list(CLASS_NAMES.values()), rotation=45)
    # ax[1,1].set_ylim([0.0, 1.0])
    ax[1,1].set_title("Attacked Image Prediction")

    fig.suptitle(f"Attack on {function} model", fontsize=20, y=1.1)

    plt.show()



# model functions

# (TP+TN)/(TP+TN+FP+FN)
def accuracy(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred)

# (TP)/(TP+FP)
def precision(y_true, y_pred):
    return np.average(metrics.precision_score(y_true, y_pred, average=None))

# (TP)/(TP+FN)
def recall(y_true, y_pred):
    return np.average(metrics.recall_score(y_true, y_pred, average=None))

def max_confidence(y_pred):
    return y_pred.max()

def min_confidence(y_pred):
    temp = np.array(
        [
            [y for y in y_pred[i][:] if np.amax(y_pred[i][:]) == y]
            for i in range(y_pred.shape[0])
        ]
    )
    temp = temp.min()
    return temp

def confusion_matrix(y_true, y_pred):
    matrix = sklearn.metrics.confusion_matrix(
        y_true.argmax(axis=1), y_pred.argmax(axis=1)
    )
    return matrix

def most_robust_class(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    trace = np.array([matrix[c][c] / np.sum(matrix[c][:]) for c in range(matrix.shape[1])])
    return CLASS_NAMES[trace.argmax()]

def least_robust_class(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    trace = np.array([matrix[c][c] / np.sum(matrix[c][:]) for c in range(matrix.shape[1])])
    return CLASS_NAMES[trace.argmin()]

def compare_model_metrics(mm1, mm2):
    print(f"Name:\t\t\t\t\t{mm1.name}\t\t{mm2.name}")
    print(f"Accuracy:\t\t\t\t{mm1.accuracy}\t\t\t\t\t{mm2.accuracy}")
    print(f"Precision:\t\t\t\t{mm1.precision}\t\t{mm2.precision}")
    print(f"Recall:\t\t\t\t\t{mm1.recall}\t\t{mm2.recall}")
    print(f"Max_confidence:\t\t\t{mm1.max_confidence}\t\t\t\t\t\t{mm2.max_confidence}")
    print(f"Min_confidence:\t\t\t{mm1.min_confidence}\t\t{mm2.min_confidence}")
    print(f"Most_robust_class:\t\t{mm1.most_robust_class}\t\t\t\t\t\t{mm2.most_robust_class}")
    print(f"Least_robust_class:\t\t{mm1.least_robust_class}\t\t\t\t\t{mm2.least_robust_class}")

class ModelMetrics:
    def __init__(self, name, x_test, y_true, y_pred):
        self.name = name
        self.x_test = x_test
        self.y_true = y_true
        self.y_pred = y_pred
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.max_confidence = 0
        self.min_confidence = 0
        self.confusion_matrix = 0
        self.most_robust_class = 0
        self.least_robust_class = 0
        self.best_attacks = []

    def buil_metrics(self):
        y_pred_new = np.array(
            [
                [
                    1.0 if np.amax(self.y_pred[i][:]) == y else 0.0
                    for y in self.y_pred[i][:]
                ]
                for i in range(self.y_pred.shape[0])
            ]
        )
        self.accuracy = accuracy(self.y_true, y_pred_new)
        self.precision = precision(self.y_true, y_pred_new)
        self.recall = recall(self.y_true, y_pred_new)
        self.max_confidence = max_confidence(self.y_pred)
        self.min_confidence = min_confidence(self.y_pred)
        self.confusion_matrix = confusion_matrix(self.y_true, self.y_pred)
        self.most_robust_class = most_robust_class(self.y_true, self.y_pred)
        self.least_robust_class = least_robust_class(self.y_true, self.y_pred)

    def show_metrics(self):
        print(f"Accuracy: {self.accuracy}")
        print(f"Precision: {self.precision}")
        print(f"Recall: {self.recall}")
        print(f"Max_confidence: {self.max_confidence}")
        print(f"Min_confidence: {self.min_confidence}")
        print(f"Confusion_matrix:\n {self.confusion_matrix}")
        print(f"Most_robust_class: {self.most_robust_class}")
        print(f"Least_robust_class: {self.least_robust_class}")

# optuna study.optimize callback

class StopWhenTrialKeepBeingPrunedCallback:
    def __init__(self, threshold: int):
        self.threshold = threshold
        self.count = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._consequtive_pruned_count += 1
        else:
            self._consequtive_pruned_count = 0

        if self._consequtive_pruned_count >= self.threshold:
            study.stop()