import numpy as np
from sklearn.metrics import confusion_matrix                            
from time import time

class Metrics(object):

    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.y = []
        self.y_pred = []
        self.confusion_matrix = None

    def add_batch(self, y, y_pred):
        self.y.append(y.flatten())
        self.y_pred.append(y_pred.flatten())



    def cul_acc(self):
        return np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)

    # def my_cm(self, y, y_pred):
    #     confusion_matrix
    #     confusion_matrix = np.array([[0, 0], [0, 0]])
    #     confusion_matrix[0][0] = np.sum((1 - y) & (1 - y_pred))
    #     confusion_matrix[0][1] = np.sum((1 - y) & y_pred)
    #     confusion_matrix[1][0] = np.sum(y & (1 - y_pred))
    #     confusion_matrix[1][1] = np.sum(y & y_pred)
    #     return confusion_matrix

    def apply(self):
        self.confusion_matrix = confusion_matrix(np.concatenate(self.y), np.concatenate(self.y_pred))
        # self.confusion_matrix = self.my_cm(np.concatenate(self.y).astype(np.uint8), np.concatenate(self.y_pred).astype(np.uint8))

        return {
            "ACC": self.cul_acc(),
        }


if __name__ == "__main__":
    pass
