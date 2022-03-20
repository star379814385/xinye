import numpy as np
from sklearn.metrics import confusion_matrix


class MetricsMethod(object):

    @staticmethod
    def cul_acc(y, y_pred):
        """

        :param y:       type: (np.int32), shape: (b, *)
        :param y_pred:  type: (np.float), shape: (b, c, *)
        :return:        type: float32
        """
        y_pred = np.argmax(y_pred, axis=1)
        return np.mean(y == y_pred)

    @staticmethod
    def cul_miou(y, y_pred):
        """

        :param y:       type: (np.int32), shape: (b, *)
        :param y_pred:  type: (np.float), shape: (b, c, *)
        :return:        type: float32
        """
        # iou = tp / (tp + fn + fp)
        y_pred = np.argmax(y_pred, axis=1)
        y = y.flatten()
        y_pred = y_pred.flatten()
        confusion_metrix = confusion_matrix(y, y_pred)
        tp = np.diag(confusion_metrix)
        fn = np.sum(confusion_metrix, axis=1)
        fp = np.sum(confusion_metrix, axis=0)
        iou = tp / (tp + fn + fp)
        # 不去除
        miou = np.nanmean(iou)
        # 去除背景类
        # miou = np.nanmean(iou[1:])
        return miou

    @staticmethod
    def cul_recall(y, y_pred, index):
        """

        :param y:       type: (np.int32), shape: (b, *)
        :param y_pred:  type: (np.float), shape: (b, c, *)
        :return:        type: float32
        """
        # get index by index.
        y_pred = np.argmax(y_pred, axis=1)
        y = y.flatten()
        y_pred = y_pred.flatten()
        y = np.where(y == index, True, False)
        y_pred = np.where(y_pred == index, True, False)
        # recall = tp / (tp + fn)
        tp = np.sum(y == y_pred)
        fn = np.sum(y == ~y_pred)
        recall = tp / (tp + fn)
        return recall

    @staticmethod
    def cul_precision(y, y_pred, index):
        """

        :param y:       type: (np.int32), shape: (b, *)
        :param y_pred:  type: (np.float), shape: (b, c, *)
        :return:        type: float32
        """
        # get object by index.
        y_pred = np.argmax(y_pred, axis=1)
        y = y.flatten()
        y_pred = y_pred.flatten()
        y = np.where(y == index, True, False)
        # precision = tp / (tp + fp)
        y_pred = np.where(y_pred == index, True, False)
        tp = np.sum(y == y_pred)
        fp = np.sum(~y == y_pred)
        precision = tp / (tp + fp)
        return precision


class Metrics(object):

    def __init__(self):
        self.y = []
        self.y_pred = []

    def reset(self):
        self.y = []
        self.y_pred = []

    def add_batch(self, y, y_pred):
        self.y = self.y.append(y)
        self.y_pred = self.y_pred.append(y_pred)

    def apply(self):
        return {}


if __name__ == "__main__":
    a = np.array([0, 1, 1, 1, 1])
    b = np.array([0, 1, 1, 0, 1])
    c = np.array([0, 0, 0, 0, 0])
    d = np.where(a == 1, True, False)
    e = np.where(b == 1, True, False)
    print(d)
    print(e)

    tp = np.sum(np.bitwise_and(d, e))
    fn = np.sum(np.bitwise_and(d, ~e))
    print(tp, fn)
