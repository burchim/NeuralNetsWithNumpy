from nnet.modules import Module, Softmax
import numpy as np

class MeanAbsoluteError(Module):

    def __init__(self):
        super(MeanAbsoluteError, self).__init__()

    def forward(self, y, y_pred):

        self.y_pred = y_pred
        self.y = y
        loss = abs(self.y_pred - self.y)
        return loss.sum() / loss.shape[0]

    def backward(self, gradient=None):

        return np.where(self.y_pred > self.y, 1, -1) / self.y_pred.shape[0]

class MeanSquaredError(Module):

    def __init__(self):
        super(MeanSquaredError, self).__init__()

    def forward(self, y, y_pred):

        self.y_pred = y_pred
        self.y = y
        loss = (self.y_pred - self.y)**2
        return loss.sum() / loss.shape[0]

    def backward(self, gradient=None):

        return 2 * (self.y_pred - self.y) / self.y_pred.shape[0]

class HuberLoss(Module):

    def __init__(self, delta=1):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y, y_pred):

        self.y_pred = y_pred
        self.y = y
        self.l1 = abs(self.y_pred - self.y)
        loss = np.where(self.l1 < self.delta, 0.5 * self.l1**2, self.delta * (self.l1 - 0.5 * self.delta))
        return loss.sum() / loss.shape[0]

    def backward(self, gradient=None):

        return np.where(self.l1 < self.delta, self.y_pred - self.y, self.delta * np.where(self.y_pred > self.y, 1, -1)) / self.y_pred.shape[0]

class CrossEntropy(Module):

    def __init__(self, ignore_index=-1, reduction='mean'):
        super(CrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, y, y_pred):

        self.y_pred = y_pred
        self.y = y

        # Compute Mask
        self.mask = np.expand_dims(np.where(np.argmax(y, axis=-1) == self.ignore_index, 0, 1), axis=-1).astype(self.y_pred.dtype)

        # Reduction
        if self.reduction == "mean":
            self.n = np.count_nonzero(self.mask)
        else:
            self.n = 1

        # Element Wise Loss
        loss = - self.y * np.log(self.y_pred)

        # Mask Loss
        loss = loss * self.mask

        return loss.sum() / self.n

    def backward(self, gradient=None):

        return - (self.y / self.y_pred) * self.mask / self.n

class SoftmaxCrossEntropy(CrossEntropy):

    def __init__(self, ignore_index=-1, reduction='mean'):
        super(SoftmaxCrossEntropy, self).__init__(ignore_index, reduction)
        self.softmax = Softmax()

    def forward(self, y, x):
        
        return super(SoftmaxCrossEntropy, self).forward(y, self.softmax(x))

    def backward(self, gradient=None):

        return (self.y_pred - self.y) * self.mask / self.n

class BinaryCrossEntropy(Module):

    def __init__(self):
        super(BinaryCrossEntropy, self).__init__()

    def forward(self, y, y_pred):

        self.y_pred = y_pred
        self.y = y
        loss = - (self.y * np.log(self.y_pred) + (1 - self.y) * np.log(1 - self.y_pred))
        return loss.sum() / loss.shape[0]

    def backward(self, gradient=None):

        return - (self.y / self.y_pred - (1 - self.y) / (1 - self.y_pred)) / self.y_pred.shape[0]