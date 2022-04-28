import numpy as np
from
from loss.loss import Loss


class CELoss(Loss):

    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape
