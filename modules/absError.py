import numpy as np
import torch
from absParams import ABSparams

class ABSLoss():
    def __init__(self, predicted,actual):
        self.predicated = predicted
        self.actual = actual
        self.hp = ABSparams()
        self.computedloss =self.computeloss(predicted, actual)
        
    def computeLoss(self, predicted, actual):
        
        predicted_new = [max(i, self.hp.epsilon) for i in predicted]
        predicted_new = [min(i, 1-self.hp.epsilon) for i in predicted_new]
        actual_ = [max(i, self.hp.epsilon) for i in actual]
        actual_ = [min(i, 1-self.hp.epsilon) for i in actual_]
        predicted_new = np.array(predicted_new)
        actual_ = np.array(actual_)
        
        return -sum(actual_ * np.log(predicted_new))
    
    def backward(self,computedLoss):
        ccomputedloss.backward()
        
