import numpy as np
from absParams import HParams

class ABSLoss():
    def __init__(self, predicted,actual):
        self.predicated = predicted
        self.actual = actual
        self.hp = HParams()
        self.computedloss =self.computeloss(predicted, actual)
        
    def comuteloss(self, predicted, actual):
        
        predicted_new = [max(i, self.hp.epsilon) for i in predicted]
        predicted_old = [min(i, 1-self.hp.epsilon) for i in actual]
        predicted_new = np.array(predicted_new)
        predicted_old = np.array(predicted_old)
        
        return -sum(predicted_old * np.log(predicted_new))
        