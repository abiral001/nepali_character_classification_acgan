from absNormalize import ABSNormalize
from absModel import absGen, absDis
from absParams import ABSparams

class ABStraining:
    def __init__(self, device):
        self.device = device
        self.params = ABSparams()
        self.normalizer = ABSNormalize()
        self.generator = absGen(self.params.random_noise, self.params.kernel_size_conv).to(self.device)
        self.discriminator = absDis(self.params.image_size, self.params.kernel_size_conv).to(self.device)

    def startTraining():
        pass

    