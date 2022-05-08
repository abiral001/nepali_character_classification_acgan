from modules.absNormalize import ABSNormalize
from modules.absModel import absGenerator, absDiscriminator
from modules.absParams import ABSparams
import torch
import numpy as np
from torch import optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision

class ABStraining:
    def __init__(self, device):
        self.device = device
        self.params = ABSparams()
        self.normalizer = ABSNormalize()
        self.generator = absGenerator(self.params.random_noise).to(self.device)
        self.discriminator = absDiscriminator().to(self.device)
        self.test_noise = torch.FloatTensor(self.params.batch_size, 110, 1, 1).normal_(0, 1)
        self.test_noise_ = np.random.normal(0, 1, (self.params.batch_size, 110))
        self.test_label = np.random.randint(0, 58, self.params.batch_size)
        self.test_onehot = np.zeros((self.params.batch_size, 58))
        self.test_onehot[np.arange(self.params.batch_size), self.test_label] = 1
        self.test_noise_[np.arange(self.params.batch_size), :58] = self.test_onehot[np.arange(self.params.batch_size)]
        self.test_noise_ = (torch.from_numpy(self.test_noise_))
        self.test_noise.data.copy_(self.test_noise_.view(self.params.batch_size, 110, 1, 1))
        self.test_noise = self.test_noise.cuda()
        self.writerFakeImg = SummaryWriter(f"logs/AC_GAN/fake/")
        self.writerRealImg = SummaryWriter(f"logs/AC_GAN/real/")
        self.writerAccuracy = SummaryWriter(f"logs/AC_GAN/accuracy/")
        self.writerGenLoss = SummaryWriter(f"logs/AC_GAN/gloss")
        self.writerDisLoss = SummaryWriter(f"logs/AC_GAN/dloss")
        self.writerGenGraph = SummaryWriter(f"logs/AC_GAN/ggraph")
        self.writerDisGraph = SummaryWriter(f"logs/AC_GAN/dgraph")

    def startTraining(self):
        self.generator.apply(self.normalizer.weights_init)
        optimD = optim.Adam(self.discriminator.parameters(), self.params.lr)
        optimG = optim.Adam(self.generator.parameters(), self.params.lr)
        source_err = nn.BCELoss()
        class_err = nn.NLLLoss()
        for epoch in range(self.params.epochs):
            if epoch <= 150:
                continue
            actualData = self.normalizer.dataGenerator(self.normalizer.processImages(), self.params.batch_size)
            batches = len(actualData)
            for idx, data in enumerate(actualData):
                # training dis
                
                ## using real data
                optimD.zero_grad()
                image, label = data
                image, label = image.cuda(), label.cuda()
                image = image.view(-1, 1, 28, 28)
                label = label.view(-1)
                source_, class_ = self.discriminator(image)
                
                real_label = torch.FloatTensor(image.size()[0]).cuda()
                real_label.fill_(1)  
                r_label = real_label.unsqueeze(1)
                source_error = source_err(source_, r_label)
                class_error = class_err(class_, label)
                real_error = source_error+class_error
                real_error.backward()
                optimD.step()
                
                accuracy = self.normalizer.compute_accuracy(class_, label)
                
                ## using fake data generated using generator
                
                noise_ = np.random.normal(0,1, (self.params.batch_size, 110))
                label = np.random.randint(0, 58, self.params.batch_size)
                
                noise=((torch.from_numpy(noise_)).float())
                noise = noise.cuda()
                label = ((torch.from_numpy(label)).long())
                label = label.cuda()
                
                generated_image = self.generator(noise)
                
                source_, class_ = self.discriminator(generated_image.detach())
                fake_label = torch.FloatTensor(generated_image.size()[0]).cuda()
                fake_label.fill_(0)
                f_label = fake_label.unsqueeze(1)
                source_error = source_err(source_, f_label)
                class_error = class_err(class_, label)
                fake_error = source_error+class_error
                fake_error.backward()
                optimD.step()
                
                # training gen
                # the gen is already fed while generating image
                
                self.generator.zero_grad()
                source_, class_ = self.discriminator(generated_image)
                real_label = torch.FloatTensor(generated_image.size()[0]).cuda()
                real_label.fill_(1)
                r_label = real_label.unsqueeze(1)
                source_error = source_err(source_, r_label)
                class_error = class_err(class_, label)
                generator_error = source_error+class_error
                generator_error.backward()
                optimG.step()
                discriminator_loss = (real_error+fake_error)/2
                if idx == 0:
                    print("Epoch {}/{},batch:{}/{}, discriminator_loss = {}, generator_loss = {}, classification_accuracy = {}".format(epoch+1, self.params.epochs, idx, batches ,discriminator_loss, generator_error, accuracy))
                    with torch.no_grad():
                        constructed = self.generator(self.test_noise).reshape(-1, 1, 28, 28)
                        fromDataset, _ = data
                        fromDataset = fromDataset.reshape(-1, 1, 28, 28)
                        img_grid_fake = torchvision.utils.make_grid(constructed, normalize=True)
                        img_grid_real = torchvision.utils.make_grid(fromDataset, normalize=True)
                        self.writerFakeImg.add_image(
                            "ACGAN_fake", img_grid_fake, global_step = epoch+1
                        )
                        self.writerRealImg.add_image(
                            "ACGAN_real", img_grid_real, global_step = epoch+1
                        )
                        self.writerAccuracy.add_scalar("Discriminator Accuracy", accuracy, global_step=epoch)
                        self.writerDisLoss.add_scalar("Discriminator Loss", discriminator_loss, global_step=epoch)
                        self.writerGenLoss.add_scalar("Generator Loss", generator_error, global_step=epoch)
                        self.writerGenGraph.add_graph(self.generator, self.test_noise)
                        self.writerDisGraph.add_graph(self.discriminator, image)

        