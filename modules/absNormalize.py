import cv2
from torchvision.transforms import transforms
import torch
from modules.absParams import ABSparams
import os
import random

class ABSNormalize:
    def __init__(self):
        self.params = ABSparams()

    def __normalizeImages(self, imagePath):
        image_list = list()
        images = os.listdir(imagePath)
        print('{} no of images found in folder path {}'.format(len(os.listdir(imagePath)), imagePath))
        for one_image in images: 
            image_read = cv2.imread(imagePath+one_image, cv2.IMREAD_GRAYSCALE)
            image_resize = cv2.resize(image_read, (self.params.image_size, self.params.image_size))
            image_resize = image_resize.reshape(self.params.image_size, self.params.image_size, self.params.color_depth)
            mean = image_resize.mean()
            sd = image_resize.std()
            image_normalize = (image_resize - mean)/ (mean * sd)
            image_tensor = transforms.ToTensor()
            tensor_img = image_tensor(image_normalize)
            image_list.append(tensor_img)
        tensor_data = torch.cat(image_list)
        return tensor_data

    def processImages(self):
        data = list()
        for label in os.listdir('./combined_dataset/'):
            if not label.endswith('.csv'):
                data.append((self.__normalizeImages('./combined_dataset/'+label+'/', self.params.batch_size), label))
        finalData = list()
        for images, label in data:
            print('Processing for {} label'.format(label))
            label = torch.as_tensor([int(label)])
            for oneImage in images:
                finalData.append((oneImage, label))
        return finalData

    def dataGenerator(self, fullData, batch_size):
        random.shuffle(fullData)
        dataloader = torch.utils.data.DataLoader(fullData, batch_size=batch_size)
        return dataloader

    def compute_accuracy(prediction, actual):
        correct = 0
        prediction_ = prediction.data.max(1)[1]
        correct = (prediction_ == actual).sum().item()
        acc = float(correct)/float(len(actual.data))*100.0
        return acc

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        