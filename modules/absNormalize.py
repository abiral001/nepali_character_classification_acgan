import cv2
from torchvision.transforms import transforms
import torch
from absParams import ABSparams
import os
import random

class ABSNormalize:

    def normalizeImages(self, imagePath, batch_size = 16):
        image_list = list()
        params = ABSparams()
        images = os.listdir(imagePath)
        print('{} no of images found in folder path {}'.format(len(os.listdir(imagePath)), imagePath))
        for one_image in images: 
            image_read = cv2.imread(imagePath+one_image, cv2.IMREAD_GRAYSCALE)
            image_resize = cv2.resize(image_read, (params.image_size, params.image_size))
            image_resize = image_resize.reshape(params.image_size, params.image_size, params.color_depth)
            mean = image_resize.mean()
            sd = image_resize.std()
            image_normalize = (image_resize - mean)/ (mean * sd)
            image_tensor = transforms.ToTensor()
            tensor_img = image_tensor(image_normalize)
            image_list.append(tensor_img)
        tensor_data = torch.cat(image_list)
        return self.shuffleAndSplit(tensor_data, batch_size)

    def shuffleAndSplit(self, data, batchSize):
        finalData = list()
        for images, label in data:
            print('Processing for {} label'.format(label))
            label = torch.as_tensor([int(label)])
            for oneImage in images:
                finalData.append(oneImage)
        random.shuffle(finalData)
        finalData = torch.cat(finalData).split(batchSize)
        return finalData
        