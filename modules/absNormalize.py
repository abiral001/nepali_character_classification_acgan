import cv2
from torchvision.transforms import transforms
import torch
from absParams import ABSparams
import os
import random
import math
from torch.utils.data import DataLoader

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
        split_size = math.ceil(len(image_list)/batch_size)
        print('Splitting the tensor into {} sizes each with {} images'.format(batch_size, split_size))
        tensor_data = torch.cat(image_list).split(batch_size)
        tensor_data = torch.cat(image_list)
        return tensor_data

    def appendAndShuffle(data):
        finalData = list()
        for images, label in data:
            print('Processing for {} label'.format(label))
            label = torch.as_tensor([int(label)])
            for oneImage in images:
                finalData.append((oneImage, label))
        random.shuffle(finalData)
        return finalData

    def dataGenerator(data, batch_size=16):
        return DataLoader(data, batch_size=batch_size)
        