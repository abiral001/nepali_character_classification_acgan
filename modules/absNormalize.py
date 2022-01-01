import cv2
from torchvision.transforms import transforms
import torch
from absParams import ABSparams

class ABSNormalize:

    def normalizeImages(images, batch_size = 16):
        image_list = list()
        params = ABSparams()
        for one_image in images: 
            image_read = cv2.imread(one_image, cv2.IMREAD_GRAYSCALE)
            image_resize = cv2.resize(image_read, (params.image_size, params.image_size))
            image_resize = image_resize.reshape(params.image_size, params.image_size, params.color_depth)
            mean = image_resize.mean()
            sd = image_resize.std()
            image_normalize = (image_resize - mean)/ (mean * sd)
            image_tensor = transforms.ToTensor()
            tensor_img = image_tensor(image_normalize)
            image_list.append(tensor_img)
        tensor_data = torch.cat(image_list).split(int(len(image_list)/batch_size))
        return tensor_data