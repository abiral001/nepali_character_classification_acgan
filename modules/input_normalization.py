import cv2
from torchvision.transforms import transforms

def normalizeImages(images):
    image_list = list()
    
    for one_image in images: 
        image_read = cv2.imread(one_image)
        image_resize = cv2.resize(image_read, (20, 20))
        mean = image_resize.mean()
        sd = image_resize.std()
        image_normalize = (image_resize - mean)/ (mean * sd)
        image_tensor = transforms.ToTensor()
        tensor_img = image_tensor(image_normalize)

    return tensor_img