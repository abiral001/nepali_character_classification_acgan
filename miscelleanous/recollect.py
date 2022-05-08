import os
import pandas as pd
import shutil
import cv2
import numpy as np

## for archive 1
# for consonants

for onefolder in sorted(os.listdir('./archive1/nhcd/nhcd/consonants/')):
    newfolder = int(onefolder) + 21
    if str(newfolder) not in os.listdir('./combined_dataset/'):
        os.mkdir('./combined_dataset/'+str(newfolder))
        count = 0
    else:
        count = len(os.listdir('./combined_dataset/'+str(newfolder)+'/'))
        print(count)
    for file in os.listdir('./archive1/nhcd/nhcd/consonants/'+onefolder+'/'):
        if file.endswith(".jpg"):
            shutil.copy('./archive1/nhcd/nhcd/consonants/'+onefolder+'/'+file, './combined_dataset/'+str(newfolder)+'/'+str(newfolder)+'_'+str(count)+'.jpg')
            image =cv2.imread('./archive1/nhcd/nhcd/consonants/'+onefolder+'/'+file, cv2.IMREAD_GRAYSCALE)
            avg_per_row = np.average(image, axis=0)
            full_avg = np.average(avg_per_row, axis=0)
            if full_avg > 127:
                image = 255-image
                os.remove('./combined_dataset/'+str(newfolder)+'/'+str(newfolder)+'_'+str(count)+'.jpg')
                cv2.imwrite('./combined_dataset/'+str(newfolder)+'/'+str(newfolder)+'_'+str(count)+'.jpg', image)
            count+=1

# for numerals
for onefolder in os.listdir('./archive1/nhcd/nhcd/numerals/'):
    newfolder = str(onefolder)
    if newfolder not in os.listdir("./combined_dataset/"):
        os.mkdir("./combined_dataset/"+newfolder)
        count = 0
    else:
        count = len(os.listdir("./combined_dataset/"+newfolder+"/"))
    print('copying file in {} folder'.format(newfolder))
    for oneFile in os.listdir('./archive1/nhcd/nhcd/numerals/'+onefolder):
        if oneFile.endswith('.jpg'):
            shutil.copy('./archive1/nhcd/nhcd/numerals/'+onefolder+'/'+ oneFile, './combined_dataset/'+newfolder+"/")
            newName = newfolder+'_'+str(count)
            os.rename('./combined_dataset/'+newfolder+"/"+oneFile, './combined_dataset/'+newfolder+"/"+newName+".jpg")
            image = cv2.imread('./combined_dataset/'+newfolder+"/"+newName+".jpg", cv2.IMREAD_GRAYSCALE)
            avg_color_per_row = np.average(image, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            if (avg_color > 127): 
                image = 255-image
                os.remove('./combined_dataset/'+newfolder+"/"+newName+".jpg")
                cv2.imwrite('./combined_dataset/'+newfolder+"/"+newName+".jpg", image)
            count+=1

# for vowels
for onefolder in os.listdir('./archive1/nhcd/nhcd/vowels/'):
    newfolder = str(int(onefolder)+9)
    os.mkdir("./combined_dataset/"+newfolder)
    print('copying file in {} folder'.format(newfolder))
    for count, oneFile in enumerate(os.listdir('./archive1/nhcd/nhcd/vowels/'+onefolder)):
        if oneFile.endswith('.jpg'):
            shutil.copy('./archive1/nhcd/nhcd/vowels/'+onefolder+'/'+ oneFile, './combined_dataset/'+newfolder+"/")
            newName = newfolder+'_'+str(count)
            os.rename('./combined_dataset/'+newfolder+"/"+oneFile, './combined_dataset/'+newfolder+"/"+newName+".jpg")
            image = cv2.imread('./combined_dataset/'+newfolder+"/"+newName+".jpg", cv2.IMREAD_GRAYSCALE)
            avg_color_per_row = np.average(image, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            if (avg_color > 127): 
                image = 255-image
                os.remove('./combined_dataset/'+newfolder+"/"+newName+".jpg")
                cv2.imwrite('./combined_dataset/'+newfolder+"/"+newName+".jpg", image)

## for archive
# for train

for onefolder in os.listdir('./archive/dhcd/train'):
    if int(onefolder)>9:
        newfolder = str(int(onefolder)+12)
    else:
        newfolder = onefolder
    if newfolder not in os.listdir('./combined_dataset/'):
        os.mkdir("./combined_dataset/"+newfolder)
        count = 0
    else:
        count = len(os.listdir('./combined_dataset/'+newfolder))
        print('Folder already exists with count = {}'.format(count))
    print('copying file in {} folder'.format(newfolder))
    for oneFile in os.listdir('./archive/dhcd/train/'+onefolder):
        if oneFile.endswith('.jpg') or oneFile.endswith('.png'):
            shutil.copy('./archive/dhcd/train/'+onefolder+'/'+ oneFile, './combined_dataset/'+newfolder+"/")
            newName = newfolder+'_'+str(count)
            os.rename('./combined_dataset/'+newfolder+"/"+oneFile, './combined_dataset/'+newfolder+"/"+newName+".jpg")
            image = cv2.imread('./combined_dataset/'+newfolder+"/"+newName+".jpg", cv2.IMREAD_GRAYSCALE)
            avg_color_per_row = np.average(image, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            if (avg_color > 127): 
                image = 255-image
                os.remove('./combined_dataset/'+newfolder+"/"+newName+".jpg")
                cv2.imwrite('./combined_dataset/'+newfolder+"/"+newName+".jpg", image)
            count+=1

# for test

for onefolder in os.listdir('./archive/dhcd/test'):
    if int(onefolder)>9:
        newfolder = str(int(onefolder)+12)
    else:
        newfolder = onefolder
    if newfolder not in os.listdir('./combined_dataset/'):
        os.mkdir("./combined_dataset/"+newfolder)
        count = 0
    else:
        count = len(os.listdir('./combined_dataset/'+newfolder))
        print('Folder already exists with count = {}'.format(count))
    print('copying file in {} folder'.format(newfolder))
    for oneFile in os.listdir('./archive/dhcd/test/'+onefolder):
        if oneFile.endswith('.jpg') or oneFile.endswith('.png'):
            shutil.copy('./archive/dhcd/test/'+onefolder+'/'+ oneFile, './combined_dataset/'+newfolder+"/")
            newName = newfolder+'_'+str(count)
            os.rename('./combined_dataset/'+newfolder+"/"+oneFile, './combined_dataset/'+newfolder+"/"+newName+".jpg")
            image = cv2.imread('./combined_dataset/'+newfolder+"/"+newName+".jpg", cv2.IMREAD_GRAYSCALE)
            avg_color_per_row = np.average(image, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            if (avg_color > 127): 
                image = 255-image
                os.remove('./combined_dataset/'+newfolder+"/"+newName+".jpg")
                cv2.imwrite('./combined_dataset/'+newfolder+"/"+newName+".jpg", image)
            count+=1

# dataset link:
# archive1: https://www.kaggle.com/ashokpant/devanagari-character-dataset
# archive: https://www.kaggle.com/ashokpant/devanagari-character-dataset-large