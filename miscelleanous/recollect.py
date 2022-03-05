import os
import pandas as pd
import shutil
import cv2

## for archive 1
# for consonants

for onefolder in os.listdir('./archive1/nhcd/nhcd/consonants/'):
    newfolder = str(int(onefolder)+22)
    os.mkdir("./combined_dataset/"+newfolder)
    print('copying file in {} folder'.format(newfolder))
    for count, oneFile in enumerate(os.listdir('./archive1/nhcd/nhcd/consonants/'+onefolder)):
        if oneFile.endswith('.jpg'):
            shutil.copy('./archive1/nhcd/nhcd/consonants/'+onefolder+'/'+ oneFile, './combined_dataset/'+newfolder+"/")
            newName = newfolder+'_'+str(count)
            os.rename('./combined_dataset/'+newfolder+"/"+oneFile, './combined_dataset/'+newfolder+"/"+newName+".jpg")

# for vowels
for onefolder in os.listdir('./archive1/nhcd/nhcd/vowels/'):
    if int(onefolder)<7:
        newfolder = str(int(onefolder)+9)
    else:
        newfolder = str(int(onefolder)+10)
    os.mkdir("./combined_dataset/"+newfolder)
    print('copying file in {} folder'.format(newfolder))
    for count, oneFile in enumerate(os.listdir('./archive1/nhcd/nhcd/vowels/'+onefolder)):
        if oneFile.endswith('.jpg'):
            shutil.copy('./archive1/nhcd/nhcd/vowels/'+onefolder+'/'+ oneFile, './combined_dataset/'+newfolder+"/")
            newName = newfolder+'_'+str(count)
            os.rename('./combined_dataset/'+newfolder+"/"+oneFile, './combined_dataset/'+newfolder+"/"+newName+".jpg")

## for archive
# for train

for onefolder in os.listdir('./archive/dhcd/train'):
    if int(onefolder)>9:
        newfolder = str(int(onefolder)+13)
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
            if int(onefolder) > 9:
                image = cv2.imread('./combined_dataset/'+newfolder+"/"+newName+".jpg", cv2.IMREAD_GRAYSCALE)
                image = 255-image
                os.remove('./combined_dataset/'+newfolder+"/"+newName+".jpg")
                cv2.imwrite('./combined_dataset/'+newfolder+"/"+newName+".jpg", image)
            count+=1

# for test

for onefolder in os.listdir('./archive/dhcd/test'):
    if int(onefolder)>9:
        newfolder = str(int(onefolder)+13)
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
            image = 255-image
            os.remove('./combined_dataset/'+newfolder+"/"+newName+".jpg")
            cv2.imwrite('./combined_dataset/'+newfolder+"/"+newName+".jpg", image)
            count+=1

# dataset link:
# archive1: https://www.kaggle.com/ashokpant/devanagari-character-dataset
# archive: https://www.kaggle.com/ashokpant/devanagari-character-dataset-large