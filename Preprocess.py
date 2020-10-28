from util import *
import os
from Configuration import Configuration
from GAF import GAF
from check import is_valid_jpg, is_valid_png
#=======================Configuration Modification=====================
config = Configuration()
config.INTERVAL_LENGTH = 50
config.WINDOW_LENGTH = 25
config.DATASET = 'MHEALTH'
config.USER_LIST = ['1','2','3','4','5','6','7','8','9','10']
config.GT_LIST = ['Standing still', 'Sitting and relaxing', 'Lying down','Walking',
            'Climbing stairs', 'Waist bends forward', 'Frontal elevation of arms',
            'Knees bending (crouching)', 'Cycling', 'Jogging', 'Running', 
            'Jump front & back']
#config.SENSOR_LIST = ['Acc1', 'Gyro1']
config.SENSOR_LIST = ['Acc1', 'Acc2', 'Acc3', 'Gyro1', 'Gyro2', 'Magnet1', 'Magnet2']
config.DEVICE_LIST = ['Unknown']
config.fresh()
#======================================================================
print('=====Preprocess start=====')
#=======================Time-series Data to GAF Figure=================
print('=====GAF start=====')
# GAF(True,config)
print('=====GAF finished=====')
#======================================================================
#===========================GAF Figure 20 to 1=========================
print('=====Nto1 start=====')
dst_dir = os.path.join(config.DATASET)
dst_dir = os.path.join(dst_dir, 'GAF4ZS')
dst_dir = os.path.join(dst_dir,'f'+str(config.INTERVAL_LENGTH))
src_dir = os.path.join(config.DATASET)
src_dir = os.path.join(src_dir, 'GAFjpg3d')
src_dir = os.path.join(src_dir,'f'+str(config.INTERVAL_LENGTH))
for device in config.DEVICE_LIST:
    dst_device_path = os.path.join(dst_dir,device)
    dst_device_path = os.path.join(dst_device_path,'train')
    src_device_path = os.path.join(src_dir,device)
    src_device_path = os.path.join(src_device_path, 'train')
    for gt in config.GT_LIST:
        dst_gt_path = os.path.join(dst_device_path,gt)
        src_gt_path = os.path.join(src_device_path,gt)
        if not os.path.exists(dst_gt_path):
            mkdir(dst_gt_path)
        picNto1(src_gt_path,dst_gt_path,config)
        print(gt + ' finished!')
    print(device + ' finished!')
print('=====Nto1 finished=====')
#===========================Seperate Train Test Set===================
print('=====Seperate start=====')
dst_dir = os.path.join(config.DATASET)
dst_dir = os.path.join(dst_dir, 'GAF4ZS')
dst_dir = os.path.join(dst_dir, 'f' + str(config.INTERVAL_LENGTH))

for device in config.DEVICE_LIST:
    dst_device_path = os.path.join(dst_dir, device)
    moveTrainTest(os.path.join(dst_device_path))
    print(device + ' finished!')
print('=====Seperate finished=====')
print('=====Preprocess finished=====')
#==========================Check if Every pic is valid================
print('=====Check start!=====')
for set in os.listdir(dst_device_path):
    set_path = os.path.join(dst_device_path, set)
    for gt in os.listdir(set_path):
        gt_path = os.path.join(set_path, gt)
        for file in os.listdir(path):
            pic_file = os.path.join(gt_path, file)
            isJpg = is_valid_jpg(pic_file)
            isPng = is_valid_png(pic_file)
            if not isJpg:
                print("jpeg : %s, png %s, file %s " % (isJpg, isPng, file))
    print(gt + ' finished!')
print(set + ' finished')
print('=====Check finished!=====')