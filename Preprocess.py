from util import *
import os
from Configuration import Configuration
from GAF import GAF

#=======================Configuration Modification=====================
config = Configuration()
config.INTERVAL_LENGTH = 200
config.WINDOW_LENGTH = 100
config.TIME_STEPS = 1
config.DATASET = 'HHAR'
config.TEST_USER = 'a'
config.USER_LIST = ['a','b','c','d','e','f','g','h','i']
config.GT_LIST = ['stand','sit','walk','stairsup','stairsdown','bike']
config.SENSOR_LIST = ['acce','gyro']
config.DEVICE_LIST = ['nexus41','nexus42','s3mini1','s3mini2']

config.fresh()
#======================================================================
print('=====Preprocess start=====')
#=======================Time-series Data to GAF Figure=================
print('=====GAF start=====')
#GAF(True,config)
print('=====GAF finished=====')
#======================================================================
#===========================GAF Figure 20 to 1=========================
print('=====Nto1 start=====')
# dst_dir = os.path.join(config.DATASET)
# dst_dir = os.path.join(dst_dir, 'GAF4ZSFC')
# dst_dir = os.path.join(dst_dir,'f'+str(config.INTERVAL_LENGTH))
# src_dir = os.path.join(config.DATASET)
# src_dir = os.path.join(src_dir, 'GAFjpg3d')
# src_dir = os.path.join(src_dir,'f'+str(config.INTERVAL_LENGTH))
# for device in config.DEVICE_LIST:
#     dst_device_path = os.path.join(dst_dir,device)
#     dst_device_path = os.path.join(dst_device_path,'train')
#     src_device_path = os.path.join(src_dir,device)
#     src_device_path = os.path.join(src_device_path, 'train')
#     for gt in config.GT_LIST:
#         dst_gt_path = os.path.join(dst_device_path,gt)
#         src_gt_path = os.path.join(src_device_path,gt)
#         if not os.path.exists(dst_gt_path):
#             mkdir(dst_gt_path)
#         picNto1(src_gt_path,dst_gt_path,config)
#         print(gt + ' finished!')
#     print(device + ' finished!')
print('=====Nto1 finished=====')
#===========================Seperate Train Test Set===================
print('=====Seperate start=====')
dst_dir = os.path.join(config.DATASET)
dst_dir = os.path.join(dst_dir, 'GAF4ZSFC')
dst_dir = os.path.join(dst_dir, 'f' + str(config.INTERVAL_LENGTH))

for device in config.DEVICE_LIST:
    dst_device_path = os.path.join(dst_dir, device)
    moveTrainTest(os.path.join(dst_device_path),config)
    print(device + ' finished!')
print('=====Seperate finished=====')
print('=====Preprocess finished=====')
#==========================Check if Every pic is valid================
print('=====Check start!=====')
dst_dir = os.path.join(config.DATASET)
dst_dir = os.path.join(dst_dir, 'GAF4ZSFC')
dst_dir = os.path.join(dst_dir, 'f' + str(config.INTERVAL_LENGTH))
for device in config.DEVICE_LIST:
    for set in os.listdir(dst_device_path):
        set_path = os.path.join(dst_device_path, set)
        for gt in os.listdir(set_path):
            gt_path = os.path.join(set_path, gt)
            for file in os.listdir(gt_path):
                pic_file = os.path.join(gt_path, file)
                isJpg = is_valid_jpg(pic_file)
                isPng = is_valid_png(pic_file)
                if not isJpg:
                    print("jpeg : %s, png %s, file %s " % (isJpg, isPng, file))
        print(set + ' finished!')
    print(device + ' finished!')
print('=====Check finished!=====')