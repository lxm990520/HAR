import os
import cv2
import numpy as np
import pandas as pd
USER_LIST = ['1','2','3','4','5','6','7','8','9','10']
SENSOR_LIST = ['Acc1', 'Acc2', 
               'Gyro1', 'Magnet1', 'Acc3', 'Gyro2', 'Magnet2']
GT_TYPES = ['Standing still', 'Sitting and relaxing', 'Lying down','Walking',
            'Climbing stairs', 'Waist bends forward', 'Frontal elevation of arms',
            'Knees bending (crouching)', 'Cycling', 'Jogging', 'Running',
            'Jump front & back']
HEADER_DICT = {'Acc1': ['Acc1_x', 'Acc1_y', 'Acc1_z'],
               'Electro1':'Electro1',
               'Electro2':'Electro2',
               'Acc2': ['Acc2_x', 'Acc2_y', 'Acc2_z'],
               'Gyro1': ['Gyro1_x', 'Gyro1_y', 'Gyro1_z'],
               'Magnet1': ['Magnet1_x', 'Magnet1_y', 'Magnet1_z'],
               'Acc3': ['Acc3_x', 'Acc3_y', 'Acc3_z'],
               'Gyro2': ['Gyro2_x', 'Gyro2_y', 'Gyro2_z'],
               'Magnet2': ['Magnet2_x', 'Magnet2_y', 'Magnet2_z'],
               'Label': 'Label'}
ROOTDIR = os.getcwd()
DATA_DIR = os.path.join(ROOTDIR,"GAFjpg3d"+"\\f150")
def takeNumber(elem):
    return int(elem.split('_')[-2])
#def takeUser(elem):
#    return elem.split('_')[0]
def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path+' 创建成功')
        return True
    else:
        print(path+' 目录已存在')
        return False
def pic20to1(src_dir, dst_dir, pic_size,set_amount):
    #gather all the pictures
    upLimit = 290
    downLimit = 0
    acce_list = []
    gyro_list = []
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if upLimit>=int(f.split('_')[-1].split('.')[0])>downLimit:
                if f.split('_')[2] == 'Acc1':
                    acce_list.append(f)
                elif f.split('_')[2] == 'Gyro1':
                    gyro_list.append(f)





    #first round sort by the user
    #acce_list.sort(key=takeUser)
    #gyro_list.sort(key=takeUser)

    for i in range(len(acce_list)//set_amount):
        temp = acce_list[set_amount*i:set_amount*(i+1)]
        #second round sort by the order number
        temp.sort(key=takeNumber)
        #update the acce_list
        acce_list[set_amount * i:set_amount * (i + 1)] = temp
        temp = gyro_list[set_amount*i:set_amount*(i+1)]
        temp.sort(key=takeNumber)
        gyro_list[set_amount*i:set_amount*(i+1)] = temp

    pic20base = np.zeros([2*pic_size, 10*pic_size, 3])

    count=0
    index=1
    for acc,gyr in zip(acce_list, gyro_list):
        if count<10:
            acc_pic = cv2.imread(os.path.join(src_dir,acc))
            # acc_pic = cv2.cvtColor(acc_pic, cv2.COLOR_RGB2GRAY)
            acc_pic = cv2.resize(acc_pic, (pic_size,pic_size))
            gyr_pic = cv2.imread(os.path.join(src_dir, gyr))
            # gyr_pic = cv2.cvtColor(gyr_pic, cv2.COLOR_RGB2GRAY)
            gyr_pic = cv2.resize(gyr_pic, (pic_size,pic_size))

            pic20base[0:pic_size,count*pic_size:(count+1)*pic_size,:] = acc_pic
            pic20base[pic_size:, count*pic_size:(count+1)*pic_size,:] = gyr_pic

            if count==9:
                save_name = acc.split('_')[0] + '_' + acc.split('_')[1] + '_' + str(index) + '.jpg'
                if not os.path.exists(save_name):
                    cv2.imwrite(os.path.join(dst_dir, save_name), pic20base)
                else:
                    print("{} already here".format(save_name))
                count=0
                index+=1
            else:
                count+=1
dst_dir = os.path.join(ROOTDIR,"GAF20")
mkdir(dst_dir)
for user in USER_LIST:
    user_dir = os.path.join(DATA_DIR,"USER_"+user)
    for type in GT_TYPES:
        type_dir = os.path.join(user_dir,type)
        dst_type_dir = os.path.join(dst_dir,type)
        mkdir(dst_type_dir)
        print(type_dir)
        pic20to1(type_dir,dst_type_dir,pic_size = 200, set_amount = 290)


def rename(dir):
    for root,dirs,files in os.walk(dir):
        for name in files:
            new_name = name.split('.', 2)[0]+'.nexus41.'+name.split('.',2)[-1]
            os.rename(os.path.join(root,name),os.path.join(root,new_name))

# rename('GAF4ZS/nexus_GASF_100')