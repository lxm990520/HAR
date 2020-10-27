import cv2
import shutil
import os
import io
import numpy as np
import tensorflow as tf
import pandas as pd
from mkdir import mkdir
import matplotlib.pyplot as plt
def resize(dir, save_path, size):
    for root, dirs, files in os.walk(dir):
        for file in files:
            filepath = os.path.join(root,file)
            try:
                image=  cv2.imread(filepath)
                dim=(size,size)
                resized = cv2.resize(image,dim)
                path = os.path.join(save_path,file)
                cv2.imwrite(path, resized)
            except:
                print(filepath)
                os.remove(filepath)
        cv2.waitKey(0)
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
def takeNumber(elem):
    return int(elem.split('_')[-1].split('.')[0])
def takeUser(elem):
    return elem.split('_')[1]
def takeSensor(slem):
    return elem.split('_')[3]

def pic20to1(src_dir, dst_dir, pic_size,set_amount,dim):
    #gather all the pictures
    upLimit = 290
    downLimit = 0
    acce_list = []
    for root, dirs, files in os.walk(acce_dir):
        for f in files:
            if upLimit>=int(f.split('.')[1])>downLimit:
                acce_list.append(f)
    gyro_list = []
    for root, dirs, files in os.walk(gyro_dir):
        for f in files:
            if upLimit>=int(f.split('.')[1])>downLimit:
                gyro_list.append(f)

    #first round sort by the user
    acce_list.sort(key=takeUser)
    gyro_list.sort(key=takeUser)

    for i in range(len(acce_list)//set_amount):
        temp = acce_list[set_amount*i:set_amount*(i+1)]
        #second round sort by the order number
        temp.sort(key=takeNumber)
        #update the acce_list
        acce_list[set_amount * i:set_amount * (i + 1)] = temp
        temp = gyro_list[set_amount*i:set_amount*(i+1)]
        temp.sort(key=takeNumber)
        gyro_list[set_amount*i:set_amount*(i+1)] = temp

    pic20base = np.zeros([dim*pic_size, 10*pic_size, 3])

    count=0
    for acc,gyr in zip(acce_list, gyro_list):
        if count<10:
            acc_pic = cv2.imread(os.path.join(acce_dir,acc))
            # acc_pic = cv2.cvtColor(acc_pic, cv2.COLOR_RGB2GRAY)
            acc_pic = cv2.resize(acc_pic, (pic_size,pic_size))
            gyr_pic = cv2.imread(os.path.join(gyro_dir, gyr))
            # gyr_pic = cv2.cvtColor(gyr_pic, cv2.COLOR_RGB2GRAY)
            gyr_pic = cv2.resize(gyr_pic, (pic_size,pic_size))

            pic20base[0:pic_size,count*pic_size:(count+1)*pic_size,:] = acc_pic
            pic20base[pic_size:, count*pic_size:(count+1)*pic_size,:] = gyr_pic

            if count==9:
                save_name = acc.split('.',1)[0]+'.nexus41.'+acc.split('.',1)[-1]
                if not os.path.exists(save_name):
                    cv2.imwrite(os.path.join(des_dir, save_name), pic20base)
                else:
                    print("{} already here".format(save_name))
                count=0
            else:
                count+=1




def picNto1(src_dir,dst_dir,config,pic_size):
    filelist = os.listdir(src_dir)
    for user in config.USER_LIST:
        filelist_sameuser = [file for file in filelist if file.split('_')[1] == user]

        print(user + ' start!')
        for pic_index in range(len(filelist_sameuser)//config.INPUT_DIM - 10 + 1):
        # 10 is a changable parameter
            row = 0
            picNbase = np.zeros([config.INPUT_DIM*pic_size, 10*pic_size, 3])
            for sensor in config.SENSOR_LIST:
                filelist_sameusersensor = [file for file in filelist_sameuser if file.split('_')[3] == sensor]
                filelist_sameusersensor.sort(key = takeNumber)
                for figure_index in range(10):
                    # 10 is a changable parameter
                    #print(pic_index + figure_index)
                    fig = cv2.imread(os.path.join(src_dir,filelist_sameusersensor[pic_index + figure_index]))
                    fig = cv2.resize(fig, (pic_size,pic_size))
                    picNbase[row * pic_size:(row+1) * pic_size, figure_index * pic_size:(figure_index + 1) * pic_size] = fig
                row += 1
            template = filelist_sameuser[0].split('_')
            template.pop(-1)
            template.pop(-1)
            template.append(str(pic_index))
            #changable, user name can be added
            save_name = '_'.join(template)
            save_name = save_name + '.jpg'
            cv2.imwrite(os.path.join(dst_dir,save_name), picNbase)




def moveTrainTest(dir):
#
    # for root, dirs, files in os.walk(os.path.join(dir,'test')):
    #     for filename in files:
    #         root_list = root.split('\\')
    #         if root_list[-2] == "test":
    #             root_list[-2] = "train"
    #             if True:
    #                 shutil.move(os.path.join(root, filename), os.path.join(*root_list, filename))
    #                 print(filename + "moved")
    #         else:
    #             raise NameError("a file is in a wrong place " + os.path.join(root, filename))
    for root, dirs, files in os.walk(os.path.join(dir,'train')):
        for filename in files:
            rootlist = root.split("\\")
            rootlist[-2] = "test"
            if not os.path.exists(os.path.join(*rootlist)):
                mkdir(os.path.join(*rootlist))
            if int(filename.split('_')[-1].split('.')[0])%5==0:#5 is changable parameter
                shutil.move(os.path.join(root, filename), os.path.join(*rootlist, filename))
                #print(filename + "moved")
def plot_confusion_matrix(cm, labels_name, title, save_dir):
    plt.figure(3)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_dir, format='jpg')
    plt.close(3)
    
if __name__ == '__main__':
    from Configuration import Configuration
    config = Configuration()
    for freq in [200]:
        config.INTERVAL_LENGTH = freq
        dst_dir = os.path.join('GAF4ZS')
        dst_dir = os.path.join(dst_dir, 'f' + str(config.INTERVAL_LENGTH))
        src_dir = os.path.join('GAFjpg3d')
        src_dir = os.path.join(src_dir, 'f' + str(config.INTERVAL_LENGTH))
        for device in config.DEVICE_LIST:
            dst_device_path = os.path.join(dst_dir, device)
            dst_device_path = os.path.join(dst_device_path, 'train')
            src_device_path = os.path.join(src_dir, device)
            for user in config.USER_LIST:
                for gt in config.GT_LIST:
                    dst_gt_path = os.path.join(dst_device_path, gt)
                    src_gt_path = os.path.join(src_device_path, gt)
                    if not os.path.exists(dst_gt_path):
                        mkdir(dst_gt_path)
                    picNto1(src_gt_path, dst_gt_path, config, 200)
                print(device + ' ' + user + ' finished!')
            print(device + ' finished!')
#=============================train2test function=======================
    from Configuration import Configuration
    config = Configuration()
    for freq in [200]:
        config.INTERVAL_LENGTH = freq
        dst_dir = os.path.join('GAF4ZS')
        dst_dir = os.path.join(dst_dir, 'f' + str(config.INTERVAL_LENGTH))
        for device in config.DEVICE_LIST:
            dst_device_path = os.path.join(dst_dir, device)
            moveTrainTest(os.path.join(dst_dir))
            print(device + ' finished!')
        print('f ' + str(freq) + 'finished!')







