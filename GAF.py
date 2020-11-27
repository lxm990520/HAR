#生成GAF的代码
#输出一张GAF三维图片

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import imageio
import cv2
from mkdir import mkdir
import os



def GAF(threeD,config):
    src_dir = os.path.join(config.DATASET, 'Phonedata')
    for device in config.DEVICE_LIST:
        device_dir = os.path.join(src_dir,device)
        for user in config.USER_LIST:
            user_dir = os.path.join(device_dir,'user_' + user)
            for file in os.listdir(user_dir):
                if not file.split('_')[-1].split('.')[0] in config.GT_LIST:#dont read file with all gestures
                    continue
                srcfile_path = os.path.join(user_dir, file)
                data = pd.read_csv(srcfile_path)
                data = data.sort_values("Index")#in case Index is not consecutive
                data.reset_index(drop=True, inplace=True)
                I = np.ones([1, config.INTERVAL_LENGTH], float)
                if threeD:
                    GAF_base = np.ones([config.INTERVAL_LENGTH, config.INTERVAL_LENGTH, 6], float)
                else:
                    GAF_base = np.ones([2 * config.INTERVAL_LENGTH, 2 * config.INTERVAL_LENGTH], float)
                for i in range(data.shape[0]//config.WINDOW_LENGTH - 1):
                    if not (data['Index'].loc[(i * config.WINDOW_LENGTH) + config.INTERVAL_LENGTH - 1] - data['Index'].loc[i * config.WINDOW_LENGTH]) == (config.INTERVAL_LENGTH - 1):
                        print("A inconsecutive point found!")
                        print(srcfile_path)
                        print((i * config.WINDOW_LENGTH) + config.INTERVAL_LENGTH - 1)
                        continue# Prevent from data that is not consecutive

                    x = np.array([data['x'].loc[i * config.WINDOW_LENGTH:(i * config.WINDOW_LENGTH) + config.INTERVAL_LENGTH - 1]])
                    y = np.array([data['y'].loc[i * config.WINDOW_LENGTH:(i * config.WINDOW_LENGTH) + config.INTERVAL_LENGTH - 1]])
                    z = np.array([data['z'].loc[i * config.WINDOW_LENGTH:(i * config.WINDOW_LENGTH) + config.INTERVAL_LENGTH - 1]])
                    ab = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))

                    zmax = np.amax(z)
                    zmin = np.amin(z)
                    zrange = zmax - zmin
                    znormal = np.divide(2 * z - zmax - zmin, zrange + 1e-6)
                    znormal_inverse = np.sqrt(np.abs(
                        I - np.power(znormal, 2)))  # !!!ATTENTION HERE!!!   np.sqrt(row)=row;np.sqrt(column)=matrix
                    GASF_z = znormal.T * znormal - znormal_inverse.T * znormal_inverse
                    GADF_z = znormal_inverse.T * znormal - znormal.T * znormal_inverse

                    ymax = np.amax(y)
                    ymin = np.amin(y)
                    yrange = ymax - ymin
                    ynormal = np.divide(2 * y - ymax - ymin, yrange + 1e-6)
                    ynormal_inverse = np.sqrt(np.abs(
                        I - np.power(ynormal, 2)))  # !!!ATTENTION HERE!!!   np.sqrt(row)=row;np.sqrt(column)=matrix
                    GASF_y = ynormal.T * ynormal - ynormal_inverse.T * ynormal_inverse
                    GADF_y = ynormal_inverse.T * ynormal - ynormal.T * ynormal_inverse

                    xmax = np.amax(x)
                    xmin = np.amin(x)
                    xrange = xmax - xmin
                    xnormal = np.divide(2 * x - xmax - xmin, xrange + 1e-6)
                    xnormal_inverse = np.sqrt(np.abs(
                        I - np.power(xnormal, 2)))  # !!!ATTENTION HERE!!!   np.sqrt(row)=row;np.sqrt(column)=matrix
                    GASF_x = xnormal.T * xnormal - xnormal_inverse.T * xnormal_inverse
                    GADF_x = xnormal_inverse.T * xnormal - xnormal.T * xnormal_inverse

                    abmax = np.amax(ab)
                    abmin = np.amin(ab)
                    abrange = abmax - abmin
                    abnormal = np.divide(2 * ab - abmax - abmin, abrange + 1e-6)  # array devide (numerator，denominator)
                    abnormal_inverse = np.sqrt(np.abs(
                        I - np.power(abnormal, 2)))  # !!!ATTENTION HERE!!!   np.sqrt(row)=row;np.sqrt(column)=matrix
                    GASF_ab = abnormal.T * abnormal - abnormal_inverse.T * abnormal_inverse
                    GADF_ab = abnormal_inverse.T * abnormal - abnormal.T * abnormal_inverse

                    if threeD:
                        GAF_base[:, :, 0] = GASF_x
                        GAF_base[:, :, 1] = GASF_y
                        GAF_base[:, :, 2] = GASF_z
                        GAF_base[:, :, 3] = GADF_x
                        GAF_base[:, :, 4] = GADF_y
                        GAF_base[:, :, 5] = GADF_z
                        #GASF_base = (GASF_base+1)*127.5
                        gt = file.split('.')[0].split('_')[-1]
                        sensor = file.split('_')[1]
                        device = ''.join([char for char in list(device) if not char == '_'])
                        savepath = config.DATASET + '\\GAFjpg3d_6c\\f' + str(config.INTERVAL_LENGTH) + '\\' + device + '\\' + 'train' + '\\' + gt
                        if not os.path.exists(savepath):
                            mkdir(savepath)
                        #GAF_resized = cv2.resize(GAF_base, (config.IMG_SIZE, config.IMG_SIZE))

                        GAF = GAF_base.flatten()
                        np.savetxt(savepath + '\\' + device.strip('_') + '_' + user + '_' + gt + '_' + sensor + '_' + str(i) + '.csv',
                                      GAF, delimiter = ',')
                    else:
                        GASF_base[0:INTERVAL_LENGTH, 0:INTERVAL_LENGTH] = GASF_x
                        GASF_base[0:INTERVAL_LENGTH, INTERVAL_LENGTH:2 * INTERVAL_LENGTH] = GASF_y
                        GASF_base[INTERVAL_LENGTH:2 * INTERVAL_LENGTH, 0:INTERVAL_LENGTH] = GASF_z
                        GASF_base[INTERVAL_LENGTH:2 * INTERVAL_LENGTH, INTERVAL_LENGTH:2 * INTERVAL_LENGTH] = GASF_ab
                        gt = file.split('.')[0].split('_')[-1]
                        sensor = file.split('_')[1]
                        device = ''.join([char for char in list(device) if not char == '_'])
                        savepath = config.DATASET + '\\GAFjpg\\f' + str(config.INTERVAL_LENGTH) + '\\' + device + '\\' + 'train' + '\\' + gt
                        if not os.path.exists(savepath):
                            mkdir(savepath)
                        GASF_resized = cv2.resize(GASF_base, (config.IMG_SIZE, config.IMG_SIZE))
                        cv2.imwrite(savepath + '\\' + device.strip('_') + '_' + user + '_' + gt + '_' + sensor + '_' + str(i) + '.jpg',
                                      GASF_resized)


            print(user + ' finished')
        print(device + ' finished')



if __name__ == '__main__':
    from Configuration import Configuration
    config = Configuration()
    for interval in [200]:
        config.INTERVAL_LENGTH = interval
        config.WINDOW_LENGTH = int(interval / 2)
        config.DATASET = 'HHAR'
        config.USER_LIST = ['a','b','c','d','e','f','g','h','i']
        config.GT_LIST = ['stand','sit','walk','stairsup','stairsdown','bike']
        config.SENSOR_LIST = ['acce','gyro']
        config.DEVICE_LIST = ['nexus41', 'nexus42', 's3mini_1', 's3mini_2']
        config.fresh()
        GAF(True,config)


