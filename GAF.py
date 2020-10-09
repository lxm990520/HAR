#生成GAF的代码
#输出一张GAF三维图片

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import imageio
import cv2
import os
GROUP_NUMBER = 300    #200
INTERVAL_LENGTH = 150   #200
WINDOW_LENGTH = 100     #a mistake here. if IL=WL, no overlap
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
#['Acc1', 'Electro1', 'Electro2', 'Acc2', 
               #'Gyro1', 'Magnet1', 'Acc3', 'Gyro2', 'Magnet2']
EDGE_LEN = 100
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
def GAF(threeD):

    for type in GT_TYPES:
        for sensor in SENSOR_LIST:
            print(sensor)
            print(type)
            data = pd.read_csv('MHEALTHDATASET/User_'+USER+'/User_'+USER+'_' + sensor + '_' + type + '.csv')  # source file
            I = np.ones([1, INTERVAL_LENGTH], float)
            if threeD:
                GASF_base = np.ones([INTERVAL_LENGTH, INTERVAL_LENGTH, 3], float)
            else:
                GASF_base = np.ones([2 * INTERVAL_LENGTH, 2 * INTERVAL_LENGTH], float)
            GADF_base = np.ones([2 * INTERVAL_LENGTH, 2 * INTERVAL_LENGTH], float)
            group_number = int((len(data.index) - INTERVAL_LENGTH)/WINDOW_LENGTH)
            for i in range(group_number):
                Header = HEADER_DICT[sensor]
                x = np.array(data[Header[0]].loc[i * WINDOW_LENGTH:(i * WINDOW_LENGTH) + INTERVAL_LENGTH - 1])
                y = np.array(data[Header[1]].loc[i * WINDOW_LENGTH:(i * WINDOW_LENGTH) + INTERVAL_LENGTH - 1])
                z = np.array(data[Header[2]].loc[i * WINDOW_LENGTH:(i * WINDOW_LENGTH) + INTERVAL_LENGTH - 1])
                ab = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))
                zmax = np.amax(z)
                zmin = np.amin(z)
                zrange = zmax - zmin
                znormal = np.divide(2 * z - zmax - zmin, zrange)
                znormal_inverse = np.sqrt(
                    I - np.power(znormal, 2))  # !!!ATTENTION HERE!!!   np.sqrt(row)=row;np.sqrt(column)=matrix
                GASF_z = znormal.T * znormal - znormal_inverse.T * znormal_inverse
                GADF_z = znormal_inverse.T * znormal - znormal.T * znormal_inverse

                ymax = np.amax(y)
                ymin = np.amin(y)
                yrange = ymax - ymin
                ynormal = np.divide(2 * y - ymax - ymin, yrange)
                ynormal_inverse = np.sqrt(
                    I - np.power(ynormal, 2))  # !!!ATTENTION HERE!!!   np.sqrt(row)=row;np.sqrt(column)=matrix
                GASF_y = ynormal.T * ynormal - ynormal_inverse.T * ynormal_inverse
                GADF_y = ynormal_inverse.T * ynormal - ynormal.T * ynormal_inverse

                xmax = np.amax(x)
                xmin = np.amin(x)
                xrange = xmax - xmin
                xnormal = np.divide(2 * x - xmax - xmin, xrange)
                xnormal_inverse = np.sqrt(
                    I - np.power(xnormal, 2))  # !!!ATTENTION HERE!!!   np.sqrt(row)=row;np.sqrt(column)=matrix
                GASF_x = xnormal.T * xnormal - xnormal_inverse.T * xnormal_inverse
                GADF_x = xnormal_inverse.T * xnormal - xnormal.T * xnormal_inverse

                abmax = np.amax(ab)
                abmin = np.amin(ab)
                abrange = abmax - abmin
                abnormal = np.divide(2 * ab - abmax - abmin, abrange)  # array devide (numerator，denominator)
                abnormal_inverse = np.sqrt(
                    I - np.power(abnormal, 2))  # !!!ATTENTION HERE!!!   np.sqrt(row)=row;np.sqrt(column)=matrix
                GASF_ab = abnormal.T * abnormal - abnormal_inverse.T * abnormal_inverse
                GADF_ab = abnormal_inverse.T * abnormal - abnormal.T * abnormal_inverse

                if threeD:
                    GASF_base[:,:,0] = GASF_x
                    GASF_base[:, :, 1] = GASF_y
                    GASF_base[:, :, 2] = GASF_z
                    GASF_base = (GASF_base+1)*128
                    mkdir('GAFjpg3d\\f150\\USER_' + USER + '\\' + type)
                    cv2.imwrite('GAFjpg3d\\f150\\USER_' + USER + '\\' + type + '\\' + USER + '_' + type + '_' + sensor + '_' + str(i) + '.jpg',
                                  GASF_base)
                else:
                    GASF_base[0:INTERVAL_LENGTH, 0:INTERVAL_LENGTH] = GASF_x
                    GASF_base[0:INTERVAL_LENGTH, INTERVAL_LENGTH:2 * INTERVAL_LENGTH] = GASF_y
                    GASF_base[INTERVAL_LENGTH:2 * INTERVAL_LENGTH, 0:INTERVAL_LENGTH] = GASF_z
                    GASF_base[INTERVAL_LENGTH:2 * INTERVAL_LENGTH, INTERVAL_LENGTH:2 * INTERVAL_LENGTH] = GASF_ab
                    mkdir('GAFjpg\\f150\\USER_' + USER + '\\' + type)
                    imageio.imsave('GAFjpg\\f150\\USER_' + USER + '\\' + type + '\\' + USER + '_' + type + '_' + sensor + '_' + str(i) + '.jpg',
                                  GASF_base)

                # GASF_resized = (GASF_base+1)*128
                # GASF_resized = GASF_resized.astype(np.int16)
                # GASF_resized = cv2.resize(GASF_resized,(EDGE_LEN,EDGE_LEN))
                # GASF = GASF_resized.reshape([1, EDGE_LEN * EDGE_LEN])
                # np.savetxt('GAF4DeepSense/acce_nexus41_GASF_100' + '/' + type + '/' + USER + type + '.' + str(i) + '.csv'
                #            ,GASF, delimiter=',')

                #file = open('GAFjpg\\acce_nexus41_'+USER+'_GASF\\readme.txt','w')
                #file.write('window length:'+str(INTERVAL_LENGTH))
                #file.close()

                # GADF_base[0:INTERVAL_LENGTH, 0:INTERVAL_LENGTH] = GADF_x
                # GADF_base[0:INTERVAL_LENGTH, INTERVAL_LENGTH:2 * INTERVAL_LENGTH] = GADF_y
                # GADF_base[INTERVAL_LENGTH:2 * INTERVAL_LENGTH, 0:INTERVAL_LENGTH] = GADF_z
                # GADF_base[INTERVAL_LENGTH:2 * INTERVAL_LENGTH, INTERVAL_LENGTH:2 * INTERVAL_LENGTH] = GADF_ab
                # imageio.imsave('GAFjpg\\acce_nexus41_a_GADF\\' + type + '\\' + USER +  type + '.' + str(i) + '.jpg', GADF_base)

            print(sensor + ' finished')
        print(type + ' finished')

for USER in USER_LIST:
    GAF(True)
    print('USER_' + USER + ' finished')
