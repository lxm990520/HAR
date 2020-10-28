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
            user_dir = os.path.join(device_dir,'USER_' + user)
            for gt in config.GT_LIST:
                for sensor in config.SENSOR_LIST:
                    filename = 'Phone_' + sensor + '_' + device + '_' + user+'_' + gt + '.csv'
                    srcfile_path = os.path.join(user_dir,filename)
                    data = pd.read_csv(srcfile_path)  # source file
                    I = np.ones([1, config.INTERVAL_LENGTH], float)
                    if threeD:
                        GASF_base = np.ones([config.INTERVAL_LENGTH, config.INTERVAL_LENGTH, 3], float)
                    else:
                        GASF_base = np.ones([2 * config.INTERVAL_LENGTH, 2 * config.INTERVAL_LENGTH], float)
                    GADF_base = np.ones([2 * config.INTERVAL_LENGTH, 2 * config.INTERVAL_LENGTH], float)
                    #print(data.shape[0]//config.WINDOW_LENGTH)
                    for i in range(data.shape[0]//config.WINDOW_LENGTH - 1):
                        #print(i)
                        x = np.array(data['x'].loc[i * config.WINDOW_LENGTH:(i * config.WINDOW_LENGTH) + config.INTERVAL_LENGTH - 1])
                        y = np.array(data['y'].loc[i * config.WINDOW_LENGTH:(i * config.WINDOW_LENGTH) + config.INTERVAL_LENGTH - 1])
                        z = np.array(data['z'].loc[i * config.WINDOW_LENGTH:(i * config.WINDOW_LENGTH) + config.INTERVAL_LENGTH - 1])
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
                            savepath = config.DATASET + '\\GAFjpg3d\\f' + str(config.INTERVAL_LENGTH) + '\\' + device + '\\' + 'train' + '\\' + gt
                            if not os.path.exists(savepath):
                                mkdir(savepath)
                            GASF_resized = cv2.resize(GASF_base, (config.IMG_SIZE, config.IMG_SIZE))
                            cv2.imwrite(savepath + '\\' + device + '_' + user + '_' + gt + '_' + sensor + '_' + str(i) + '.jpg',
                                          GASF_resized)
                        else:
                            GASF_base[0:INTERVAL_LENGTH, 0:INTERVAL_LENGTH] = GASF_x
                            GASF_base[0:INTERVAL_LENGTH, INTERVAL_LENGTH:2 * INTERVAL_LENGTH] = GASF_y
                            GASF_base[INTERVAL_LENGTH:2 * INTERVAL_LENGTH, 0:INTERVAL_LENGTH] = GASF_z
                            GASF_base[INTERVAL_LENGTH:2 * INTERVAL_LENGTH, INTERVAL_LENGTH:2 * INTERVAL_LENGTH] = GASF_ab
                            savepath = config.DATASET + '\\GAFjpg\\f' + str(config.INTERVAL_LENGTH) + '\\' + device + '\\' + 'train' + '\\' + gt
                            if not os.path.exists(savepath):
                                mkdir(savepath)
                            GASF_resized = cv2.resize(GASF_base, (config.IMG_SIZE, config.IMG_SIZE))
                            cv2.imwrite(savepath + '\\' + device + '_' + user + '_' + gt + '_' + sensor + '_' + str(i) + '.jpg',
                                          GASF_resized)

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
            print(user + ' finished')
        print(device + ' finished')



if __name__ == '__main__':
    from Configuration import Configuration
    config = Configuration()
    for int in [200]:
        config.INTERVAL_LENGTH = int
        config.DATASET = 'MHEALTH'
        config.USER_LIST = ['1','2','3','4','5','6','7','8','9','10']
        config.GT_LIST = ['Standing still', 'Sitting and relaxing', 'Lying down','Walking',
            'Climbing stairs', 'Waist bends forward', 'Frontal elevation of arms',
            'Knees bending (crouching)', 'Cycling', 'Jogging', 'Running', 
            'Jump front & back']
        config.SENSOR_LIST = ['Acc1', 'Gyro1']
        config.DEVICE_LIST = ['Unknown']
        GAF(True,config)


