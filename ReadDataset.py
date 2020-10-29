import json
import os
import csv
from util import *
from Configuration import Configuration


def read_log(filename):
    user_list = []
    with open(filename,'r') as file_to_read:
        for line in file_to_read.readlines():
            listFromLine = line.split()
            user_list.append(listFromLine)
    return user_list




if __name__ == '__main__':
    config = Configuration()
    config.DATASET = 'MHEALTH'
    config.USER_LIST = ['1','2','3','4','5','6','7','8','9','10']
    #============================='Null' can not be removed!==================================
    config.GT_LIST = ['Null','Standing still', 'Sitting and relaxing', 'Lying down','Walking',
            'Climbing stairs', 'Waist bends forward', 'Frontal elevation of arms',
            'Knees bending (crouching)', 'Cycling', 'Jogging', 'Running', 
            'Jump front & back']
    #=========================================================================================
    #====================================Use Acc1 and Gyro1 Only==============================
    #config.SENSOR_LIST = ['Acc1', 'Gyro1']
    #=========================================================================================
    #====================================Without Electro1&2===================================
    config.SENSOR_LIST = ['Acc1', 'Acc2', 'Acc3', 'Gyro1', 'Gyro2', 'Magnet1', 'Magnet2']
    #=========================================================================================
    config.DEVICE_LIST = ['Unknown']
    config.HEADER_DICT = {'Acc1': ['Acc1_x', 'Acc1_y', 'Acc1_z'], 
               'Electro1':['Electro1'], 
               'Electro2':['Electro2'], 
               'Acc2': ['Acc2_x', 'Acc2_y', 'Acc2_z'], 
               'Gyro1': ['Gyro1_x', 'Gyro1_y', 'Gyro1_z'], 
               'Magnet1': ['Magnet1_x', 'Magnet1_y', 'Magnet1_z'], 
               'Acc3': ['Acc3_x', 'Acc3_y', 'Acc3_z'], 
               'Gyro2': ['Gyro2_x', 'Gyro2_y', 'Gyro2_z'], 
               'Magnet2': ['Magnet2_x', 'Magnet2_y', 'Magnet2_z'],
               'Label': 'Label'}
    config.HEADER_LIST = ['Acc1_x', 'Acc1_y', 'Acc1_z', 
               'Electro1', 'Electro2', 
               'Acc2_x', 'Acc2_y', 'Acc2_z', 
               'Gyro1_x', 'Gyro1_y', 'Gyro1_z', 
               'Magnet1_x', 'Magnet1_y', 'Magnet1_z', 
               'Acc3_x', 'Acc3_y', 'Acc3_z', 
               'Gyro2_x', 'Gyro2_y', 'Gyro2_z', 
               'Magnet2_x', 'Magnet2_y', 'Magnet2_z',
               'Label']
    dataset_dir = os.path.join(config.DATASET)
    dataset_dir = os.path.join(dataset_dir, 'Phonedata')
    for device in config.DEVICE_LIST:
        device_dir = os.path.join(dataset_dir, device)
        for user in config.USER_LIST:
            user_dir = os.path.join(device_dir, 'USER_' + user)
            src_dir = os.path.join('RawDataset', config.DATASET)
            df_src = pd.read_table(os.path.join(src_dir, 'mHealth_subject' + user + '.log'), names = config.HEADER_LIST)
            for gt in config.GT_LIST:
                subdf = df_src[df_src.Label == config.GT_LIST.index(gt)]
                for sensor in config.SENSOR_LIST:
                    name_list = ['Phone', sensor, device, user, gt]
                    file_name = '_'.join(name_list)
                    file_name = file_name + '.csv'
                    df_dst = subdf[config.HEADER_DICT[sensor]]
                    df_dst.columns = df_dst.columns.map(lambda x:x.split('_')[-1])
                    if not os.path.exists(user_dir):
                        mkdir(user_dir)
                    df_dst.to_csv(os.path.join(user_dir, file_name))


