import json
import os
import csv
from util import *
import re
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
    config.DATASET = 'HAR'
    config.USER_LIST = [str(x) for x in range(1,31)]
    #============================='Null' can not be removed!==================================
    config.GT_LIST = ['Null','Walking', 'Walking_upstairs', 'Walking_downstaris',
                        'Sitting', 'Standing', 'Laying', 'Stand_to_sit',
                        'Sit_to_stand', 'Sit_to_lie', 'Lie_to_sit',
                        'Stand_to_lie', 'Lie_to_stand']
    #=========================================================================================
    #====================================Use Acc1 and Gyro1 Only==============================
    #config.SENSOR_LIST = ['Acc1', 'Gyro1']
    #=========================================================================================
    #====================================Without Electro1&2===================================
    config.SENSOR_LIST = ['acc','gyro']
    #=========================================================================================
    config.DEVICE_LIST = ['SII']
    config.HEADER_DICT = {'Acc': ['Acc1_x', 'Acc1_y', 'Acc1_z'], 
               'Gyro': ['Gyro1_x', 'Gyro1_y', 'Gyro1_z']}
    config.HEADER_LIST = ['x', 'y', 'z']

    Label_header = ['ExpId', 'UserId', 'GtId', 'Start', 'End']
    dataset_dir = os.path.join(config.DATASET)
    dataset_dir = os.path.join(dataset_dir, 'Phonedata')
    rawdata_dir = os.path.join('RawDataset', config.DATASET)
    record = pd.read_csv(os.path.join(rawdata_dir, 'labels.txt'), names = Label_header, sep = ' ')
    for file in os.listdir(rawdata_dir):
        if file == 'labels.txt':
            continue
        exp = file.split('_')[1]
        expId = int(re.findall(r"\d+", exp)[0])
        #print(expId)
        user = file.split('_')[-1].split('.')[0]
        userId = int(re.findall(r"\d+", user)[0])
        #print(userId)
        sensor = file.split('_')[0]

        device = config.DEVICE_LIST[0]

        df = pd.read_csv(os.path.join(rawdata_dir, file), names = config.HEADER_LIST, sep = ' ')
        record_user = record[record.UserId == userId]
        record_exp = record[record.ExpId == expId]
        for row in record_exp.itertuples():
            gt = config.GT_LIST[row.GtId]
            dst_name_list = ['Phone', sensor, device, str(expId), str(userId), gt]
            dst_name = '_'.join(dst_name_list) + '.csv'

            subdf = df.iloc[row.Start + 1: row.End + 1]#index could be wrong
            dst_dir_list = ['HAR', 'Phonedata', 'SII', user]
            dst_dir = os.path.join(*dst_dir_list)
            if not os.path.exists(dst_dir):
                mkdir(dst_dir)
            subdf.to_csv(os.path.join(dst_dir,dst_name),mode = 'a', header = False)

