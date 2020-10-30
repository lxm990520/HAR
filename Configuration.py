import os
from util import *
import time
class Configuration:
    def prt_config(self):
        #print('GROUP_NUMBER : ' + str(self.GROUP_NUMBER))
        print('=============HyperParameter Configuration============')
        print('INTERVAL_LENGTH : ' + str(self.INTERVAL_LENGTH))
        print('WINDOW_LENGTH : ' + str(self.WINDOW_LENGTH))
        print('EDGE_LEN : ' + str(self.EDGE_LEN))
        print('IMG_SIZE : ' + str(self.IMG_SIZE))
        print('TIME_STEPS : ' + str(self.TIME_STEPS))
        print('=============Dataset Configuration============')
        print('DATASET : ' + str(self.DATASET))
        print('USER_LIST : ' + str(self.USER_LIST))
        print('SENSOR_LIST : ' + str(self.SENSOR_LIST))
        print('DEVICE_LIST : ' + str(self.DEVICE_LIST))
        print('GT_LIST : ' + str(self.GT_LIST))
    def save_config(self):
        txt_dir = os.path.join(self.SAVE_DIR,'Configuration_record.txt')
        with open(txt_dir, "a") as r:
            r.write('=============HyperParameter Configuration============')
            r.write('\nINTERVAL_LENGTH : ' + str(self.INTERVAL_LENGTH))
            r.write('\nWINDOW_LENGTH : ' + str(self.WINDOW_LENGTH))
            r.write('\nEDGE_LEN : ' + str(self.EDGE_LEN))
            r.write('\nIMG_SIZE : ' + str(self.IMG_SIZE))
            r.write('\nTIME_STEPS : ' + str(self.TIME_STEPS))
            r.write('\n=============Dataset Configuration============')
            r.write('\nDATASET : ' + str(self.DATASET))
            r.write('\nUSER_LIST : ' + str(self.USER_LIST))
            r.write('\nSENSOR_LIST : ' + str(self.SENSOR_LIST))
            r.write('\nDEVICE_LIST : ' + str(self.DEVICE_LIST))
            r.write('\nGT_LIST : ' + str(self.GT_LIST))
    def __init__(self):
        self.GROUP_NUMBER = 300    #200
        self.INTERVAL_LENGTH = 150   #200
        self.WINDOW_LENGTH = 100     #a mistake here. if IL=WL, no overlap
        self.EDGE_LEN = 100
        self.IMG_SIZE = 200
        self.TIME_STEPS = 10
        self.DATASET = 'HAR'
        self.USER_LIST = [str(x) for x in range(1,31)]
        self.EXP_LIST = [str(x) for x in range(1,61)]
        self.SENSOR_LIST = ['acc','gyro']
        self.DEVICE_LIST = ['SII']
        self.GT_LIST = ['Walking', 'Walking_upstairs', 'Walking_downstaris',
                        'Sitting', 'Standing', 'Laying', 'Stand_to_sit',
                        'Sit_to_stand', 'Sit_to_lie', 'Lie_to_sit',
                        'Stand_to_lie', 'Lie_to_stand']
        
        

    def fresh(self):
        self.INPUT_DIM = len(self.SENSOR_LIST)
        self.OUTPUT_DIM = len(self.GT_LIST)
        self.VERSION = time.strftime("%m-%d-%I-%M")
        dir_list = [self.DATASET, 'GAF4ZS', 'f' + str(self.INTERVAL_LENGTH), self.DEVICE_LIST[0]]
        self.DATASET_DIR = os.path.join(*dir_list) 
        dir_list = [self.DATASET, 'Result', 'f' + str(self.INTERVAL_LENGTH), self.DEVICE_LIST[0], self.VERSION]
        self.SAVE_DIR = os.path.join(*dir_list)
        if not os.path.exists(self.SAVE_DIR):
            mkdir(self.SAVE_DIR)
        self.save_config()
        