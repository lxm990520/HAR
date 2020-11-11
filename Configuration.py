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
        print('LEARNING_RATE : ' + str(self.LEARNING_RATE))
        print('DECAY : ' + str(self.DECAY))
        print('BATCH_SIZE : ' + str(self.BATCH_SIZE))
        print('=============Dataset Configuration============')
        print('DATASET : ' + str(self.DATASET))
        print('USER_LIST : ' + str(self.USER_LIST))
        print('SENSOR_LIST : ' + str(self.SENSOR_LIST))
        print('DEVICE_LIST : ' + str(self.DEVICE_LIST))
        print('GT_LIST : ' + str(self.GT_LIST))
        print('=============Weights Configuration============')
        print('LOAD_DIR : ' + str(self.LOAD_DIR))
    def save_config(self):
        txt_dir = os.path.join(self.SAVE_DIR,'Configuration_record.txt')
        with open(txt_dir, "a") as r:
            r.write('=============HyperParameter Configuration============')
            r.write('\nINTERVAL_LENGTH : ' + str(self.INTERVAL_LENGTH))
            r.write('\nWINDOW_LENGTH : ' + str(self.WINDOW_LENGTH))
            r.write('\nEDGE_LEN : ' + str(self.EDGE_LEN))
            r.write('\nIMG_SIZE : ' + str(self.IMG_SIZE))
            r.write('\nTIME_STEPS : ' + str(self.TIME_STEPS))
            r.write('\nLEARNING_RATE : ' + str(self.LEARNING_RATE))
            r.write('\nDECAY : ' + str(self.DECAY))
            r.write('\nBATCH_SIZE : ' + str(self.BATCH_SIZE))
            r.write('\n=============Dataset Configuration============')
            r.write('\nDATASET : ' + str(self.DATASET))
            r.write('\nUSER_LIST : ' + str(self.USER_LIST))
            r.write('\nSENSOR_LIST : ' + str(self.SENSOR_LIST))
            r.write('\nDEVICE_LIST : ' + str(self.DEVICE_LIST))
            r.write('\nGT_LIST : ' + str(self.GT_LIST))
            r.write('\n=============Weights Configuration============')
            r.write('\nLOAD_DIR : ' + str(self.LOAD_DIR))
    def __init__(self):
        self.GROUP_NUMBER = 300    #200
        self.INTERVAL_LENGTH = 200   #200
        self.WINDOW_LENGTH = 100     #a mistake here. if IL=WL, no overlap
        self.EDGE_LEN = 100
        self.IMG_SIZE = 200
        self.TIME_STEPS = 10
        self.LEARNING_RATE = 0.0001
        self.DECAY = 0
        self.BATCH_SIZE = 128
        self.LOAD_DIR = None
        self.DATASET = 'HHAR'
        self.USER_LIST = ['a','b','c','d','e','f','g','h','i']
        self.SENSOR_LIST = ['acce','gyro']
        self.DEVICE_LIST = ['nexus41','nexus42','s3_1','s3mini_1','s3mini_2']
        self.GT_LIST = ['stand','sit','walk','stairsup','stairsdown','bike']
        
        

    def fresh(self):
        self.INPUT_DIM = len(self.SENSOR_LIST)
        self.OUTPUT_DIM = len(self.GT_LIST)
        dir_list = [self.DATASET, 'GAF4ZS', 'f' + str(self.INTERVAL_LENGTH), self.DEVICE_LIST[0]]
        self.DATASET_DIR = os.path.join(*dir_list)
        self.VERSION = time.strftime("%m-%d-%H-%M")


    def save(self):
        dir_list = [self.DATASET, 'Result', 'f' + str(self.INTERVAL_LENGTH), self.DEVICE_LIST[0], self.VERSION]
        self.SAVE_DIR = os.path.join(*dir_list)
        if not os.path.exists(self.SAVE_DIR):
            mkdir(self.SAVE_DIR)
        self.save_config()
        