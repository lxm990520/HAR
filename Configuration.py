import os
from util import *
import time
class Configuration:
    def prt_config(self):
        print('GROUP_NUMBER : ' + str(self.GROUP_NUMBER))
        print('INTERVAL_LENGTH : ' + str(self.INTERVAL_LENGTH))
        print('WINDOW_LENGTH : ' + str(self.WINDOW_LENGTH))
        print('USER_LIST : ' + str(self.USER_LIST))
        print('SENSOR_LIST : ' + str(self.SENSOR_LIST))
        print('DEVICE_LIST : ' + str(self.DEVICE_LIST))
        print('EDGE_LEN : ' + str(self.EDGE_LEN))
    def __init__(self):
        self.GROUP_NUMBER = 300    #200
        self.INTERVAL_LENGTH = 150   #200
        self.WINDOW_LENGTH = 100     #a mistake here. if IL=WL, no overlap
        self.EDGE_LEN = 100
        self.IMG_SIZE = 200
        self.TIME_STEPS = 10
        self.DATASET = 'HHAR'
        self.USER_LIST = ['a','b','c','d','e','f','g','h','i']
        self.SENSOR_LIST = ['acce','gyro']
        self.DEVICE_LIST = ['nexus41','nexus42']
        self.GT_LIST = ['stand','sit','walk','stairsup','stairsdown','bike']
        
        

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
        