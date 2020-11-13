import warnings
import tensorflow as tf
import os
import cv2
import time
import numpy as np
#==================Mask the Message==================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#====================================================
#==================Use CPU to train==================
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#====================================================
#==================Timeline Analysis=================
#from tensorflow.python.client import timeline
#from keras import callbacks
#====================================================
from tensorflow import keras
from tensorflow.keras.layers import AveragePooling3D,Reshape,Conv3D,Conv2D,AveragePooling2D,Dropout,\
    MaxPool3D,concatenate,LSTM,Bidirectional,Dense,Activation,RNN,GRU,Softmax,BatchNormalization
# import process4ZS as process
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from Configuration import Configuration
import util
import csv

SEPCTURAL_SAMPLES = 10  # d(k), dimension for each measurement(e.g. x,y,z...)
WIDE = 10  # 20       #amount of time intervals
DROPOUT_RATIO = 0.5
REGULARIZER_RATE = 0.0001
BUFFER_SIZE = 1000

TOTAL_ITER_NUM = 30000  # 0000


warnings.filterwarnings("ignore")
#==============================Configuration===========================================
config = Configuration()
config.INTERVAL_LENGTH = 200
config.WINDOW_LENGTH = 100
config.LEARNING_RATE = 0.1
config.BATCH_SIZE = 64
config.DECAY = 0
config.DATASET = 'HHAR'
config.USER_LIST = ['a','b','c','d','e','f','g','h','i']
config.GT_LIST = ['stand','sit','walk','stairsup','stairsdown','bike']
config.SENSOR_LIST = ['acce','gyro']
config.DEVICE_LIST = ['nexus41']
#config.LOAD_DIR = 'HHAR\\Result\\f200\\nexus41\\11-12-20-13'
config.LOAD_DIR = None
config.fresh()
config.save()




class Tfdata():
    def __init__(self, dir, config):
        self.data_dir = dir
        self.config = config

    def get_file_hhar(self, file_dir):
        """Output filename_list&label_list containing all files under this directory
        """

        # read all images and create relative index and label
        images = []
        temp = []
        # here I need to arrange the folders and files by my self
        print(file_dir)
        for root, sub_folders, files in os.walk(file_dir):
            for name in files:
                images.append(os.path.join(root, name))
            for name in sub_folders:
                temp.append(os.path.join(root, name))

        labels = []
        for one_folder in temp:  # temp is a list of subfolders
            # learn this op by debug
            n_img = len(os.listdir(one_folder))  # os.listdir:list all files&dirs in this dir
            letter = one_folder.split('\\')[-1]
            flag = False
            for gt in self.config.GT_LIST:
                if gt in letter:
                    labels = np.append(labels, n_img * [self.config.GT_LIST.index(gt)])
                    flag = True
                    break
            if not flag:
                raise TypeError("unknown label, check data!!!")


            # ********************ATTENTION, CUSTIMIZE************************************************
            #===================================HHAR==================================================
            # if 'bike' in letter:
            #     labels = np.append(labels, n_img * [
            #         0])  # establish a huge huge lebel array for all elements in the same classification
            # elif 'sit' in letter:
            #     labels = np.append(labels, n_img * [1])
            # elif 'stairsup' in letter:
            #     labels = np.append(labels, n_img * [2])
            # elif 'stairsdown' in letter:
            #     labels = np.append(labels, n_img * [3])
            # elif 'stand' in letter:
            #     labels = np.append(labels, n_img * [4])
            # elif 'walk' in letter:
            #     labels = np.append(labels, n_img * [5])
            # else:
            #     raise TypeError("unknown label, check data!!!")
            #=================================MHEALTH===================================================
            # if 'Standing still' in letter:
            #     labels = np.append(labels, n_img * [
            #         0])  # establish a huge huge lebel array for all elements in the same classification
            # elif 'Sitting and relaxing' in letter:
            #     labels = np.append(labels, n_img * [1])
            # elif 'Lying down' in letter:
            #     labels = np.append(labels, n_img * [2])
            # elif 'Walking' in letter:
            #     labels = np.append(labels, n_img * [3])
            # elif 'Climbing stairs' in letter:
            #     labels = np.append(labels, n_img * [4])
            # elif 'Waist bends forward' in letter:
            #     labels = np.append(labels, n_img * [5])
            # elif 'Frontal elevation of arms' in letter:
            #     labels = np.append(labels, n_img * [6])
            # elif 'Knees bending (crouching)' in letter:
            #     labels = np.append(labels, n_img * [7])
            # elif 'Cycling' in letter:
            #     labels = np.append(labels, n_img * [8])
            # elif 'Jogging' in letter:
            #     labels = np.append(labels, n_img * [9])
            # elif 'Running' in letter:
            #     labels = np.append(labels, n_img * [10])
            # elif 'Jump front & back' in letter:
            #     labels = np.append(labels, n_img * [11])
            # else:
            #     raise TypeError("unknown label, check data!!!")
            # ******************************************************************************************

        temp = np.array([images, labels])
        temp = temp.transpose()  # transpose to a n_img*2 array, so that each match has 2 elements: image&label
        np.random.shuffle(temp)

        # image_list = list(temp[1])
        # label_list = list(temp[0])
        image_list = list(temp[:, 0])
        label_list = list(temp[:, 1])
        label_list = [int(float(i)) for i in label_list]

        return image_list, label_list

    def read_image(self, filename, label):
        """func used in dataset.map
        convert file's dir 2 tensor
        """
        image = tf.io.read_file(filename)

        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [self.config.IMG_SIZE * self.config.INPUT_DIM, self.config.IMG_SIZE * 10])#10 is a changable parameter
        image = image * 2.0 - 1.0#map image to [-1,1]
        #image = tf.image.per_image_standardization(image)#image standardization
        # label = tf.one_hot(label, OUT_DIM)
        return image, label

    def acquire_data(self, is_shuffle=True):
        "Get usable tf.data data with a input dir"
        self.raw_images, self.raw_labels = self.get_file_hhar(self.data_dir)
        #print(self.raw_images)
        #print(self.raw_labels)
        print("data amount:{}".format(len(self.raw_labels)))
        img_data = tf.data.Dataset.from_tensor_slices((self.raw_images, self.raw_labels))
        img_data = img_data.map(self.read_image)
        if is_shuffle:
           img_data = img_data.shuffle(BUFFER_SIZE)
        img_data = img_data.batch(self.config.BATCH_SIZE, drop_remainder=True)

        return img_data


class LossHistory(keras.callbacks.Callback):
    def __init__(self,config):
        self.save_dir = config.SAVE_DIR
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        save_dir = self.save_dir
#========================================Record Accuracy and Loss==============================
        record_dir = os.path.join(self.save_dir,'Temp_record.csv')

        record = [str(len(self.losses['epoch'])), logs.get('loss'), logs.get('acc'), logs.get('val_loss'), logs.get('val_acc')]
        if os.path.isfile(record_dir):
            with open(record_dir, 'a', newline='')as f:
                writer = csv.writer(f, dialect='excel')
                writer.writerow(record)
        else:
            with open(record_dir, 'w', newline='')as f:
                writer = csv.writer(f, dialect='excel')
                writer.writerow(['Epoch', 'Train_loss', 'Train_acc', 'Val_loss', 'Val_acc'])
                writer.writerow(record)
#==============================================================================================
#========================================Plot Accuracy Figure==================================
        acc_dir = os.path.join(self.save_dir,'Accuracy.jpg')
        loss_type = 'epoch'
        iters = range(len(self.losses[loss_type]))
        plt.figure(0)
        plt.plot(iters, self.accuracy[loss_type], 'r')
        plt.plot(iters, self.val_acc[loss_type], 'b')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('accuracy')
        plt.savefig(acc_dir)
        plt.close(0)
#==============================================================================================
#=======================================Plot Loss Figure=======================================
        loss_dir = os.path.join(self.save_dir,'Loss.jpg')
        plt.figure(1)
        plt.plot(iters, self.losses[loss_type], 'r')
        plt.plot(iters, self.val_loss[loss_type], 'b')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.savefig(loss_dir)
        plt.close(1)
#==============================================================================================
# ==============================Model Save Weight========================================
        weight_name = 'zs_halfoverlap.h5'
        weight_dir = os.path.join(self.save_dir, weight_name)
        self.model.save_weights(weight_dir)

# =======================================================================================

    def on_train_end(self, batch, logs={}):
        loss_type = 'epoch'
        iters = range(len(self.losses[loss_type]))
#=======================================Record Accuracy========================================
        txt_dir = os.path.join(self.save_dir,'Accuracy_record.txt')
        with open(txt_dir, "a") as r:
            r.write("*********************************************************************************\n")
            r.write("here is the accuracy of validation:\n")
            r.write("this is no merge version\n")
            r.write(time.strftime("%Y-%m-%d %I:%M:%S %p\n"))
            for i in range(len(self.val_acc[loss_type])):
                r.write("{}\n".format(str(self.val_acc[loss_type][i])))
            r.write("*********************************************************************************\n")
            r.write("here is the loss of validation :\n")
            r.write("this is no merge version\n")
            r.write(time.strftime("%Y-%m-%d %I:%M:%S %p\n"))
            for i in range(len(self.val_loss[loss_type])):
                r.write("{}\n".format(str(self.val_loss[loss_type][i])))
            r.write("*********************************************************************************\n")
            r.write("here is the accuracy of training:\n")
            r.write("this is no merge version\n")
            r.write(time.strftime("%Y-%m-%d %I:%M:%S %p\n"))
            for i in range(len(self.accuracy[loss_type])):
                r.write("{}\n".format(str(self.accuracy[loss_type][i])))
            r.write("*********************************************************************************\n")
            r.write("here is the loss of training:\n")
            r.write("this is no merge version\n")
            r.write(time.strftime("%Y-%m-%d %I:%M:%S %p\n"))
            for i in range(len(self.losses[loss_type])):
                r.write("{}\n".format(str(self.losses[loss_type][i])))

#==============================================================================================
#========================================Plot Loss Figure======================================
        loss_dir = os.path.join(self.save_dir, 'Loss.jpg')
        plt.figure(1)
        plt.plot(iters, self.losses[loss_type], 'r', label="train")
        plt.plot(iters, self.val_loss[loss_type], 'b', label="test")
        plt.grid(True)
        plt.legend()
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.savefig(loss_dir)
#==============================================================================================
#=======================================Plot Accuracy Figure===================================
        acc_dir = os.path.join(self.save_dir, 'Accuracy.jpg')
        plt.figure(0)
        plt.plot(iters, self.accuracy[loss_type], 'r', label="train")
        plt.plot(iters, self.val_acc[loss_type], 'b', label="test")
        plt.grid(True)
        plt.legend()
        plt.xlabel(loss_type)
        plt.ylabel('accuracy')
        plt.savefig(acc_dir)
#=============================================================================================

class ZeepSenseEasy():
    def __init__(self,config):
        self.config = config
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(gpus)
        self.single_img_length = 200

        self.single_input = keras.Input(shape=(self.config.INPUT_DIM * self.single_img_length, 10 * self.single_img_length, 3), \
                                        name="Input")
        self.sensor_input = [x for x in range(self.config.INPUT_DIM)]
        self.conv1 = [x for x in range(self.config.INPUT_DIM)]
        self.conv2 = [x for x in range(self.config.INPUT_DIM)]
        self.conv3 = [x for x in range(self.config.INPUT_DIM)]
        self.sensor_output = [x for x in range(self.config.INPUT_DIM)]
        for sensor in range(self.config.INPUT_DIM):
            self.sensor_input[sensor] = self.single_input[:, sensor * self.single_img_length : (sensor + 1) * self.single_img_length, :, :]
            self.sensor_input[sensor] = Reshape((10, self.single_img_length, self.single_img_length, 3), name = "Input_" + self.config.SENSOR_LIST[sensor])(self.sensor_input[sensor])
#=======================================================Conv1==============================================
            self.conv1[sensor] = Conv3D(64, (1, 5, 5), (1, 3, 3),
                                        name="conv_" + self.config.SENSOR_LIST[sensor] + "_1",
                                        kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE))(self.sensor_input[sensor])
            self.conv1[sensor] = BatchNormalization(name = "conv_" + self.config.SENSOR_LIST[sensor] + "_1_bn")(self.conv1[sensor])
            self.conv1[sensor] = Activation("relu", name="conv_" + self.config.SENSOR_LIST[sensor] + "_1_relu")(self.conv1[sensor])
            self.conv1[sensor] = Dropout(DROPOUT_RATIO, noise_shape=[self.config.BATCH_SIZE, 1, 1, 1, self.conv1[sensor].shape[-1]], name="conv_" + self.config.SENSOR_LIST[sensor] + "_1_dropout")(self.conv1[sensor])
            self.conv1[sensor] = AveragePooling3D((1, 2, 2),name="conv_" + self.config.SENSOR_LIST[sensor] + "_1_pool")(self.conv1[sensor])
#=======================================================Conv2==============================================
            self.conv2[sensor] = Conv3D(64, (1, 3, 3), (1, 1, 1),
                                        name="conv_" + self.config.SENSOR_LIST[sensor] + "_2",
                                        kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE))(self.conv1[sensor])
            self.conv2[sensor] = BatchNormalization(name = "conv_" + self.config.SENSOR_LIST[sensor] + "_2_bn")(self.conv2[sensor])
            self.conv2[sensor] = Activation("relu", name="conv_" + self.config.SENSOR_LIST[sensor] + "_2_relu")(self.conv2[sensor])
            self.conv2[sensor] = Dropout(DROPOUT_RATIO, noise_shape=[self.config.BATCH_SIZE, 1, 1, 1, self.conv2[sensor].shape[-1]], name="conv_" + self.config.SENSOR_LIST[sensor] + "_2_dropout")(self.conv2[sensor])
            self.conv2[sensor] = AveragePooling3D((1, 2, 2), name="conv_" + self.config.SENSOR_LIST[sensor] + "_2_pool")(self.conv2[sensor])
#=======================================================Conv3==============================================
            self.conv3[sensor] = Conv3D(64, (1, 3, 3), (1, 1, 1),
                                        name="conv_" + self.config.SENSOR_LIST[sensor] + "_3",
                                        kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE))(self.conv2[sensor])
            self.conv3[sensor] = BatchNormalization(name="conv_" + self.config.SENSOR_LIST[sensor] + "_3_bn")(self.conv3[sensor])
            self.conv3[sensor] = Activation("relu", name="conv_" + self.config.SENSOR_LIST[sensor] + "_3_relu")(self.conv3[sensor])
            self.conv3[sensor] = Dropout(DROPOUT_RATIO,
                                         noise_shape=[self.config.BATCH_SIZE, 1, 1, 1, self.conv3[sensor].shape[-1]],
                                         name="conv_" + self.config.SENSOR_LIST[sensor] + "_3_dropout")(self.conv3[sensor])
            self.conv3[sensor] = AveragePooling3D((1, 2, 2),
                                                  name="conv_" + self.config.SENSOR_LIST[sensor] + "_3_pool")(self.conv3[sensor])
#======================================================Output==============================================
            self.sensor_output[sensor] = Reshape((10, 1, self.conv3[sensor].shape[-2] * self.conv3[sensor].shape[-3], self.conv3[sensor].shape[-1]), name="output_" + self.config.SENSOR_LIST[sensor])(self.conv3[sensor])#attention here, maybe errorous
#==========================================================================================================
#===========================================Merge Input=============================
#============================no attention===========================================
        self.merge_noattention = concatenate(self.sensor_output,axis = -3, name = "input_merge")
        self.merge_input = self.merge_noattention

#==============================attention====================================================
        #self.merge_input = concatenate([self.acce_output, self.gyro_output], axis=-3, name="Input_merge")
        #self.merge_attention_input = Reshape((10,2,self.merge_input.shape[-2]*self.merge_input.shape[-1]))(self.merge_input)
        #self.merge_attention = tf.matmul(self.merge_attention_input,self.merge_attention_input,transpose_b=True)
        #self.merge_attention = tf.matmul(tf.ones((1,2),dtype=tf.float32),self.merge_attention)
        #softmax_layer = Softmax(axis = -1)
        #self.merge_attention = softmax_layer(self.merge_attention)
        #merge_attention_dim = self.merge_attention.get_shape()
        #print("merge_attention_dim")
        #print(merge_attention_dim)
        #self.merge_attention_output = tf.matmul(self.merge_attention,self.merge_attention_input)
        #self.merge_attention_output = Reshape((10,16,16,64))(self.merge_attention_output)
        #merge_attention_output_dim = self.merge_attention_output.get_shape()
        #print("merge_attention_output_dim")
        #print(merge_attention_output_dim)
        #self.merge_input = self.merge_attention_output
#=========================================Merge Conv1==================================
        self.merge_conv1 = Conv3D(64, kernel_size=(1,2,8),
                                   name='conv_merge_1',
                                   strides=( 1, 1,1),
                                   padding='SAME',
                                   kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE)
                                   )(self.merge_input)
        self.merge_conv1 = BatchNormalization(name="conv_merge_1_bn")(self.merge_conv1)
        self.merge_conv1 = Activation("relu", name="conv_merge_1_relu")(self.merge_conv1)
        self.merge_conv1 = Dropout(DROPOUT_RATIO, noise_shape=[self.config.BATCH_SIZE, 1, 1, 1, self.merge_conv1.shape[-1]],
                             name="conv_merge_1_dropout")(self.merge_conv1)
        self.merge_conv1 = AveragePooling3D((1, 1,2))(self.merge_conv1)
#========================================Merge Conv2==================================
        self.merge_conv2 = Conv3D(64, kernel_size=(1,2,6),
                                   name='conv_merge_2',
                                   strides=( 1, 1,1),
                                   padding='SAME',
                                   kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE)
                                   )(self.merge_conv1)
        self.merge_conv2 = BatchNormalization(name="conv_merge_2_bn")(self.merge_conv2)
        self.merge_conv2 = Activation("relu", name="conv_merge_2_relu")(self.merge_conv2)
        self.merge_conv2 = Dropout(DROPOUT_RATIO, noise_shape=[self.config.BATCH_SIZE, 1, 1, 1, self.merge_conv2.shape[-1]],
                             name="conv_merge_2_dropout")(self.merge_conv2)
        self.merge_conv2 = AveragePooling3D((1, 1,2))(self.merge_conv2)
#=====================================================================================
#========================================Merge Conv3==================================
        self.merge_conv3 = Conv3D(64, kernel_size=(1,2,4),
                                   name='conv_merge_3',
                                   strides=( 1, 1,1),
                                   padding='SAME',
                                   kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE)
                                   )(self.merge_conv2)
        self.merge_conv3 = BatchNormalization(name="conv_merge_3_bn")(self.merge_conv3)
        self.merge_conv3 = Activation("relu", name="conv_merge_3_relu")(self.merge_conv3)
        self.merge_conv3 = Dropout(DROPOUT_RATIO, noise_shape=[self.config.BATCH_SIZE, 1, 1, 1, self.merge_conv3.shape[-1]],
                             name="conv_merge_3_dropout")(self.merge_conv3)
        self.merge_conv3 = AveragePooling3D((1, 1,2))(self.merge_conv3)
#=====================================================================================
        # self.conv2 = Conv3D(64, kernel_size=(1, 1, 5),
        #                            name="conv_merge_2",
        #                            strides=(1, 1, 3),
        #                            #  padding='SAME',
        #                            kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE)
        #                            )(self.conv1)
        # # self.conv2 = BatchNormalization(name="conv_merge_2_batchnorm")(self.conv2)
        # self.conv2 = Activation("relu", name="conv_merge_2_relu")(self.conv2)
        # self.conv2 = Dropout(DROPOUT_RATIO, noise_shape=[BATCH_SIZE, 1, 1, 1, 64], name="conv_merge_2_dropout")(
        #     self.conv2)

        self.conv_output = self.merge_conv3
        self.rnn_input = Reshape((10, self.conv_output.shape[-1] * self.conv_output.shape[-2] * self.conv_output.shape[-3]), name="Output_merge")(self.conv_output)

        self.rnn = LSTM(120, return_sequences=True, name="RNN_1_Modified", kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE))(self.rnn_input)


        self.rnn = LSTM(120, return_sequences=True, name="RNN_2", kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE))(self.rnn)
        self.sum_rnn_out = tf.reduce_sum(self.rnn, axis=1, keep_dims=False)
        self.avg_rnn_out = self.sum_rnn_out / tf.cast(self.rnn.shape[-2], tf.float32) # to be modified
        self.rnn_output = Dense(self.config.OUTPUT_DIM, 'softmax',kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE), name="Softmax")(self.avg_rnn_out)

        self.model = keras.Model(
            inputs=self.single_input,
            outputs=self.rnn_output)

    def train(self, file_dir, val_dir, epochs, save_dir=None, load_dir = None):
        # =======================================Timeline Analysis=======================================
        #timeline_callback = callbacks.TensorBoard(log_dir = os.path.join(config.SAVE_DIR,'logs'), write_graph=True, write_images=False)
        #run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()
        #self.model.compile(optimizer=keras.optimizers.Adam(lr = self.config.LEARNING_RATE, decay = self.config.DECAY),
                           #loss=keras.losses.SparseCategoricalCrossentropy(),
                           #metrics=['acc'], options=run_options, run_metadata=run_metadata)
        #================================================================================================

        self.model.compile(optimizer=keras.optimizers.Adam(lr = self.config.LEARNING_RATE, decay = self.config.DECAY),
                           loss=keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['acc'])

        data_init = Tfdata(file_dir,self.config)
        data = data_init.acquire_data(True)

        val_data_init = Tfdata(val_dir,self.config)
        val_data = val_data_init.acquire_data(False)

        print("Training results:")

        self.history = LossHistory(self.config)

#=======================================Model Load Weight=======================================
        if not load_dir == None:
            load_weight_dir = os.path.join(load_dir,"zs_halfoverlap.h5")
            self.model.load_weights(load_weight_dir, by_name = True)
#===============================================================================================
        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                                      patience = 10, mode = 'auto',
                                      epsilon = 0.0001, verbose = 1)
        self.model.fit(data,
                       epochs=epochs,
                       verbose=2,
                       validation_data=val_data,
                       callbacks=[self.history, reduce_lr])
#==============================Timeline Analysis========================================
        #self.model.fit(data,
                       #epochs=epochs,
                       #verbose=2,
                       #validation_data=val_data,
                       #callbacks=[self.history, reduce_lr, timeline_callback])
        #tl = timeline.Timeline(run_metadata.step_stats)
        #ctf = tl.generate_chrome_trace_format()
        #with open(os.path.join(config.SAVE_DIR,'timeline.json'), 'w') as f:
            #f.write(ctf)
        #print('timeline.json has been saved!')
#==============================Model Save Weight========================================
        weight_name = 'zs_halfoverlap.h5'
        weight_dir = os.path.join(config.SAVE_DIR, weight_name)
        self.model.save_weights(weight_dir)
#=======================================================================================

    def evaluate(self, val_dir, save_dir):
        val_data_init = Tfdata(val_dir,self.config)
        val_data = val_data_init.acquire_data(False)
        print("y_true in evaluation")
        print(val_data_init.raw_labels)



        Y_pred = self.model.predict(val_data)
        y_pred = np.argmax(Y_pred, axis=1)

        y_true = val_data_init.raw_labels[0:len(y_pred) // self.config.BATCH_SIZE * self.config.BATCH_SIZE]
        print("y_pred in confusion matrix.")
        print(y_pred.tolist())
        output_matrix = confusion_matrix(y_true, y_pred) #/ (len(y_true))
        cm_dir = os.path.join(save_dir,'CM.jpg')
        util.plot_confusion_matrix(output_matrix, self.config.GT_LIST,"Normalized Confusion Matrix",cm_dir)

        # plt.figure(2)
        # tick_marks = np.arange(len(config.GT_LIST))
        # plt.xticks(tick_marks, config.GT_LIST, rotation=90)
        # plt.yticks(tick_marks, config.GT_LIST)
        # plt.imshow(output_matrix, cmap=plt.cm.Reds)
        # plt.xlabel("predicted labels", fontsize='large')
        # plt.ylabel("true labels", fontsize="large")
        # plt.title("Normalized Confusion Matrix")
        # plt.colorbar(orientation='vertical')
        # plt.savefig(cm_dir, bbox_inches='tight')
        # plt.tight_layout()
        # plt.close(2)

        f1 = metrics.f1_score(y_true, y_pred, average='micro')
        f2 = metrics.f1_score(y_true, y_pred, average='macro')
        print('micro f1 score: {}, macro f1 score:{}'.format(f1,f2))

#================================Run the Model=========================================
example = ZeepSenseEasy(config)
example.model.summary()
train_dir = os.path.join(config.DATASET_DIR, 'train_halfoverlap')
test_dir = os.path.join(config.DATASET_DIR, 'test_halfoverlap')
val_dir = os.path.join(config.DATASET_DIR, 'test_halfoverlap')#to swap test to val


example.train(train_dir,
              test_dir,
              epochs=300, save_dir= config.SAVE_DIR, load_dir = config.LOAD_DIR)
example.evaluate(val_dir,config.SAVE_DIR)



