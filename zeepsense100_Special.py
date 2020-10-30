import warnings
import tensorflow as tf
import os
import cv2
import time
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import AveragePooling3D,Reshape,Conv3D,Conv2D,AveragePooling2D,Dropout,\
    MaxPool3D,concatenate,LSTM,Bidirectional,Dense,Activation,RNN,GRU,Softmax
# import process4ZS as process
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from Configuration import Configuration
import util

SEPCTURAL_SAMPLES = 10  # d(k), dimension for each measurement(e.g. x,y,z...)
WIDE = 10  # 20       #amount of time intervals
DROPOUT_RATIO = 0.3
REGULARIZER_RATE = 0.005
BUFFER_SIZE = 2000

BATCH_SIZE = 16  # 64
TOTAL_ITER_NUM = 30000  # 0000


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
            #print(len(files))
            #print(sub_folders)

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
        image = tf.image.resize(image, [200 * self.config.INPUT_DIM, 200 * 10])#10 is a changable parameter

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
        # if is_shuffle:
        #   img_data = img_data.shuffle(BUFFER_SIZE)
        img_data = img_data.batch(BATCH_SIZE, drop_remainder=True)

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
    def on_train_end(self, batch, logs={}):
        loss_type = 'epoch'
        iters = range(len(self.losses[loss_type]))
#=======================================Record Accuracy========================================
        txt_dir = os.path.join(self.save_dir,'Accuracy_record.txt')
        with open(txt_dir, "a") as r:
            r.write("\n*********************************************************************************\n\n\n")
            r.write("\nhere is the accuracy of validation:\n")
            r.write("\nthis is no merge version\n")
            r.write(time.strftime("%Y-%m-%d %I:%M:%S %p\n"))
            for i in range(len(self.val_acc[loss_type])):
                r.write("{}\n".format(str(self.val_acc[loss_type][i])))
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
        #gpus = None
        # if gpus:
        #     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        #     try:
        #         tf.config.experimental.set_virtual_device_configuration(
        #             gpus[0],
        #             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
        #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        #     except RuntimeError as e:
        #         # Virtual devices must be set before GPUs have been initialized
        #         print(e)

        self.single_img_length = 200

        self.single_input = keras.Input(shape=(self.config.INPUT_DIM * self.single_img_length, 10 * self.single_img_length, 3), \
                                        name="Input")
        self.sensor_input = [x for x in range(self.config.INPUT_DIM)]
        self.conv1 = [x for x in range(self.config.INPUT_DIM)]
        self.conv2 = [x for x in range(self.config.INPUT_DIM)]
        self.sensor_output = [x for x in range(self.config.INPUT_DIM)]
        for sensor in range(self.config.INPUT_DIM):
            self.sensor_input[sensor] = self.single_input[:, sensor * self.single_img_length : (sensor + 1) * self.single_img_length, :, :]
            self.sensor_input[sensor] = Reshape((10, self.single_img_length, self.single_img_length, 3), name = "Input_" + self.config.SENSOR_LIST[sensor])(self.sensor_input[sensor])
#=======================================================Conv1==============================================
            self.conv1[sensor] = Conv3D(64, (1, 5, 5), (1, 3, 3),
                                        name="conv_" + self.config.SENSOR_LIST[sensor] + "_1",
                                        kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE))(self.sensor_input[sensor])
            self.conv1[sensor] = Activation("relu", name="conv_" + self.config.SENSOR_LIST[sensor] + "_1_relu")(self.conv1[sensor])
            self.conv1[sensor] = Dropout(DROPOUT_RATIO, noise_shape=[BATCH_SIZE, 1, 1, 1, self.conv1[sensor].shape[-1]], name="conv_" + self.config.SENSOR_LIST[sensor] + "_1_dropout")(self.conv1[sensor])
            self.conv1[sensor] = AveragePooling3D((1, 2, 2),name="conv_" + self.config.SENSOR_LIST[sensor] + "_1_pool")(self.conv1[sensor])  
#=======================================================Conv2==============================================
            self.conv2[sensor] = Conv3D(64, (1, 3, 3), (1, 1, 1),
                                        name="conv_" + self.config.SENSOR_LIST[sensor] + "_2",
                                        padding="SAME",
                                        kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE))(self.conv1[sensor])
            self.conv2[sensor] = Activation("relu", name="conv_" + self.config.SENSOR_LIST[sensor] + "_2_relu")(self.conv2[sensor])
            self.conv2[sensor] = Dropout(DROPOUT_RATIO, noise_shape=[BATCH_SIZE, 1, 1, 1, self.conv2[sensor].shape[-1]], name="conv_" + self.config.SENSOR_LIST[sensor] + "_2_dropout")(self.conv2[sensor])
            self.conv2[sensor] = AveragePooling3D((1, 2, 2), name="conv_" + self.config.SENSOR_LIST[sensor] + "_2_pool")(self.conv2[sensor])
#======================================================Output==============================================
            self.sensor_output[sensor] = Reshape((10, 1, 16*16, self.conv1[sensor].shape[-1]), name="output_" + self.config.SENSOR_LIST[sensor])(self.conv2[sensor])#attention here, maybe errorous
#==========================================================================================================
        # self.acce_input = self.single_input[:, 0:self.single_img_length, :, :]
        # self.gyro_input = self.single_input[:, self.single_img_length:, :, :]
        # self.acce_input = Reshape((10, self.single_img_length, self.single_img_length, 3), name="Input_acce")(
        #     self.acce_input)
        # self.gyro_input = Reshape((10, self.single_img_length, self.single_img_length, 3), name="Input_gyro")(
        #     self.gyro_input)
        # acc_input_dim = self.acce_input.get_shape()
        # print(acc_input_dim)
        # gyro_input_dim = self.gyro_input.get_shape()
        # print(gyro_input_dim)
        # self.conv1a = Conv3D(64, (1, 5, 5), (1, 3, 3),
        #                             name="conv_acce_1",
        #                             kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE)
        #                             )(self.acce_input)
        # self.conv1a = self.acce_input
        # self.conv1a = BatchNormalization(name="conv_acce_1_batchnorm")(self.conv1a)
        # self.conv1a = Activation("relu", name="conv_acce_1_relu")(self.conv1a)
        # self.conv1a = Dropout(DROPOUT_RATIO, noise_shape=[BATCH_SIZE, 1, 1, 1, self.conv1a.shape[-1]], name="conv_acce_1_dropout")(
        #     self.conv1a)
        # self.conv1a = AveragePooling3D((1, 2, 2),name="conv_acce_1_pool")(self.conv1a)
        # acc_conv1a_dim = self.conv1a.get_shape()
        # print(acc_conv1a_dim)

        # self.conv2a = Conv3D(64, (1, 3, 3), (1, 1, 1),
        #                             name="conv_acce_2",
        #                             padding="SAME",
        #                             kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE)
        #                             )(self.conv1a)
        # self.conv2a = BatchNormalization(name="conv_acce_2_batchnorm")(self.conv2a)
        # self.conv2a = Activation("relu", name="conv_acce_2_relu")(self.conv2a)
        # self.conv2a = Dropout(DROPOUT_RATIO, noise_shape=[BATCH_SIZE, 1, 1, 1, self.conv1a.shape[-1]], name="conv_acce_2_dropout")(
        #     self.conv2a)
        # self.conv2a = AveragePooling3D((1, 2, 2), name="conv_acce_2_pool")(self.conv1a)
        # self.conv3a = Conv3D(64, (1, 3, 3), (1, 2, 2),
        #                             name="conv_acce_3",
        #                             kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE)
        #                             )(self.conv2a)
        # # self.conv3a = BatchNormalization(name="conv_acce_3_batchnorm")(self.conv3a)
        # self.conv3a = Activation("relu", name="conv_acce_3_relu")(self.conv3a)
        # self.conv3a = Dropout(DROPOUT_RATIO, noise_shape=[BATCH_SIZE, 1, 1, 1, 64], name="conv_acce_3_dropout")(
        #     self.conv3a)
        # acc_conv2a_dim = self.conv2a.get_shape()
        # print(acc_conv2a_dim)

        # self.acce_output = Reshape((10, 1, 16*16, self.conv1a.shape[-1]), name="output_acce")(self.conv2a)
        # self.acce_output = self.conv3a
        # self.acce_output = AveragePooling3D((1,3,3),(1,2,2))(self.conv3a)
        # self.acce_output = Reshape((10,1,10*10,64),name="output_acce")(self.acce_output)
        # **********************************************************************************************
        # acc_output_dim = self.acce_output.get_shape()
        # print(acc_output_dim)
        # self.conv1g = Conv3D(64, (1, 5, 5), (1, 3, 3),
        #                             name="conv_gyro_1",
        #                             kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE)
        #                             )(self.gyro_input)
        # # self.conv1g = self.gyro_input
        # # self.conv1g = BatchNormalization(name="conv_gyro_1_batchnorm")(self.conv1g)
        # self.conv1g = Activation("relu", name="conv_gyro_1_relu")(self.conv1g)
        # self.conv1g = Dropout(DROPOUT_RATIO, noise_shape=[BATCH_SIZE, 1, 1, 1, self.conv1g.shape[-1]], name="conv_gyro_1_dropout")(
        #     self.conv1g)
        # self.conv1g = AveragePooling3D((1, 2, 2), name="conv_gyro_1_pool")(self.conv1g)
        # gyro_conv1g_dim = self.conv1g.get_shape()
        # print(gyro_conv1g_dim)
        # self.conv2g = Conv3D(64, (1, 3, 3), (1, 1, 1),
        #                             name="conv_gyro_2",
        #                             padding="SAME",
        #                             kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE)
        #                             )(self.conv1g)
        # # self.conv2g = BatchNormalization(name="conv_gyro_2_batchnorm")(self.conv2g)
        # self.conv2g = Activation("relu", name="conv_gyro_2_relu")(self.conv2g)
        # self.conv2g = Dropout(DROPOUT_RATIO, noise_shape=[BATCH_SIZE, 1, 1, 1, self.conv2g.shape[-1]],
        #                       name="conv_gyro_2_dropout")(self.conv2g)
        # self.conv2g = AveragePooling3D((1, 2, 2), name="conv_gyro_2_pool")(self.conv1g)
        # # self.conv3g = Conv3D(64, (1, 3, 3), (1, 2, 2),
        # #                             name="conv_gyro_3",
        # #                             kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE)
        # #                             )(self.conv2g)
        # # # self.conv3g = BatchNormalization(name="conv_gyro_3_batchnorm")(self.conv3g)
        # # self.conv3g = Activation("relu", name="conv_gyro_3_relu")(self.conv3g)
        # # self.conv3g = Dropout(DROPOUT_RATIO, noise_shape=[BATCH_SIZE, 1, 1, 1, 64], name="conv_gyro_3_dropout")(
        # #     self.conv3g)
        # gyro_conv2g_dim = self.conv2g.get_shape()
        # print(gyro_conv2g_dim)
        # self.gyro_output = Reshape((10, 1, 16*16, self.conv1g.shape[-1]), name="output_gyro")(self.conv2g)
        # self.gyro_output = self.conv3g
        # self.gyro_output = AveragePooling3D((1,3,3),(1,2,2))(self.conv3g)
        # self.gyro_output = Reshape((10,1,10*10,64),name="output_gyro")(self.gyro_output)
        # ***********************************************************************************************************************
        # gyro_output_dim = self.gyro_output.get_shape()
        # print(gyro_output_dim)
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
        self.conv1 = Conv3D(64, kernel_size=(1,2,5),
                                   name='conv_merge_1',
                                   strides=( 1, 1,1),
                                   # padding='SAME',
                                   kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE)
                                   )(self.merge_input)
        # self.conv1 = BatchNormalization(name="conv_merge_1_batchnorm")(self.conv1)
        self.conv1 = Activation("relu", name="conv_merge_1_relu")(self.conv1)
        self.conv1 = Dropout(0.2, noise_shape=[BATCH_SIZE, 1, 1, 1, self.conv1.shape[-1]],
                             name="conv_merge_1_dropout")(self.conv1)
        self.conv1 = AveragePooling3D((1, 1,3))(self.conv1)
#========================================Merge Conv2==================================
        # self.conv2 = Conv3D(64, kernel_size=(1,3,5),
        #                            name='conv_merge_2',
        #                            strides=( 1, 1,1),
        #                            #padding='SAME',
        #                            kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE)
        #                            )(self.conv1)
        # # self.conv1 = BatchNormalization(name="conv_merge_1_batchnorm")(self.conv1)
        # self.conv2 = Activation("relu", name="conv_merge_2_relu")(self.conv2)
        # self.conv2 = Dropout(0.2, noise_shape=[BATCH_SIZE, 1, 1, 1, self.conv2.shape[-1]],
        #                      name="conv_merge_2_dropout")(self.conv2)
        # self.conv2 = AveragePooling3D((1, 1,3))(self.conv2)
#=====================================================================================
#========================================Merge Conv3==================================
        # self.conv3 = Conv3D(64, kernel_size=(1,1,5),
        #                            name='conv_merge_2',
        #                            strides=( 1, 1,1),
        #                            # padding='SAME',
        #                            kernel_regularizer=keras.regularizers.l2(REGULARIZER_RATE)
        #                            )(self.conv3)
        # # self.conv1 = BatchNormalization(name="conv_merge_1_batchnorm")(self.conv1)
        # self.conv3 = Activation("relu", name="conv_merge_3_relu")(self.conv3)
        # self.conv3 = Dropout(0.2, noise_shape=[BATCH_SIZE, 1, 1, 1, self.conv3.shape[-1]],
        #                      name="conv_merge_2_dropout")(self.conv3)
        # self.conv3 = AveragePooling3D((1, 1,3))(self.conv3)
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

        self.conv_output = self.conv1
        self.rnn_input = Reshape((10, self.conv_output.shape[-1] * self.conv_output.shape[-2] * self.conv_output.shape[-3]), name="Output_merge")(self.conv_output)

        self.rnn = LSTM(120, return_sequences=True, name="RNN_1")(self.rnn_input)
        self.sum_rnn_out = tf.reduce_sum(self.rnn, axis=1, keep_dims=False)
        self.rnn = LSTM(20, name="RNN_2")(self.rnn)
        self.rnn_output = Dense(self.config.OUTPUT_DIM, 'softmax', name="Softmax")(self.rnn)

        self.model = keras.Model(
            inputs=self.single_input,
            outputs=self.rnn_output)

    def train(self, file_dir, val_dir, epochs, save_dir=None):

        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss=keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['acc'])

        data_init = Tfdata(file_dir,self.config)
        data = data_init.acquire_data()

        val_data_init = Tfdata(val_dir,self.config)
        val_data = val_data_init.acquire_data(False)

        print("Training results:")

        self.history = LossHistory(self.config)

        self.model.fit(data,
                       epochs=epochs,
                       verbose=2,
                       validation_data=val_data,
                       callbacks=[self.history])

        # self.model.save(save_dir)

    def evaluate(self, val_dir, save_dir):
        val_data_init = Tfdata(val_dir,self.config)
        val_data = val_data_init.acquire_data(False)
        print("y_true in evaluation")
        print(val_data_init.raw_labels)

        Y_pred = self.model.predict(val_data)
        y_pred = np.argmax(Y_pred, axis=1)
        y_true = val_data_init.raw_labels[0:len(y_pred) // BATCH_SIZE * BATCH_SIZE]
        print("y_ture in confusion matrix.")
        print(y_true)
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




warnings.filterwarnings("ignore")
#==============================Configuration===========================================
config = Configuration()
config.INTERVAL_LENGTH = 50
config.WINDOW_LENGTH = 25
config.DATASET = 'HAR'
config.USER_LIST = [str(x) for x in range(1,31)]
config.GT_LIST = ['Walking', 'Walking_upstairs', 'Walking_downstaris',
                  'Sitting', 'Standing', 'Laying', 'Stand_to_sit',
                  'Sit_to_stand', 'Sit_to_lie', 'Lie_to_sit',
                  'Stand_to_lie', 'Lie_to_stand']
config.EXP_LIST = [str(x) for x in range(1,62)]
#config.SENSOR_LIST = ['Acc1', 'Gyro1']
config.SENSOR_LIST = ['acc','gyro']
config.DEVICE_LIST = ['SII']

config.fresh()
#=====================================================================================
example = ZeepSenseEasy(config)
example.model.summary()
train_dir = os.path.join(config.DATASET_DIR, 'train_halfoverlap')
test_dir = os.path.join(config.DATASET_DIR, 'test_halfoverlap')
val_dir = os.path.join(config.DATASET_DIR, 'test_halfoverlap')#to swap test to val
h5_dir_list = [config.DATASET,'GAF4ZS','f' + str(config.INTERVAL_LENGTH), 'zs_original.h5']
h5_name = 'zs_halfoverlap.h5'
h5_dir = os.path.join(config.SAVE_DIR, h5_name)
example.train(train_dir,
              test_dir,
              epochs=100, save_dir=h5_dir)
example.evaluate(val_dir,config.SAVE_DIR)
