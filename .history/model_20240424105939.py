from tensorflow.keras import backend as K
import os
import numpy as np
import random
import  tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Reshape,UpSampling2D, Add,DepthwiseConv2D,Dropout,BatchNormalization,ZeroPadding2D,add, multiply,Conv2DTranspose,Flatten,Activation,AveragePooling2D,Dense,SeparableConv2D,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler,CSVLogger,ReduceLROnPlateau
from tensorflow.keras.layers import Lambda
from tensorflow.keras.preprocessing.image import img_to_array,ImageDataGenerator
from tensorflow.keras.utils import plot_model
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class all_model(object):
    def __init__(self,loss,loss_weights,optimizer,metrics,input_height,input_width,nclass,nchannel):
        self.LOSS = loss
        self.OPTIMIZER = optimizer
        self.METRICS = metrics
        self.input_height = input_height
        self.input_width=input_width
        self.nClasses=nclass
        self.nchannel=nchannel
        self.model = None
        self.img_input=Input(shape=(self.input_height, self.input_width, self.nchannel))
        self.loss_weights=loss_weights
    
    def UNET(self):
    #     Patch_size = 256
        inputs=self.img_input
        conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

        up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
        merge6 = Concatenate(axis = -1)([conv4, up6])
        conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = Concatenate(axis = -1)([conv3,up7])
        conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = Concatenate(axis = -1)([conv2,up8])
        conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = Concatenate(axis = -1)([conv1,up9])
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

        outputs = Conv2D(self.nClasses, 1, activation = 'softmax',padding = 'same')(conv9)

        self.model=Model(inputs=inputs,outputs=outputs)
        self.model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)
        return self.model
    
    def Encoder(self,inputs):
        # inputs=Input(shape(22,224,3))

        #stage 1 downsample
        x_stage_1=self.depconvgroup(inputs,64,'stage1',rate=1)
        #256
        x=MaxPooling2D(pool_size=(2,2),strides=(2,2))(x_stage_1)

        #stage 2   downsample
        x_stage_2=self.depconvgroup(x,128,'stage2',rate=1)
        #128
        x=MaxPooling2D(pool_size=(2,2),strides=(2,2))(x_stage_2)

        #stage 3   downsample
        x_stage_3=self.depconvgroup(x,256,'stage3',rate=1)
        #64
        x=MaxPooling2D(pool_size=(2,2),strides=(2,2))(x_stage_3)

        #stage 4,5,6
        x_stage_4=self.depconvgroup(x,512,'stage4',rate=1)
        #32
        x_stage_5=self.depconvgroup(x_stage_4,512,'stage5',rate=2)
        #32
        x_stage_6=self.depconvgroup(x_stage_5,512,'stage6',rate=4)

        return x_stage_1,x_stage_2,x_stage_3,x_stage_4,x_stage_5,x_stage_6
    
    def OUNet(self):
        inputs=self.img_input
        shape=np.array([self.input_height,self.input_width]).astype(int)
        
        x_stage_1,x_stage_2,x_stage_3,x_stage_4,x_stage_5,x_stage_6=self.Encoder(inputs)
        # print(x_stage_1.shape,x_stage_2.shape,x_stage_3.shape,x_stage_4.shape,x_stage_5.shape,x_stage_6.shape)
        #[256,128,64,32,32,32]
        x_c6=self.depconvgroup(x_stage_6,512,'stage6-1',rate=4)

        #skip connection
        x_c6=Concatenate(axis=-1,name="concat_6")([x_c6,x_stage_6])
        x_c5=self.depconvgroup(x_c6,512,'stage5-1',rate=2)

        x_c5=Concatenate(axis=-1,name="concat_5")([x_c5,x_stage_5])
        x_c4=self.depconvgroup(x_c5,512,'stage4-1',rate=1)

        x_c4=Concatenate(axis=-1,name="concat_4")([x_c4,x_stage_4])

        x_c3= UpSampling2D(size=(2, 2))(x_c4)
        x_c3=self.depconvgroup(x_c3,256,'stage3-1',rate=1)

        x_c3=Concatenate(axis=-1,name="concat_3")([x_c3,x_stage_3])

        x_c2= UpSampling2D(size=(2, 2))(x_c3)
        x_c2=self.depconvgroup(x_c2,128,'stage2-1',rate=1)
# 
        x_c2=Concatenate(axis=-1,name="concat_2")([x_c2,x_stage_2])

        x_c1= UpSampling2D(size=(2, 2))(x_c2)
        x_c1=self.depconvgroup(x_c1,128,'stage1-1',rate=1)
        x_c1=Concatenate(axis=-1,name="concat_1")([x_c1,x_stage_1])

        """output 6 path"""
        output_6=Conv2D(filters=64,kernel_size=3,strides=1,padding="same",name="output_6")(x_c6)
        output_6= BatchNormalization(momentum=0.95, axis=-1)(output_6)
        output_6 = Activation(activation='relu')(output_6)
        output_6= UpSampling2D(size=(8, 8))(output_6)

        output_5=Conv2D(filters=64,kernel_size=3,strides=1,padding="same",name="output_5")(x_c5)
        output_5= BatchNormalization(momentum=0.95, axis=-1)(output_5)
        output_5 = Activation(activation='relu')(output_5)
        output_5= UpSampling2D(size=(8, 8))(output_5)

        output_4=Conv2D(filters=64,kernel_size=3,strides=1,padding="same",name="output_4")(x_c4)
        output_4= BatchNormalization(momentum=0.95, axis=-1)(output_4)
        output_4 = Activation(activation='relu')(output_4)
        output_4= UpSampling2D(size=(8, 8))(output_4)

        output_3=Conv2D(filters=64,kernel_size=3,strides=1,padding="same",name="output_3")(x_c3)
        output_3= BatchNormalization(momentum=0.95, axis=-1)(output_3)
        output_3 = Activation(activation='relu')(output_3)
        output_3= UpSampling2D(size=(4, 4))(output_3)

        output_2=Conv2D(filters=64,kernel_size=3,strides=1,padding="same",name="output_2")(x_c2)
        output_2= BatchNormalization(momentum=0.95, axis=-1)(output_2)
        output_2 = Activation(activation='relu')(output_2)
        output_2= UpSampling2D(size=(2, 2))(output_2)

        output_1=Conv2D(filters=64,kernel_size=3,strides=1,padding="same",name="output_1")(x_c1)
        output_1= BatchNormalization(momentum=0.95, axis=-1)(output_1)
        output_1 = Activation(activation='relu')(output_1)

        outputs=Concatenate(axis=-1,name="final_concat")([output_6,output_5,output_4,output_3,output_2,output_1])
        
        output=self.channel_attention(outputs,128)
        outputs=self.depconvgroup(outputs,64,'outALL',rate=1)
        
        output_s=Conv2D(filters=64,kernel_size=3,strides=1,padding="same",name="outputs_pre")(outputs)
        output_s= BatchNormalization(momentum=0.95, axis=-1)(output_s)
        output_s = Activation(activation='relu')(output_s)

        outputs=Conv2D(filters=self.nClasses,kernel_size=1,strides=1,padding="same",name="final_outputs")(outputs)
        outputs = Activation(activation='softmax')(outputs)

        self.model=Model(inputs=inputs,outputs=outputs)
        self.model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)
        return self.model