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
## convnext deeplabv3
    def depConv_BN(self,x,filters,rate,block):
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1),dilation_rate=(rate, rate),padding="same",name='depth'+block)(x)
        # x = LayerNormalization(epsilon=1e-6)(x)
        x =BatchNormalization()(x)
        x = Conv2D(filters, kernel_size=(1,1), strides=(1,1), padding="same",name='conv2d'+block)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size=(1,1), strides=(1,1), padding="same",name='conv2d2'+block)(x)
        return x
    def convnext_dlv3p(self):
        # inputs=self.img_input
        inputs, levels = self.convnext50_head()
        [f1, f2, f3, f4, f5] = levels
        # print(K.int_shape(f1),K.int_shape(f2),K.int_shape(f3))
        # print(K.int_shape(f4),K.int_shape(f5))
        #128.128.64.32.16
        b4 = GlobalAveragePooling2D()(f5)

        # from (b_size, channels)->(b_size, 1, 1, channels)
        b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
        b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
        b4 = Conv2D(256, (1, 1), padding='same',use_bias=False, name='image_pooling')(b4)
        # b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
        # b4 = Activation('relu')(b4)
        # upsample. have to use compat because of the option align_corners
        size_before = K.int_shape(f5)
        b4 = Lambda(self.interpolation, arguments={'shape': (size_before[1], size_before[2])})(b4)
#         print(K.int_shape(b4))

    # simple 1x1
        b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(f5)
        # b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
        # b0 = Activation('relu', name='aspp0_activation')(b0)
        # Dilated conv block
        atrous_rates = (6, 12, 18)

        # rate = 6 (12)
        b1 = self.depConv_BN(f5, 256, rate=atrous_rates[0],block='rate6')
        # rate = 12 (24)
        b2 = self.depConv_BN(f5, 256, rate=atrous_rates[1],block='rate12')
        # rate = 18 (36)
        b3 = self.depConv_BN(f5, 256, rate=atrous_rates[2],block='rate18')

        # concatenate ASPP branches & project
        x = Concatenate()([b4, b0, b1, b2, b3])
        x = Conv2D(256, (1, 1), padding='same',use_bias=False, name='concat_projection')(x)
        # x = BatchNormalization(name='concat_projection_BN')(x)
        # x = Activation('relu')(x)
        x = Dropout(0.1)(x)
        # X=32
        # DeepLab v.3+ decoder
#         size_before2 = tf.keras.backend.int_shape(f4)
#         x = Lambda(self.interpolation, arguments={'shape': (size_before2[1], size_before2[2])})(x)
        x = UpSampling2D(size=(4,4), data_format='channels_last', interpolation='bilinear', name='4x_Upsampling1')(x)
        x = BatchNormalization(name='concat_projection_BN')(x)

        dec_skip1 = Conv2D(128, (1, 1), padding='same',use_bias=False, name='feature_projection0')(f3)
        # dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        # dec_skip1 = Activation(tf.nn.relu)(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        
        x = Conv2D(128, (1, 1), padding='same',use_bias=False, name='feature_projection1')(x)
        x = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear', name='2x_Upsampling1')(x)

        # dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        # dec_skip1 = Activation(tf.nn.relu)(dec_skip1)
        x = Concatenate()([x, f2])
        # X=64
        x = self.depConv_BN(x, 256, 1,'filters')
        x = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear', name='2x_Upsampling2')(x)
        # X=256
#         x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
#         size_before3 = tf.keras.backend.int_shape(inputs)
#         x = Lambda(self.interpolation, arguments={'shape': (size_before3[1], size_before3[2])})(x)
        x = BatchNormalization(name='concat_projection_BN2')(x)

        outputs=Conv2D(filters=self.nClasses,kernel_size=1,strides=1,padding="same",name="final_outputs")(x)
        outputs = Activation(activation='softmax')(outputs)

        self.model=Model(inputs=inputs,outputs=outputs)
        self.model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)
        # self.model.summary()
        return self.model
## Dlinknet_PRO
    def residual_block(self,input_tensor, num_filters):
        x = Conv2D(num_filters, (1, 1), padding='same')(input_tensor)
        x = Conv2D(num_filters, (3, 3), padding='same')(x)
        x = Conv2D(num_filters, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)

        input_tensor = Conv2D(num_filters, (1, 1), padding='same')(input_tensor)
        input_tensor = BatchNormalization()(input_tensor)
        res_tensor = Add()([input_tensor, x])
        res_tensor = Activation('relu')(res_tensor)
        return res_tensor
    def dilated_center_block(self,input_tensor, num_filters):

        dilation_1 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(1, 1), padding='same')(input_tensor)
        dilation_1 = Activation('relu')(dilation_1)

        dilation_2 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(2, 2), padding='same')(dilation_1)
        dilation_2 = Activation('relu')(dilation_2)

        dilation_4 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(4, 4), padding='same')(dilation_2)
        dilation_4 = Activation('relu')(dilation_4)

        dilation_8 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(8, 8), padding='same')(dilation_4)
        dilation_8 = Activation('relu')(dilation_8)

        final_diliation = Add()([input_tensor, dilation_1, dilation_2, dilation_4, dilation_8])

        return final_diliation
    def decoder_block(self,input_tensor, num_filters):
        decoder_tensor = Conv2D(num_filters, (1, 1), padding='same')(input_tensor)
        decoder_tensor = BatchNormalization()(decoder_tensor)
        decoder_tensor = Activation('relu')(decoder_tensor)

        decoder_tensor = Conv2DTranspose(num_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(decoder_tensor)
        decoder_tensor = BatchNormalization()(decoder_tensor)
        decoder_tensor = Activation('relu')(decoder_tensor)

        decoder_tensor = Conv2D(num_filters, (1, 1), padding='same')(decoder_tensor)
        decoder_tensor = BatchNormalization()(decoder_tensor)
        decoder_tensor = Activation('relu')(decoder_tensor)
        return decoder_tensor
    def encoder_block(self,input_tensor, num_filters, num_res_blocks):
        encoded = self.residual_block(input_tensor, num_filters)
        while num_res_blocks > 1:
            encoded = self.residual_block(encoded, num_filters)
            num_res_blocks -= 1
        encoded_pool = MaxPooling2D((2, 2), strides=(2, 2))(encoded)
        return encoded, encoded_pool
    def low_fup(self,input_tensor, num_filters):
        encoded = self.residual_block(input_tensor, num_filters)
        decoder_tensor = Conv2DTranspose(num_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(encoded)
        decoder_tensor = BatchNormalization()(decoder_tensor)
        decoder_tensor = Activation('relu')(decoder_tensor)
        fup=self.convrelu(decoder_tensor,num_filters)
        return encoded,fup
    def low_fupc(self,input_tensor1, input_tensor2,num_filters):
        encoded = self.residual_block(input_tensor1, num_filters)
        ccfeature=Concatenate()([encoded, input_tensor2])
        fup=self.convrelu(ccfeature,num_filters)
        decoder_tensor = Conv2DTranspose(num_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(fup)
        decoder_tensor = BatchNormalization()(decoder_tensor)
        decoder_tensor = Activation('relu')(decoder_tensor)
        return decoder_tensor
    def convrelu(self,input_tensor,num_filters):
        x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        x = Activation('relu')(x)
        return x
    def D_LINK_PRO(self):
        inputs=self.img_input
        inputs_ = Conv2D(64, kernel_size=(7, 7), padding='same')(inputs)
        #256
        inputs_ = BatchNormalization()(inputs_)
        inputs_ = Activation('relu')(inputs_)
        max_pool_inputs = MaxPooling2D((2, 2), strides=(2, 2))(inputs_)
        #128
        encoded_1, encoded_pool_1 = self.encoder_block(max_pool_inputs, num_filters=64, num_res_blocks=3)
        #128,64
        encoded_2, encoded_pool_2 = self.encoder_block(encoded_pool_1, num_filters=128, num_res_blocks=4)
        #64,32
        encoded_3, encoded_pool_3 = self.encoder_block(encoded_pool_2, num_filters=256, num_res_blocks=6)
        #32,16
        encoded_4, encoded_pool_4 = self.encoder_block(encoded_pool_3, num_filters=512, num_res_blocks=3)
        #16,8

        center = self.dilated_center_block(encoded_4, 512)
        #16
 
        new_encoded3,lowfeature_1=self.low_fup(encoded_3,128)
#         print(K.int_shape(lowfeature_1))
        #32,64
        lowfeature_2=self.low_fupc(encoded_2,lowfeature_1,64)
#         print(K.int_shape(lowfeature_2))
        #128
        lowfeature_3=self.low_fupc(max_pool_inputs,lowfeature_2,32)
        #256
        lowfeature=self.convrelu(lowfeature_3,32)
        #256

        decoded_1 = Concatenate()([self.decoder_block(center, 256), encoded_3])
        decoded_1 = self.convrelu(decoded_1,256)
#         print(K.int_shape(decoded_1))
        #32
        decoded_2 = Concatenate()([self.decoder_block(decoded_1, 128), encoded_2])
        decoded_2 = self.convrelu(decoded_2,128)
#         print(K.int_shape(decoded_2))
        #64
        decoded_3 = Concatenate()([self.decoder_block(decoded_2, 64), encoded_1])
        decoded_3 = self.convrelu(decoded_3,64)
#         decoded_3 = Conv2D(64, (3, 3), padding='same')(decoded_3)
#         decoded_3 = Activation('relu')(decoded_3)
        #128
        decoded_4 = Concatenate()([self.decoder_block(decoded_3, 64), lowfeature])
        decoded_4 = self.convrelu(decoded_4,64)   
        #256
#         
        final=Conv2D(256, (3, 3), padding='same')(decoded_4)
#         final = Conv2DTranspose(32, kernel_size=(3, 3), padding='same')(decoded_4)
        outputs=Conv2D(filters=self.nClasses,kernel_size=1,strides=1,padding="same",name="final_outputs")(final)
        outputs = Activation(activation='softmax')(outputs)
        self.model=Model(inputs=inputs,outputs=outputs)
        self.model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)
#         self.model.summary()
        return self.model