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
## DEEPLAB V3 PLUS

    def interpolation(self,x, shape,method=0):
        # """0：双线性差值。1：最近邻居法。2：双三次插值法。3：面积插值法"""
        import tensorflow as tf
        # The height and breadth to which the pooled feature maps are to be interpolated
        h_to, w_to = shape
        # Bilinear Interpolation (Default method of this tf function is method=ResizeMethod.BILINEAR)
        resized = tf.image.resize(x, [h_to, w_to])
        return resized
    def SepConv_BN(self,x, filters, prefix, stride=1, kernel_size=3, rate=1, epsilon=1e-5):

        x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),padding='same', use_bias=False, name=prefix + '_depthwise')(x)
        x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (1, 1), padding='same',use_bias=False, name=prefix + '_pointwise')(x)
        x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
        x = Activation('relu')(x)

        return x
    def Xception_head2(self):
        img_input=self.img_input
        ## enter flow
        x = Conv2D(32, (3, 3), strides=(2, 2),padding='same', use_bias=False, name='block1_conv1')(img_input)
        x = BatchNormalization(name='block1_conv1_bn')(x)
        x = Activation('relu', name='block1_conv1_act')(x)
        x = Conv2D(64, (3, 3), padding='same',use_bias=False, name='block1_conv2')(x)
        x = BatchNormalization(name='block1_conv2_bn')(x)
        x = Activation('relu', name='block1_conv2_act')(x)
        f1=x#128
        residual = Conv2D(128, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
        x = BatchNormalization(name='block2_sepconv1_bn')(x)
        x = Activation('relu', name='block2_sepconv2_act')(x)
        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
        x = BatchNormalization(name='block2_sepconv2_bn')(x)
        x = SeparableConv2D(128, (3, 3), padding='same', strides=(2, 2), name='block2_pool')(x)
        x = add([x, residual])
        
        f2=x#64
        residual = Conv2D(256, (1, 1), strides=(1, 1),padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu', name='block3_sepconv1_act')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
        x = BatchNormalization(name='block3_sepconv1_bn')(x)
        x = Activation('relu', name='block3_sepconv2_act')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
        x = BatchNormalization(name='block3_sepconv2_bn')(x)

        x = SeparableConv2D(256, (3, 3), padding='same',strides=(1, 1), name='block3_pool')(x)
        x = add([x, residual])
        f3=x#64
        residual = Conv2D(728, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu', name='block4_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
        x = BatchNormalization(name='block4_sepconv1_bn')(x)
        x = Activation('relu', name='block4_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
        x = BatchNormalization(name='block4_sepconv2_bn')(x)

        x = SeparableConv2D(728, (3, 3), padding='same',strides=(2, 2), name='block4_pool')(x)
        x = add([x, residual])
        ## middle flow
        for i in range(8):
            residual = x
            prefix = 'block' + str(i + 5)

            x = Activation('relu', name=prefix + '_sepconv1_act')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
            x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
            x = Activation('relu', name=prefix + '_sepconv2_act')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
            x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
            x = Activation('relu', name=prefix + '_sepconv3_act')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
            x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

            x = add([x, residual])

        ## exit flow
        residual = Conv2D(1024, (1, 1), strides=(1,1),padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu',name='block21_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same',dilation_rate=(2, 2), use_bias=False, name='block21_sepconv1')(x)
        x = BatchNormalization(name='block21_sepconv1_bn')(x)
        x = Activation('relu', name='block21_sepconv2_act')(x)
        x = SeparableConv2D(1024, (3, 3), padding='same',dilation_rate=(2, 2),use_bias=False, name='block21_sepconv2')(x)
        x = BatchNormalization(name='block21_sepconv2_bn')(x)
        f4=x #32
        # the last stage stride=1,which is different with the original xception.
        x = SeparableConv2D(1024, (3, 3), strides=(1,1), dilation_rate=(2, 2),padding='same', name='block21_pool')(x)
        x = add([x, residual])

        x = SeparableConv2D(1536, (3, 3), padding='same',dilation_rate=(4, 4), use_bias=False, name='block22_sepconv1')(x)
        x = BatchNormalization(name='block22_sepconv1_bn')(x)
        x = Activation('relu', name='block22_sepconv1_act')(x)

        x = SeparableConv2D(1536, (3, 3), padding='same',dilation_rate=(4, 4), use_bias=False, name='block23_sepconv1')(x)
        x = BatchNormalization(name='block23_sepconv1_bn')(x)
        x = Activation('relu', name='block23_sepconv1_act')(x)

        x = SeparableConv2D(2048, (3, 3), padding='same', dilation_rate=(4, 4),use_bias=False, name='block24_sepconv2')(x)
        x = BatchNormalization(name='block24_sepconv2_bn')(x)
        x = Activation('relu', name='block24_sepconv2_act')(x)
        f5=x #32
#         fgap = GlobalAveragePooling2D(name='avg_pool')(x)
#         x = Dense(1, activation='sigmoid', name='predictions')(x)

#         model = Model(self.img_input, x, name='xception')
        return img_input,[f1, f2, f3, f4, f5] 
    def deeplabv3p(self):
        inputs=self.img_input
        XCEPTION_input2, levels = self.Xception_head2()
        [f1, f2, f3, f4, f5] = levels
#         print(K.int_shape(f3),K.int_shape(f4),K.int_shape(f5))
        #F3 64,F4=F5=32
        b4 = GlobalAveragePooling2D()(f5)

        # from (b_size, channels)->(b_size, 1, 1, channels)
        b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
        b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
        b4 = Conv2D(256, (1, 1), padding='same',use_bias=False, name='image_pooling')(b4)
        b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
        b4 = Activation('relu')(b4)
        # upsample. have to use compat because of the option align_corners
        size_before = K.int_shape(f5)
        b4 = Lambda(self.interpolation, arguments={'shape': (size_before[1], size_before[2])})(b4)
#         print(K.int_shape(b4))

    # simple 1x1
        b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(f5)
        b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
        b0 = Activation('relu', name='aspp0_activation')(b0)    
        # Dilated conv block
        atrous_rates = (6, 12, 18)

        # rate = 6 (12)
        b1 = self.SepConv_BN(f5, 256, 'aspp1',rate=atrous_rates[0])
        # rate = 12 (24)
        b2 = self.SepConv_BN(f5, 256, 'aspp2',rate=atrous_rates[1])
        # rate = 18 (36)
        b3 = self.SepConv_BN(f5, 256, 'aspp3',rate=atrous_rates[2])

        # concatenate ASPP branches & project
        x = Concatenate()([b4, b0, b1, b2, b3])
        x = Conv2D(256, (1, 1), padding='same',use_bias=False, name='concat_projection')(x)
        x = BatchNormalization(name='concat_projection_BN')(x)
        x = Activation('relu')(x)
        x = Dropout(0.1)(x)
        # X=32
        # DeepLab v.3+ decoder
#         size_before2 = tf.keras.backend.int_shape(f4)
#         x = Lambda(self.interpolation, arguments={'shape': (size_before2[1], size_before2[2])})(x)
        x = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear', name='4x_Upsampling1')(x)

        dec_skip1 = Conv2D(48, (1, 1), padding='same',use_bias=False, name='feature_projection0')(f3)
        dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = Activation(tf.nn.relu)(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        # X=64
        x = self.SepConv_BN(x, 256, 'decoder_conv0')
        x = self.SepConv_BN(x, 256, 'decoder_conv1')      
        x = UpSampling2D(size=(4,4), data_format='channels_last', interpolation='bilinear', name='4x_Upsampling2')(x)
        # X=256
#         x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
#         size_before3 = tf.keras.backend.int_shape(inputs)
#         x = Lambda(self.interpolation, arguments={'shape': (size_before3[1], size_before3[2])})(x)

        outputs=Conv2D(filters=self.nClasses,kernel_size=1,strides=1,padding="same",name="final_outputs")(x)
        outputs = Activation(activation='softmax')(outputs)

        self.deeplabv3model=Model(inputs=inputs,outputs=outputs)
        self.deeplabv3model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)
        return self.deeplabv3model

    def depconvgroup(self,x, filters, prefix, stride=1, kernel_size=3, rate=1, epsilon=1e-5):
        x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),padding='same', use_bias=False, name=prefix + '_depthwise')(x)
        # x = LayerNormalization(epsilon=1e-6)(x)
        x =BatchNormalization()(x)

        x = Conv2D(filters, kernel_size=(1,1), strides=(1,1), padding="same",name='conv2d'+prefix)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size=(1,1), strides=(1,1), padding="same",name='conv2d2'+prefix)(x)
        return x
    def channel_attention(self,input_feature, outfilter=256,ifadd=True):
        filter_kernels = input_feature.shape[-1]
        print(filter_kernels)
        z = GlobalAveragePooling2D()(input_feature)
        z = Reshape((1,1,filter_kernels))(z)
        s = Dense(filter_kernels, activation='relu', use_bias=False)(z)
        s = Dense(filter_kernels, activation='sigmoid', use_bias=False)(s)
        x = multiply([input_feature, s])
        if ifadd:
            x = add([input_feature, x])
        x = Conv2D(outfilter, (1, 1), padding='same')(x)
        return x
    
    def Encoder(self,inputs):
        # inputs=Input(shape(22,224,3))

        #stage 1 第一次下采样
        x_stage_1=self.depconvgroup(inputs,64,'stage1',rate=1)
        #256
        x=MaxPooling2D(pool_size=(2,2),strides=(2,2))(x_stage_1)

        #stage 2   第二次下采样
        x_stage_2=self.depconvgroup(x,128,'stage2',rate=1)
        #128
        x=MaxPooling2D(pool_size=(2,2),strides=(2,2))(x_stage_2)

        #stage 3   第三次下采样
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
    def convattunet(self):
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
    ## Dlinknet
    def residual_block(self,input_tensor, num_filters):
        x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        x = Conv2D(num_filters, (3, 3), padding='same')(x)
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
    def dlinknet(self):
        inputs=self.img_input
        inputs_ = Conv2D(64, kernel_size=(3, 3), padding='same')(inputs)
        inputs_ = BatchNormalization()(inputs_)
        inputs_ = Activation('relu')(inputs_)
        max_pool_inputs = MaxPooling2D((2, 2), strides=(2, 2))(inputs_)

        encoded_1, encoded_pool_1 = self.encoder_block(max_pool_inputs, num_filters=64, num_res_blocks=3)
        encoded_2, encoded_pool_2 = self.encoder_block(encoded_pool_1, num_filters=128, num_res_blocks=4)
        encoded_3, encoded_pool_3 = self.encoder_block(encoded_pool_2, num_filters=256, num_res_blocks=6)
        encoded_4, encoded_pool_4 = self.encoder_block(encoded_pool_3, num_filters=512, num_res_blocks=3)

        center = self.dilated_center_block(encoded_4, 512)

        decoded_1 = Add()([self.decoder_block(center, 256), encoded_3])
        decoded_2 = Add()([self.decoder_block(decoded_1, 128), encoded_2])
        decoded_3 = Add()([self.decoder_block(decoded_2, 64), encoded_1])
        decoded_4 = self.decoder_block(decoded_3, 64)

        final = Conv2DTranspose(32, kernel_size=(3, 3), padding='same')(decoded_4)
        outputs=Conv2D(filters=self.nClasses,kernel_size=1,strides=1,padding="same",name="final_outputs")(final)
        outputs = Activation(activation='softmax')(outputs)
        self.model=Model(inputs=inputs,outputs=outputs)
        self.model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)
        self.model=Model(inputs=inputs,outputs=outputs)
        self.model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)
        # self.model.summary()
        return self.model

   ## BISENET
    def conv_bn_act(self,inputs, n_filters=64, kernel=(2, 2), strides=1, activation='relu'):

        conv = Conv2D(n_filters, kernel_size= kernel, strides = strides,padding='same', data_format='channels_last')(inputs)
        conv = BatchNormalization()(conv)
        conv = Activation(activation)(conv)

        return conv
    def conv_act(self,inputs, n_filters, kernel = (1,1), activation = 'relu', pooling = False):
        if pooling:
            conv = AveragePooling2D(pool_size=(1, 1), padding='same', data_format='channels_last')(inputs)
            conv = Conv2D(n_filters, kernel_size= kernel, strides=1)(conv)
            conv = Activation(activation)(conv)
        else:
            conv = Conv2D(n_filters, kernel_size= kernel, strides=1)(inputs)
            conv = Activation(activation)(conv)
        return conv
    def ARM(self,inputs, n_filters):

        # ARM (Attention Refinement Module)
        # Refines features at each stage of the Context path
        # Negligible computation cost
        arm = AveragePooling2D(pool_size=(1, 1), padding='same', data_format='channels_last')(inputs)
        arm = self.conv_bn_act(arm, n_filters, (1, 1), activation='sigmoid')
        arm = multiply([inputs, arm])

        return arm
    def CP_ARM(self,layer_13, layer_14):

        # ARM
        ARM_13 = self.ARM(layer_13, 1024)
        ARM_14 = self.ARM(layer_14, 2048)

        fgap = GlobalAveragePooling2D()(layer_14)
        ARM_14 = multiply([fgap, ARM_14])

        ARM_13 = UpSampling2D(size=2, data_format='channels_last', interpolation='nearest')(ARM_13)
        ARM_14 = UpSampling2D(size=4, data_format='channels_last', interpolation='nearest')(ARM_14)

        context_features = Concatenate(axis=-1)([ARM_13, ARM_14])
        return context_features,ARM_13,ARM_14
    def FFM(self,input_sp, input_cp, n_classes):
        # FFM (Feature Fusion Module)
        # used to fuse features from the SP & CP
        # because SP encodes low-level and CP high-level features
        ffm = Concatenate(axis=-1)([input_sp, input_cp])
        conv = self.conv_bn_act(ffm, n_classes, (3, 3), strides= 1)

        conv_1 = self.conv_act(conv, n_classes, (1,1), pooling= True)
        conv_1 = self.conv_act(conv_1, n_classes, (1,1), activation='sigmoid')

        ffm = multiply([conv, conv_1])
        ffm = Add()([conv, ffm])

        return ffm
    
    def BISENet(self):
        inputs=self.img_input
        XCEPTION_input, levels = self.Xception_head()
        [f1, f2, f3, f4, f5] = levels

#         x = Lambda(lambda image: ktf.image.resize_images(image, (self.input_height, self.input_weight)))(inputs)
        # Spatial Path (conv_bn_act with strides = 2 )
        SP = self.conv_bn_act(inputs, 32, strides=2)
        SP = self.conv_bn_act(SP, 64, strides=2)
        SP = self.conv_bn_act(SP, 128, strides=2)
        # Context_path (Xception backbone and Attetion Refinement Module(ARM))
        # Context path & ARM
        con_arm,ARM_13,ARM_14 = self.CP_ARM(f4, f5)
        # Feature Fusion Module(FFM)
        FFM = self.FFM(SP, con_arm, self.nClasses)
        # Upsampling the ouput to normal size
        outputs = UpSampling2D(size=(8,8), data_format='channels_last',name="upsample_8",interpolation='bilinear')(FFM)
        outputs=Conv2D(self.nClasses,1,strides=1,activation = 'softmax',name="lossend")(outputs)
        self.model = Model(inputs=inputs,outputs=outputs)
        self.model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS,metrics=self.METRICS)
        # self.model.summary()
        return self.model
## CONVNEXT50_head
    def group_block(self,inputs, filters, stage, block, strides=(1,1)):

        conv_name = "conv2d-{stage}{block}-branch".format(stage=str(stage),block=str(block))
        bn_name = "batchnorm-{stage}{block}-branch".format(stage=str(stage),block=str(block))

        x_shortcut = inputs
        x = inputs
        x = DepthwiseConv2D(kernel_size=(7,7), strides=(1,1), padding="same",name='depthconv2d'+conv_name)(x)
        # x = LayerNormalization(epsilon=1e-6)(x)
        x =BatchNormalization()(x)
        x = Conv2D(filters*4, kernel_size=(1,1), strides=(1,1), padding="same", name=conv_name+'pointconv1'+conv_name)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size=(1,1), strides=(1,1), padding="same", name=conv_name+'pointconv2'+conv_name)(x)

        # Last step of the identity block, shortcut concatenation
        x = Add()([x,x_shortcut])
        x = Activation('relu')(x)

        return x
    def downsizelayer(self,inputs, filters, stage, block):
        conv_name = "conv2d-{stage}{block}-branch".format(stage=str(stage),block=str(block))
        # x = LayerNormalization(epsilon=1e-6)(inputs)
        x =BatchNormalization()(inputs)

        # x=Conv2D(filters, kernel_size=(3,3),strides=2, padding="same",name='stridconv'+conv_name)(x)
        x=Conv2D(filters, kernel_size=(2,2),strides=2, padding="same",name='stridconv'+conv_name)(x)
        return x
    def convnext50_head(self):
        img_input=self.img_input
        # x = Conv2D(96, kernel_size=(3,3), strides=(2,2),padding='same')(img_input)
        x = Conv2D(96, kernel_size=(4,4), strides=(2,2),padding='same')(img_input)
        # x = LayerNormalization(epsilon=1e-6)(x)
        x =BatchNormalization()(x)

        f1=x #128
        #stem
        depths=[96, 192,384, 768]
        # depths=[128, 256, 512, 1024]#96 192 384 768
        # x = self.downsizelayer(inputs=x, filters=filters[0])(x)
        x = self.group_block(inputs=x, filters=depths[0], stage=2, block="b")
        x = self.group_block(inputs=x, filters=depths[0], stage=2, block="c")
        x = self.group_block(inputs=x, filters=depths[0], stage=2, block="d")

        f2=x #128
        x = self.downsizelayer(inputs=x, filters=depths[1],stage=3, block="a")
        x = self.group_block(inputs=x, filters=depths[1], stage=3, block="b")
        x = self.group_block(inputs=x, filters=depths[1], stage=3, block="c")
        x = self.group_block(inputs=x, filters=depths[1], stage=3, block="d")
        f3=x #64
        x = self.downsizelayer(inputs=x, filters=depths[2],stage=4, block="a")
        x = self.group_block(inputs=x, filters=depths[2], stage=4, block="b")
        x = self.group_block(inputs=x, filters=depths[2], stage=4, block="c")
        x = self.group_block(inputs=x, filters=depths[2], stage=4, block="d")
        x = self.group_block(inputs=x, filters=depths[2], stage=4, block="e")
        x = self.group_block(inputs=x, filters=depths[2], stage=4, block="f")
        # x = self.group_block(inputs=x, filters=depths[2], stage=4, block="g")
        # x = self.group_block(inputs=x, filters=depths[2], stage=4, block="h")
        # x = self.group_block(inputs=x, filters=depths[2], stage=4, block="i")
        f4=x #32
        # Stage 5
        x = self.downsizelayer(inputs=x, filters=depths[3],stage=5, block="a")
        x = self.group_block(inputs=x, filters=depths[3], stage=5, block="b")
        x = self.group_block(inputs=x, filters=depths[3], stage=5, block="c")
        x = self.group_block(inputs=x, filters=depths[3], stage=5, block="d")

        f5=x #16
 
        # Average pooling
#         x = AveragePooling2D(pool_size=(2,2), padding="same")(x)
#         x = Flatten()(x)
#         x = Dense(classes, activation="softmax", kernel_initializer=glorot_uniform(seed=0),name="fc{cls}".format(cls=str(classes)))(x)
#         model = Model(inputs=img_input, outputs=x, name="resnet50")
#         model.summary()
        return img_input,[f1, f2, f3, f4, f5]
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