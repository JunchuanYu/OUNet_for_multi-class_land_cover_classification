
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
import sys
from random import shuffle
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

def show_5_images(image,label=None,n=1):
    
    for k in range(n):
        fig=plt.figure(figsize=(25,30))
        for i in range(5):
            plt.subplot(k+1,5,k*5+i+1)
            plt.imshow((image[k*5+i,:,:,:3]))
            plt.grid (False)
            plt.axis('off')
        plt.show()
        if label is not None:
            fig=plt.figure(figsize=(25,30))
            for i in range(5):
                plt.subplot(k+1,5,k*5+i+1)
                plt.imshow(label[k*5+i,:,:])
                plt.grid (False)
                plt.axis('off')
        plt.show()
def num_count(arr,num):
    temp=arr.reshape(-1)
    index=np.where(temp==num)
    return np.count_nonzero(index)
## 打印训练曲线，确认训练效果

def plot_fig(H,outdir):
    N=len(H.history['loss'])
    plt.style.use("ggplot")
    plt.figure(figsize=(10,6))
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    # plt.ylim(0,1)

    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(outdir)
def plot_func(data,label):
    fig=plt.figure(figsize=(25,5))
    for i in range(10):
        plt.subplot(1,10,i+1)
        plt.axis('off')
        plt.imshow(data[i,:,:,0:3])
        plt.title('Image'+str(i+1))
    fig.text(0, 0.5, "Image", fontsize=16, va='center', ha='right')
    fig.tight_layout()

    plt.show()
    fig=plt.figure(figsize=(25,5))
    for i in range(10):
        plt.subplot(1,10,i+1)
        plt.axis('off')
        plt.imshow((label[i,:,:]))
        plt.title('Mask'+str(i+1))
    fig.text(0, 0.55, "Mask", fontsize=16,va='center', ha='right')
    fig.tight_layout()
    plt.show()
def new_val_plot(data,label,pred):
    fig=plt.figure(figsize=(25,5))
    for i in range(8):
        plt.subplot(1,8,i+1)
        plt.imshow(data[i,:,:,:3])
        plt.axis('off')
    plt.show()
    fig=plt.figure(figsize=(25,5))
    for i in range(8):
        plt.subplot(1,8,i+1)
        plt.imshow(label[i,:,:])
        plt.axis('off')
    plt.show()
    fig=plt.figure(figsize=(25,5))
    for i in range(8):
        plt.subplot(1,8,i+1)
        plt.imshow(pred[i,:,:])
        plt.axis('off')
    plt.show()
def val_plot_func(data,label,pred):
    fig=plt.figure(figsize=(30,5))
    for i in range(10):
        plt.subplot(1,10,i+1)
        plt.axis('off')
        plt.imshow(data[i,:,:,0:3])
        plt.title('Image'+str(i+1))
    fig.text(-0.04, 0.5, "Image", fontsize=16, va='center', ha='left')
    fig.tight_layout()
    plt.show()

    fig=plt.figure(figsize=(30,5))
    for i in range(10):
        plt.subplot(1,10,i+1)
        plt.axis('off')
        plt.imshow(label[i,:,:])
        # plt.title('Mask'+str(i+1))
    fig.text(-0.04, 0.5, "Mask", fontsize=16, va='center', ha='left')
    fig.tight_layout()
    plt.show()
    fig=plt.figure(figsize=(30,5))
    for i in range(10):
        plt.subplot(1,10,i+1)
        plt.axis('off')
        plt.imshow(pred[i,:,:])
        # plt.title('Prediction'+str(i+1))
    fig.text(-0.04, 0.5, "Prediction", fontsize=16, va='center', ha='left')
    fig.tight_layout()
    plt.show()
def plot_fig(H,outdir):
    N=len(H.history['loss'])
    plt.style.use("ggplot")
    plt.figure(figsize=(10,6))
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.ylim(0,1)

    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(outdir)
def suffle_data(imgd):
    index = [i for i in range(len(imgd))]
    shuffle(index)
    newimg = imgd[index, :, :, :]
    print(newimg.shape)
    return newimg
def stretch(data, lower_percent=5, higher_percent=95):##设置分位数可以剔除个别异常值
    min = np.percentile(data, lower_percent)
    max = np.percentile(data, higher_percent)  
    # out = a + (data - min) * (b - a) / (max - min)   
    data[data<min] = min
    data[data>max] = max
    return data
def stretch_n(band, lower_percent=5, higher_percent=95): 
    band=np.array(band,dtype=np.float32)
    c = np.percentile(band, lower_percent)*1.0
    d = np.percentile(band, higher_percent)*1.0       
    band[band<c] = c
    band[band>d] = d
    out =  (band - c)  / (d - c)  
    # print(np.max(out),np.min(out),c,d)  
    return out.astype(np.float32)

def adjust_contrast(data,n_band=3):    #通过循环对各个波段进行拉伸
    data=np.array(data,dtype=np.float32)
    for img in data:
        for k in range(n_band):
            img[:,:,k] = stretch_n(img[:,:,k])
    return data
def random_crop(image,crop_sz):
    img_sz=image.shape[:2]
    random_x = np.random.randint(0,img_sz[0]-crop_sz+1) ##生成随机点
    random_y = np.random.randint(0,img_sz[1]-crop_sz+1)
    s_img = image[random_x:random_x+crop_sz,random_y:random_y+crop_sz,:] ##以随机点为起始点生成样本框，进行切片
    return s_img
def label_hot(label,n_label=1):
    listlabel=[]
    for i in label:
        mask=i.flatten()
        mask=to_categorical(mask, num_classes=n_label)
        listlabel.append(mask)
    msk=np.asarray(listlabel,dtype='float32')
    msk=msk.reshape((label.shape[0],label.shape[1],label.shape[2],n_label))
#     print(msk.shape)
    return msk
def interpolation(x, shape,method=0):
    import tensorflow as tf
    h_to, w_to = shape
    resized = tf.image.resize(x, [h_to, w_to])
    return resized
def Load_image_by_Gdal(file_path):
    img_file = gdal.Open(file_path, gdal.GA_ReadOnly)
    img_bands = img_file.RasterCount#band num
    img_height = img_file.RasterYSize#height
    img_width = img_file.RasterXSize#width
    img_arr = img_file.ReadAsArray()#获取投影信息
    geomatrix = img_file.GetGeoTransform()#获取仿射矩阵信息
    projection = img_file.GetProjectionRef()
    return img_bands,img_arr, geomatrix, projection
def read_tiff(file):
    img_bands,img_arr, geomatrix, projection =Load_image_by_Gdal(file)
    if img_bands >1 :
        img_arr=img_arr.transpose(( 1, 2,0))
    return img_arr, geomatrix, projection
def Write_Tiff(img_arr, geomatrix, projection,path):
#     img_bands, img_height, img_width = img_arr.shape
    if 'int8' in img_arr.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img_arr.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(img_arr.shape) == 3:
        img_bands, img_height, img_width = img_arr.shape
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, int(img_width), int(img_height), int(img_bands), datatype)
    #     print(path, int(img_width), int(img_height), int(img_bands), datatype)
        if(dataset!= None) and (geomatrix !='') and (projection!=''):
            dataset.SetGeoTransform(geomatrix) #写入仿射变换参数
            dataset.SetProjection(projection) #写入投影
        for i in range(img_bands):
            dataset.GetRasterBand(i+1).WriteArray(img_arr[i])
        del dataset

    elif len(img_arr.shape) == 2:
        # img_arr = np.array([img_arr])
        img_height, img_width = img_arr.shape
        img_bands=1
        #创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, int(img_width), int(img_height), int(img_bands), datatype)
    #     print(path, int(img_width), int(img_height), int(img_bands), datatype)
        if(dataset!= None) and (geomatrix !='') and (projection!=''):
            dataset.SetGeoTransform(geomatrix) #写入仿射变换参数
            dataset.SetProjection(projection) #写入投影
        dataset.GetRasterBand(1).WriteArray(img_arr)
        del dataset
def center_predict(img,model,batch_size,n_label,strides=128,img_size=256):
    corner_size=int(0.25*img_size)
    h,w,c = img.shape
    padding_h = (h//strides + 1) * strides+corner_size+corner_size
    padding_w = (w//strides + 1) * strides+corner_size+corner_size
    
    padding_img = np.zeros((padding_h,padding_w,c),dtype=np.float16)
    padding_img[corner_size:corner_size+h,corner_size:corner_size+w,:] = img[:,:,:]
    mask_whole = np.zeros((padding_h,padding_w,n_label),dtype=np.float16)
    crop_batch=[]
    for i in range(h//strides+1):
        for j in range(w//strides+1):
            crop_img = padding_img[i*strides:i*strides+img_size,j*strides:j*strides+img_size,:]
            ch,cw,c = crop_img.shape
            
            if ch != img_size or cw != img_size:
                continue
            crop_batch.append(crop_img)
            
    crop_batch=np.array(crop_batch)
    start_time=time.time()
    pred=model.predict(crop_batch,batch_size=batch_size)

    for i in range(h//strides+1):
        for j in range(w//strides+1):
            mask_whole[i*strides+corner_size:i*strides+img_size-corner_size,j*strides+corner_size:j*strides+img_size-corner_size] = pred[(i+1-1)*(w//strides+1)+(j+1)-1,corner_size:img_size-corner_size,corner_size:img_size-corner_size]
    score = mask_whole[corner_size:corner_size+h,corner_size:corner_size+w]
    end_time=time.time()
    print('pred_time:',end_time-start_time)
    return score
def ConfusionMatrix(numClass, imgPredict, Label):  
    #  返回混淆矩阵
    mask = (Label >= 0) & (Label < numClass)  
    label = numClass * Label[mask] + imgPredict[mask]  
    count = np.bincount(label, minlength = numClass**2)  
    confusionMatrix = count.reshape(numClass, numClass)  
    return confusionMatrix

def OverallAccuracy(confusionMatrix):  
    #  返回所有类的整体像素精度OA
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()  
    return OA

def Precision(confusionMatrix):  
    #  返回所有类别的精确率precision  
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    return precision  

def Recall(confusionMatrix):
    #  返回所有类别的召回率recall
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    return recall

def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score
def IntersectionOverUnion(confusionMatrix):  
    #  返回交并比IoU
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    return IoU

def MeanIntersectionOverUnion(confusionMatrix):  
    #  返回平均交并比mIoU
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    mIoU = np.nanmean(IoU)  
    return mIoU

def call_matric(label_arr,gt_arr,index,if_show=True):
    y_true=gt_arr.reshape((gt_arr.shape[0]*gt_arr.shape[1]*gt_arr.shape[2],1))
    y_predict=label_arr.reshape((label_arr.shape[0]*label_arr.shape[1]*label_arr.shape[2],1))
    confusionMatrix=confusion_matrix(y_true, y_predict)
    precision = Precision(confusionMatrix)
    recall = Recall(confusionMatrix)
    oa = OverallAccuracy(confusionMatrix)
    IoU = IntersectionOverUnion(confusionMatrix)
    mIOU = MeanIntersectionOverUnion(confusionMatrix)
    f1score = F1Score(confusionMatrix)
    oa=precision/precision*oa
    mIOU=precision/precision*mIOU
    temp=np.column_stack((precision,recall,f1score,IoU,oa,mIOU))
    mean=np.mean(temp,axis=0)
    result=np.vstack((temp,mean.transpose()))
    name=['precision','recall','F1-score','iou','oa','miou']
    df2 = pd.DataFrame((result))
    df2.index=index
    df2.columns=name
    return df2