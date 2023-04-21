import os
from glob import glob
import numpy as np
import glob

from pathlib import Path

from osgeo import osr
from osgeo import gdal

import keras
from keras import utils
from keras.applications import Xception, InceptionResNetV2
from keras import layers

import tensorflow as tf

from PIL import Image
import cv2

from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt

import json


def read_file_names(dir_img):

  file_names=np.array(glob.glob(str(dir_img.joinpath('*.*'))))

  fileid = [Path(x).name[0:3] for x in file_names]
  file_names = [x for _,x in sorted(zip(fileid,file_names))]

  return file_names


# function to remove aux.xml from folder (happens when tif is open in QGIS)
def removeAUX(path_dir):
    dir_list = os.listdir(path_dir)

    for item in dir_list:
        if item.endswith(".aux.xml"):
            os.remove(os.path.join(path_dir, item))
            print(str(item)+" removed")
            
            
# data scaling II
def FitSplineRBF_array(row_array):

    if np.sum(row_array != 0) < 3:
        return

    w = np.asarray(row_array == 0) | np.asarray(row_array == NODATA_VALUE)
    heights = np.linspace(1, len(row_array), len(row_array))

    x_col = heights[~w]
    y_col = row_array[~w]
    sm = 0.25
    rbf = Rbf(x_col, y_col, smooth=sm)

    x_excluded = heights[w]
    y_pred = rbf(x_excluded)

    row_array[w] = y_pred


def minmax_custom_scaling(image_band_stacked):
    X = image_band_stacked
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (255 - 0) + 0
    return X_scaled, int(X.max()), int(X.min())


def target_data_process(target_array):
    '''Converting tri-mask of {1, 2, 3} to three categories.'''
    #return keras.utils.to_categorical(target_array-1)
    return keras.utils.to_categorical(target_array) #use when class numbering goes from 0


# loading label data
def image_to_array_gdal(filenames):
    # number of files
    L = len(filenames)

    # allocation
    raster_list = []
    for file_index in range(L):
        gdal_ds = gdal.Open(filenames[file_index])
        raster_array = gdal_ds.ReadAsArray()
        raster_list.append(raster_array)
    out = np.stack(raster_list, axis=0)
    return out


#shuffle indices            
def shuffle_ind(L,random_seed):
    '''
    Generating random shuffled indices.
    
    Input
    ----------
        L: an int that defines the largest index
        
    Output
    ----------
        a numpy array of shuffled indices with shape = (L,)
    '''
    np.random.seed(random_seed) 
    ind = np.arange(L)
    np.random.shuffle(ind)
    return ind


def flip_hori(img, mask):
    img_flip = tf.image.flip_left_right(img)
    mask_flip = tf.image.flip_left_right(mask)
    return img_flip, mask_flip


def flip_vert(img, mask):
    img_vflip = tf.image.flip_up_down(img)
    mask_vflip = tf.image.flip_up_down(mask)
    return img_vflip, mask_vflip


def rotate(img, mask):
    img_rot = tf.image.rot90(img)
    mask_rot = tf.image.rot90(mask)
    return img_rot, mask_rot



def probab2class(img,img_size,class_names):
    # extract class by probability
    # converts an array of class probabilities to the array of class the pixel most likely belongs to
    # img - array you want to extract the classes from
    # size - size of the image (height, width)
    
    class_array=np.zeros([img_size,img_size]) #create array of zeroes in size of original raster

    for i in range(img_size):
        for j in range(img_size):
            test_value=img[i,j]
            pred_class = class_names[test_value.argmax()]
            class_array[i,j]=pred_class
    
    return class_array


def array2raster(rasterfn,newRasterfn,array):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32,['COMPRESS=DEFLATE', 'TILED=YES','BIGTIFF=IF_NEEDED'])
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


def plot(name,source,version,var1,var2):
    title = name
    plt.plot(source.history[var1])
    plt.plot(source.history[var2])
    plt.title(name+' '+version)
    plt.ylabel(name)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    return plt.show()


def output_export(raster_list, L_test, model_output, size, class_names, output_folder, version, export=False):
    temp_list = []

    if export == False:
        for i in range(L_test):
            temp = model_output[(i,)]

            predict_class = probab2class(temp, size, class_names)
            array = predict_class

            temp_list.append(array)

        out2 = np.stack(temp_list, axis=0)

    else:
        os.mkdir(output_folder)
        #output_path = output_folder + '/TFL_' + version + '_'
        output_path = output_folder.joinpath(f'TFL_{version}_')

        for i in range(L_test):
            temp = model_output[(i,)]

            predict_class = probab2class(temp, size, class_names)
            array = predict_class
            temp_list.append(array)

            rasterfn = raster_list[i]
            name = os.path.basename(os.path.normpath(raster_list[i]))
            newRasterfn = output_path.joinpath(name)

            array2raster(rasterfn, newRasterfn, array)
        out2 = np.stack(temp_list, axis=0)

    return out2
    