import os

from sklearn.preprocessing import MinMaxScaler

from osgeo import gdal
import numpy
import json
from glob import glob


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

def normalization(image_list,band_list):

    ds = gdal.Open(str(image_list[0]))
    image_array = ds.ReadAsArray()
    nrow = numpy.shape(image_array)[1]
    ncol = numpy.shape(image_array)[2]

    image_dict = {}
    scaling_dict = {}
    for band_number, band_item in enumerate(band_list):

        # collect a particular band values from all images and do min max scaling
        image_band_stack = []
        image_band_stacked=[]
        image_band_stack_scaled_custom = []
        for image_item in image_list:
            ds = gdal.Open(str(image_item))
            image_array = ds.ReadAsArray()
            image_band = image_array[band_number,:,:]
            image_band_stack.append(image_band.flatten())
        image_band_stacked = numpy.stack(image_band_stack, axis=0)
        # FIX ME do spline interpolation for missing data in each column of image_band_stack using the following line
        # numpy.apply_along_axis(FitSplineRBF_array, 0, image_band_stacked)

        image_band_stack_scaled_custom, band_max, band_min = minmax_custom_scaling(image_band_stacked)

        # create a dict where for each item: image name is key and put the band values in it
        for row_index in range(len(image_band_stack_scaled_custom)):
            image_band_stack_scaled_custom_row = image_band_stack_scaled_custom[row_index,:]
            if band_number ==0:
                image_dict[image_list[row_index]] = []
                image_dict[image_list[row_index]].append(image_band_stack_scaled_custom_row.reshape(nrow, ncol))
            else:
                image_dict[image_list[row_index]].append(image_band_stack_scaled_custom_row.reshape(nrow, ncol))

        scaling_dict[band_item] = {'max':band_max, 'min':band_min}

    image_array_ordered = numpy.moveaxis(numpy.array(list(image_dict.values())), [0, 1, 2, 3], [0, -1, -3, -2])
    #image_array_ordered = numpy.moveaxis(numpy.array(list(image_dict.values())), [0, 1, 2, 3,4,5,6,7,8], [0, -1, -8,-7,-6,-5,-4,-3, -2])
    return image_array_ordered, scaling_dict

"""
# store the band max and min somewhere
scaling_parameter_filepath = 'scaling_max_min.json'
json_object = json.dumps(scaling_dict)
# Writing to file
with open(scaling_parameter_filepath, "w") as outfile:
    outfile.write(json_object)

# to read json file
parameterfile = open(scaling_parameter_filepath,)
scaling_parameters = json.load(parameterfile)
print("here")
"""