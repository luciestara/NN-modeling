import tensorflow as tf

import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, Concatenate, BatchNormalization, Activation, Dropout
from keras.models import Model

#kernel_size=3

#Define model encoder - decoder (VGG backbone)
def conv_block(inputs,num_filters,kernel_size):
  x = Conv2D(num_filters,kernel_size,padding='same')(inputs)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(num_filters,kernel_size,padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  return x

def define_decoder(inputs,skip_layer,num_filters,kernel_size):
  init = tf.keras.initializers.RandomNormal(stddev=0.02)
  x = Conv2DTranspose(num_filters,(2,2),strides=(2,2),padding='same',kernel_initializer=init)(inputs)
  g = Concatenate()([x,skip_layer])
  g = conv_block(g,num_filters,kernel_size)
  return g


def vgg16_unet(vgg16, inputs,num_classes, dropout,kernel_size):
    vgg16.trainable = False
    s1 = vgg16.get_layer('block1_conv2').output
    s2 = vgg16.get_layer('block2_conv2').output
    s3 = vgg16.get_layer('block3_conv3').output
    s4 = vgg16.get_layer('block4_conv3').output  # bottleneck/bridge layer from vgg16
    
    b1 = vgg16.get_layer('block5_conv3').output  # 32

    # Decoder Block
    d1 = define_decoder(b1, s4, 512,kernel_size)
    d2 = define_decoder(d1, s3, 256,kernel_size)
    d3 = define_decoder(d2, s2, 128,kernel_size)
    d4 = define_decoder(d3, s1, 64,kernel_size)  # output layer
    dropout = Dropout(dropout)(d4)
    outputs = Conv2D(num_classes, 1, padding='same', activation='softmax')(dropout)
    model = Model(inputs, outputs)

    return model