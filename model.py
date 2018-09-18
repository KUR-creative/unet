import numpy as np 
import os
#import skimage.io as io
#import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

'''
'''


# https://www.kaggle.com/aglotero/another-iou-metric  
# iou = tp / (tp + fp + fn)
from skimage.morphology import label
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = label(y_true_in > 0.5)
    y_pred = label(y_pred_in > 0.5)
    
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), 
                                  bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    '''
    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)
    '''
    tp, fp, fn = precision_at(0.5, iou)
    if (tp + fp + fn) > 0:
        ret = tp / (tp + fp + fn)
    else:
        ret = 0
    return ret

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.array(np.mean(metric), dtype=np.float32)

def mean_iou__(label, pred):
    metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float32)
    return metric_value

def mean_iou_(y_true, y_pred, smooth=100):
    y_true = K.cast(y_true > 0.5, dtype='float32')
    y_pred = K.cast(y_pred > 0.5, dtype='float32')
    intersection = K.sum(y_true * y_pred, axis=-1)
    sum_ = K.sum(y_true + y_pred, axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac #* 100 #(1 - jac) * smooth

#https://stackoverflow.com/questions/45947351/how-to-use-tensorflow-metrics-in-keras
# IOU metric: https://www.pyimagesearch.com/wp-content/uploads/2016/09/iou_equation.png
import functools
import tensorflow as tf
def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

@as_keras_metric
def mean_iou(y_true, y_pred, num_classes=2):
    return tf.metrics.mean_iou(y_true, y_pred, num_classes)

def create_weighted_binary_crossentropy(zero_weight, one_weight):
    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)
    return weighted_binary_crossentropy

def set_layer_BN_relu(input,layer_fn,*args,**kargs):
    x = layer_fn(*args,**kargs)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def down_block(x, cnum, kernel_init, filter_vec=(3,3,1), maxpool2x=True):
    for n in filter_vec:
        x = set_layer_BN_relu(x, Conv2D, cnum, (n,n), 
                              padding='same', kernel_initializer=kernel_init)
    if maxpool2x:
        pool = MaxPooling2D(pool_size=(2,2))(x)
        return x, pool
    else:
        return x

def up_block(from_horizon, upward, cnum, kernel_init, filter_vec=(3,3,1)):
    upward = Conv2DTranspose(cnum, (2,2), padding='same', strides=(2,2), kernel_initializer=kernel_init)(upward)
    merged = concatenate([from_horizon,upward], axis=3)
    for n in filter_vec:
        merged = set_layer_BN_relu(merged, Conv2D, cnum, (n,n), padding='same', kernel_initializer=kernel_init)
    return merged

def unet(pretrained_weights = None,input_size = (256,256,1),
         kernel_init='he_normal', 
         num_filters=64, num_maxpool = 4, filter_vec=(3,3,1),
         lr=1e-4, decay=0.0, weight_0=0.5, weight_1=0.5):
    '''
    depth = 4
    inp -> 0-------8 -> out
            1-----7
             2---6
              3-5
               4
    '''
    cnum = num_filters
    depth = num_maxpool

    x = inp = Input(input_size)

    down_convs = [None] * depth
    for i in range(depth): 
        down_convs[i], x = down_block(x, 2**i * cnum, kernel_init, filter_vec=filter_vec)
    x = down_block(x, 2**depth * cnum, kernel_init, filter_vec=filter_vec, maxpool2x=False)    
    for i in reversed(range(depth)): 
        x = up_block(down_convs[i], x, 2**i * cnum, kernel_init, filter_vec=filter_vec)

    out = Conv2D(1,(1,1),padding='same',kernel_initializer=kernel_init,activation = 'sigmoid')(x)

    model = Model(input=inp, output=out)
    from keras_contrib.losses.jaccard import jaccard_distance
    model.compile(optimizer = Adadelta(lr),#Adam(lr = lr,decay=decay), 
                  loss=lambda y_true,y_pred:jaccard_distance(y_true,y_pred), #,smooth=lr
                  metrics=[mean_iou])
    #model.compile(optimizer = Adam(lr = lr), loss = 'binary_crossentropy', metrics = ['accuracy'])
    #model.compile(optimizer = Adam(lr = lr,decay=decay), loss='binary_crossentropy',metrics=[mean_iou])
    #model.compile(optimizer = Adam(lr = lr), 
                  #loss = create_weighted_binary_crossentropy(weight_0,weight_1),
                  #metrics = ['accuracy'])
    #model.compile(optimizer = Adadelta(lr), loss = 'binary_crossentropy', metrics=[mean_iou])
    

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

if __name__ == '__main__':
    from keras.utils import plot_model
    model = unet()
    model.summary()
    plot_model(model, to_file='C_model.png', show_shapes=True)

'''
conv0, half = down_block( inp, 2**0 * cnum, kernel_init)
conv1, half = down_block(half, 2**1 * cnum, kernel_init)
conv2, half = down_block(half, 2**2 * cnum, kernel_init)
conv3, half = down_block(half, 2**3 * cnum, kernel_init)

conv4       = down_block(half, 2**4 * cnum, kernel_init, maxpool2x=False)

conv5 = up_block(conv3, conv4, 2**3 * cnum, kernel_init)
conv6 = up_block(conv2, conv5, 2**2 * cnum, kernel_init)
conv7 = up_block(conv1, conv6, 2**1 * cnum, kernel_init)
conv8 = up_block(conv0, conv7, 2**0 * cnum, kernel_init)
'''

'''
conv1 = set_layer_BN_relu(  inp, Conv2D,  64, (3,3), padding='same', kernel_initializer=kernel_init)
conv1 = set_layer_BN_relu(conv1, Conv2D,  64, (3,3), padding='same', kernel_initializer=kernel_init)
conv1 = set_layer_BN_relu(conv1, Conv2D,  64, (1,1), padding='same', kernel_initializer=kernel_init)
pool = MaxPooling2D(pool_size=(2,2))(conv1)

conv2 = set_layer_BN_relu( pool, Conv2D, 128, (3,3), padding='same', kernel_initializer=kernel_init)
conv2 = set_layer_BN_relu(conv2, Conv2D, 128, (3,3), padding='same', kernel_initializer=kernel_init)
conv2 = set_layer_BN_relu(conv2, Conv2D, 128, (1,1), padding='same', kernel_initializer=kernel_init)
pool = MaxPooling2D(pool_size=(2,2))(conv2)

conv3 = set_layer_BN_relu( pool, Conv2D, 256, (3,3), padding='same', kernel_initializer=kernel_init)
conv3 = set_layer_BN_relu(conv3, Conv2D, 256, (3,3), padding='same', kernel_initializer=kernel_init)
conv3 = set_layer_BN_relu(conv3, Conv2D, 256, (1,1), padding='same', kernel_initializer=kernel_init)
pool = MaxPooling2D(pool_size=(2,2))(conv3)

conv4 = set_layer_BN_relu( pool, Conv2D, 512, (3,3), padding='same', kernel_initializer=kernel_init)
conv4 = set_layer_BN_relu(conv4, Conv2D, 512, (3,3), padding='same', kernel_initializer=kernel_init)
conv4 = set_layer_BN_relu(conv4, Conv2D, 512, (1,1), padding='same', kernel_initializer=kernel_init)
pool = MaxPooling2D(pool_size=(2,2))(conv4)

conv5 = set_layer_BN_relu( pool, Conv2D,1024, (3,3), padding='same', kernel_initializer=kernel_init)
conv5 = set_layer_BN_relu(conv5, Conv2D,1024, (3,3), padding='same', kernel_initializer=kernel_init)
conv5 = set_layer_BN_relu(conv5, Conv2D,1024, (1,1), padding='same', kernel_initializer=kernel_init)

conv6 = Conv2DTranspose(512, (2,2), padding='same', strides=(2,2), kernel_initializer=kernel_init)(conv5)
merge6 = concatenate([conv4,conv6], axis=3)
conv6 = set_layer_BN_relu(merge6, Conv2D, 512, (3,3), padding='same', kernel_initializer=kernel_init)
conv6 = set_layer_BN_relu( conv6, Conv2D, 512, (3,3), padding='same', kernel_initializer=kernel_init)
conv6 = set_layer_BN_relu( conv6, Conv2D, 512, (1,1), padding='same', kernel_initializer=kernel_init)

conv7 = Conv2DTranspose(256, (2,2), padding='same', strides=(2,2), kernel_initializer=kernel_init)(conv6)
merge7 = concatenate([conv3,conv7], axis=3)
conv7 = set_layer_BN_relu(merge7, Conv2D, 256, (3,3), padding='same', kernel_initializer=kernel_init)
conv7 = set_layer_BN_relu( conv7, Conv2D, 256, (3,3), padding='same', kernel_initializer=kernel_init)
conv7 = set_layer_BN_relu( conv7, Conv2D, 256, (1,1), padding='same', kernel_initializer=kernel_init)

conv8 = Conv2DTranspose(128, (2,2), padding='same', strides=(2,2), kernel_initializer=kernel_init)(conv7)
merge8 = concatenate([conv2,conv8], axis=3)
conv8 = set_layer_BN_relu(merge8, Conv2D, 128, (3,3), padding='same', kernel_initializer=kernel_init)
conv8 = set_layer_BN_relu( conv8, Conv2D, 128, (3,3), padding='same', kernel_initializer=kernel_init)
conv8 = set_layer_BN_relu( conv8, Conv2D, 128, (1,1), padding='same', kernel_initializer=kernel_init)

conv9 = Conv2DTranspose(64, (2,2), padding='same', strides=(2,2), kernel_initializer=kernel_init)(conv8)
merge9 = concatenate([conv1,conv9], axis=3)
conv9 = set_layer_BN_relu(merge9, Conv2D,  64, (3,3), padding='same', kernel_initializer=kernel_init)
conv9 = set_layer_BN_relu( conv9, Conv2D,  64, (3,3), padding='same', kernel_initializer=kernel_init)
conv9 = set_layer_BN_relu( conv9, Conv2D,  64, (1,1), padding='same', kernel_initializer=kernel_init)
'''
