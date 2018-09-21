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

def jaccard_distance(y_true, y_pred, smooth=100, weight1=1.):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(y_true + y_pred, axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth * weight1 #weighted to label 1

def jaccard_coefficient(y_true, y_pred, smooth=100, weight1=1.):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(y_true + y_pred, axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

'''
def jaccard_distance(y_true, y_pred, smooth=100, weight1=1.):
    intersection = K.sum(K.abs(y_true * y_pred), axis=(1,2,3))
    sum_ = K.sum(y_true + y_pred, axis=(1,2,3))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth * weight1 #weighted to label 1
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
def mean_iou_(y_true, y_pred, num_classes=2):
    return tf.metrics.mean_iou(y_true, y_pred, num_classes)

def build_weighted_binary_crossentropy(weight_0, weight_1):
    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * weight_1 + (1. - y_true) * weight_0
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
         num_classes=1, last_activation='sigmoid',
         num_filters=64, num_maxpool = 4, filter_vec=(3,3,1),
         weight_0=0.5, weight_1=0.5,
         loss='jaccard',optimizer='Adadelta'):
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

    print('nc:',num_classes, 'la:',last_activation)
    out = Conv2D(num_classes, (1,1), padding='same',
                 kernel_initializer=kernel_init, activation = last_activation)(x)

    if loss == 'jaccard':
        loss = lambda y_true,y_pred:jaccard_distance(y_true,y_pred,100)#,weight_1)
    elif loss == 'wbce': 
        loss = build_weighted_binary_crossentropy(weight_0, weight_1)

    @as_keras_metric
    def mean_iou(y_true, y_pred):
        return tf.metrics.mean_iou(y_true, y_pred, num_classes)
    model = Model(input=inp, output=out)
    model.compile(optimizer=optimizer,#Adam(lr = lr,decay=decay), 
                  loss=loss, metrics=[mean_iou])
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
    model = unet(num_classes=4,last_activation='softmax')
    model.summary()
    plot_model(model, to_file='C_model.png', show_shapes=True)
