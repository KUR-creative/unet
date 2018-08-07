from model import *
from data import *
import os
import numpy as np
import skimage.io as io
import skimage.transform as trans
import cv2
from skimage.viewer import ImageViewer
from utils import file_paths
from itertools import cycle, islice
from data_gen import augmenter, rgb2rgbk, rgbk2rgb
from keras.callbacks import TensorBoard

import re
def human_sorted(iterable):
    ''' Sorts the given iterable in the way that is expected. '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(iterable, key = alphanum_key)

def preprocess(img):
    c = 1 if len(img.shape) == 2 else 3
    h,w = img.shape[:2]
    img = (img / 255).astype(np.float32)
    return img.reshape((h,w,c))

def load_imgs(img_dir, mode_flag=cv2.IMREAD_GRAYSCALE):
    return list(map(lambda path: preprocess(cv2.imread(path, mode_flag)),
                    human_sorted(file_paths(img_dir))))

def gen(imgs, masks, batch_size, num_classes=1):
    assert len(imgs) == len(masks)
    img_flow = cycle(imgs)
    mask_flow = cycle(masks)
    while True:
        aug_det = aug.to_deterministic()
        img_batch = aug_det.augment_images( list(islice(img_flow,batch_size)) )
        mask_batch = aug_det.augment_images( list(islice(mask_flow,batch_size)) )
        if num_classes == 4:
            mask_batch = rgb2rgbk(mask_batch)
        yield img_batch, mask_batch

#---------------------- experiment setting --------------------------
IMG_SIZE = 256
batch_size = 4 
num_classes = 4
num_epochs = 400#200#4000

train_dir = 'data/test_rgb//train'
valid_dir = 'data/test_rgb/valid'
test_dir = 'data/test_rgb/test'
output_dir = 'data/test_rgb/output/'
save_model_path = 'test_rgb.h5' ## NOTE
history_path = 'test_rgb.yml' ## NOTE
steps_per_epoch = 2 # num images: 48 = (12 step) * (4 batch_size)

#train_dir = 'data/gray3masks+3_sep/train'
#valid_dir = 'data/gray3masks+3_sep/valid'
#test_dir = 'data/gray3masks+3_sep/test'
#output_dir = 'data/gray3masks+3_sep/output/'
#save_model_path = 'new_manga.h5' ## NOTE
#history_path = 'new_manga_history.yml' ## NOTE
#steps_per_epoch = 60 # num images: 48 = (12 step) * (4 batch_size)

#train_dir = 'data/seg_data/train'
#valid_dir = 'data/seg_data/valid/'
#test_dir = 'data/seg_data/test/'
#output_dir = 'data/seg_data/output/'
#steps_per_epoch = 8 # 32 = 8step * 4batch
#save_model_path = 'seg_data.h5' ## NOTE
#history_path = 'seg_data_history.yml' ## NOTE

#train_dir = 'data/Benigh_74sep/train'
#valid_dir = 'data/Benigh_74sep/valid'
#test_dir = 'data/Benigh_74sep/test'
#output_dir = 'data/Benigh_74sep/output/'
#save_model_path = 'benigh.h5' ## NOTE
#history_path = 'benigh_history.yml' ## NOTE
#steps_per_epoch = 10 # num images: 37 = (10 step) * (4 batch_size)

#train_dir = 'data/Malignant_91sep/train'
#valid_dir = 'data/Malignant_91sep/valid'
#test_dir = 'data/Malignant_91sep/test'
#output_dir = 'data/Malignant_91sep/output/'
#save_model_path = 'malignant.h5' ## NOTE
#history_path = 'malignant_history.yml' ## NOTE
#steps_per_epoch = 12 # num images: 48 = (12 step) * (4 batch_size)
#--------------------------------------------------------------------

#-------------------- ready to generate batch -----------------------
train_imgs = load_imgs(train_dir+'/image') 
train_masks = load_imgs(train_dir+'/label', cv2.IMREAD_COLOR)
valid_imgs = load_imgs(valid_dir+'/image')
valid_masks = load_imgs(valid_dir+'/label', cv2.IMREAD_COLOR)
test_imgs = load_imgs(test_dir+'/image')
test_masks = load_imgs(test_dir+'/label', cv2.IMREAD_COLOR)

print('# train_imgs: ', len(train_imgs))
print('# train_masks:', len(train_masks))
print('# valid_imgs: ', len(valid_imgs))
print('# valid_masks:', len(valid_masks))
print('# test_imgs:  ', len(test_imgs))
print('# test_masks: ', len(test_masks))

aug = augmenter(batch_size, IMG_SIZE)
#learning_rate = 0.1e-10# 1.0e-9: best -> now, lower(slower). 
#learning_rate = 100 # for jaccard loss
#decay = 0.1
#0 ~ 10000: 1.0e-7 later: 1.0e-8
learning_rate = 1.0 # for Adadelta

my_gen = gen(train_imgs, train_masks, batch_size, num_classes=4)
valid_gen = gen(valid_imgs, valid_masks, batch_size, num_classes=4)
test_gen = gen(test_imgs, test_masks, batch_size, num_classes=4)

''' # for DEBUG
for trs,vl,ts in zip(my_gen,valid_gen,test_gen):
    print(len(trs))
    print(type(trs))
    print(trs[0].shape)
    for i in range(4):
        cv2.imshow( 'tr i',trs[0][i] )
        cv2.imshow( 'tr m',trs[1][i] )
        cv2.imshow( 'vl i', vl[0][i] )
        cv2.imshow( 'vl m', vl[1][i] )
        cv2.imshow( 'ts i', ts[0][i] )
        cv2.imshow( 'ts m', ts[1][i] )
        cv2.waitKey(0)
'''
ones = np.zeros((num_classes,))
for mask in train_masks:
    rgbk_mask = rgb2rgbk(mask)
    # cv2 bgr order.
    b,g,r,k = np.sum(rgbk_mask, axis=(0,1))
    ones += (b,g,r,k)
    #cv2.imshow('mask',rgbk_mask)
    #print(ones, b,g,r,k); cv2.waitKey(0)
'''
for mask in valid_masks:
    ones += np.sum(rgb2rgbk(mask), axis=(0,1))
for mask in test_masks:
    ones += np.sum(rgb2rgbk(mask), axis=(0,1))
'''
sum_ones = np.sum(ones)
weights = sum_ones / ones
weights /= np.sum(weights)
#print('---->',ones)
#print('---->',ones / num_ones)
print('blue        green      red        black')
print(weights)
    
#--------------------------------------------------------------------
    
#---------------------------- train model ---------------------------
#loaded_model = save_model_path ## NOTE
loaded_model = None
model = unet(pretrained_weights=loaded_model, num_out_channels=num_classes, #NOTE: rgbk output!
             input_size=(IMG_SIZE,IMG_SIZE,1),weights=weights,
             lr=learning_rate)#, decay=decay#, weight_0=weight_0, weight_1=weight_1) 
model_checkpoint = ModelCheckpoint(save_model_path, 
                                   #monitor='val_loss',
                                   monitor='loss',
                                   verbose=1, save_best_only=True)
history = model.fit_generator(my_gen, steps_per_epoch=steps_per_epoch, epochs=num_epochs, ## NOTE
                              validation_data=valid_gen, validation_steps=3,#)
                              callbacks=[model_checkpoint,
                                         TensorBoard(batch_size=batch_size,
                                                     write_graph=False)])
#--------------------------------------------------------------------

#--------------------------- save results ---------------------------
#origin_dir = output_dir + '/origin'
#answer_dir = output_dir + '/answer'
origin_dir = output_dir + '/image' # train set
answer_dir = output_dir + '/label' # train set
result_dir = output_dir + '/result'

origins = load_imgs(origin_dir)
answers = load_imgs(answer_dir, cv2.IMREAD_COLOR)
assert len(origins) == len(answers)

num_imgs = len(origins)
aug_det = augmenter(num_imgs,IMG_SIZE).to_deterministic()

origins = aug_det.augment_images(origins)
answers = aug_det.augment_images(answers)
predictions = model.predict_generator((img.reshape(1,IMG_SIZE,IMG_SIZE,1) for img in origins), 
                                      num_imgs, verbose=1)
print(predictions.shape)
predictions = rgbk2rgb(predictions)
print(predictions.shape)

for idx,(org,ans,pred) in enumerate(zip(origins,answers,predictions)):
    cv2.imwrite(os.path.join(result_dir,"%d.png"%idx),
                (org * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(result_dir,"%dans.png"%idx),
                (ans * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(result_dir,"%dpred.png"%idx),
                (pred * 255).astype(np.uint8))

test_metrics = model.evaluate_generator(test_gen, steps=3)
print(model.metrics_names)
print(test_metrics)
#--------------------------------------------------------------------

#-------------------- visualize loss & accuracy ---------------------
import yaml
with open(history_path,'w') as f:
    f.write(yaml.dump(dict(loss = list(map(np.asscalar,history.history['loss'])),
                            acc = list(map(np.asscalar,history.history['mean_iou'])),
                       val_loss = list(map(np.asscalar,history.history['val_loss'])),
                        val_acc = list(map(np.asscalar,history.history['val_mean_iou'])),
                      test_loss = np.asscalar(test_metrics[0]),
                       test_acc = np.asscalar(test_metrics[1]) )))

import matplotlib.pyplot as plt
plt.clf()
plt.plot(history.history['loss'], 'b', label='train loss')
plt.plot(history.history['val_loss'], 'r', label='valid loss')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.legend(fontsize=10)
plt.draw()
plt.show()

plt.clf()
plt.plot(history.history['mean_iou'], 'b', label='train accuracy')
plt.plot(history.history['val_mean_iou'], 'r', label='valid accuracy')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.legend(fontsize=10)
plt.draw()
plt.show()
#--------------------------------------------------------------------
