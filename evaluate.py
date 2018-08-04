import cv2
import numpy as np
import skimage.io as io
from sklearn.metrics import confusion_matrix 
import re
import model
import utils

def human_sorted(iterable):
    ''' Sorts the given iterable in the way that is expected. '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(iterable, key = alphanum_key)

def preprocess(img):
    h,w = img.shape[:2]
    img = (img / 255).astype(np.float32)
    return img.reshape((h,w,1))

def load_imgs(img_dir): 
    return list(map(lambda path: preprocess(cv2.imread(path, 0)),
                    human_sorted(utils.file_paths(img_dir))))

def iou(a,p,thr=0.5):
    a = (a.flatten() >= thr).astype(np.uint8)
    p = (p.flatten() >= thr).astype(np.uint8)
    cnfmat = confusion_matrix(a, p, labels=[0, 1])
    I = np.diag(cnfmat)
    P = cnfmat.sum(axis=0)
    GT = cnfmat.sum(axis=1)
    U = GT + P - I
    #print('U',U)
    #print(cnfmat)
    #print('I',I)
    #print('P',P)
    #print('GT',GT)
    return I / U.astype(np.float32)

def modulo_padded(img, modulo=16):
    h,w = img.shape[:2]
    h_padding = modulo - (h % modulo)
    w_padding = modulo - (w % modulo)
    print(':', h,h_padding, w,w_padding)
    if len(img.shape) == 3:
        return np.pad(img, [(0,h_padding),(0,w_padding),(0,0)], mode='reflect')
    elif len(img.shape) == 2:
        return np.pad(img, [(0,h_padding),(0,w_padding)], mode='reflect')

output_dir = 'data/Benigh_74sep/tmp/'
origin_dir = output_dir + '/origin'
answer_dir = output_dir + '/answer'
result_dir = output_dir + '/result'

# load image,answer
origins = load_imgs(origin_dir)
answers = load_imgs(answer_dir)
iou_arr = []
for origin, answer in zip(origins,answers):
    org_h,org_w = origin.shape[:2]

    # pad and reshape image as unet input 
    img = modulo_padded(origin)
    #print('i',img.shape)
    #print('o',origin.shape)
    img_shape = img.shape #NOTE grayscale!
    img_bat = img.reshape((1,) + img_shape) # size 1 batch

    # get segmentation map
    model_path = 'benigh.h5'
    segnet = model.unet(model_path, (None,None,1)) # img h,w must be x16(multiple of 16)
    segmap = segnet.predict(img_bat, batch_size=1, verbose=1)

    # crop and reshape output
    segmap = segmap[:,:org_h,:org_w,:].reshape((org_h,org_w))
    #print(origin.shape, answer.shape, segmap.shape)
    '''
    io.imshow_collection([origin.reshape((org_h,org_w)), 
                          answer.reshape((org_h,org_w)),  
                          segmap.reshape((org_h,org_w)),]); io.show() # output
                          '''

    # calculate iou
    iou_score = iou(answer,segmap)
    print(iou_score, np.mean(iou_score))
    iou_arr.append(iou_score)
print('total mean IoU:', np.mean(iou_arr))
