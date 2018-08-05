import cv2
import numpy as np
import skimage.io as io
from sklearn.metrics import confusion_matrix 
import re
import model
import utils
import os
from tqdm import tqdm
import yaml

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
    h_padding = (modulo - (h % modulo)) % modulo
    w_padding = (modulo - (w % modulo)) % modulo
    #print(':', h,h_padding, w,w_padding)
    if len(img.shape) == 3:
        return np.pad(img, [(0,h_padding),(0,w_padding),(0,0)], mode='reflect')
    elif len(img.shape) == 2:
        return np.pad(img, [(0,h_padding),(0,w_padding)], mode='reflect')

def evaluate(segnet, inputs, answers):
    #print(len(inputs))
    result_tuples = []
    iou_arr = []
    for inp, answer in tqdm( zip(inputs,answers), total=len(inputs) ):
        org_h,org_w = inp.shape[:2]

        # pad and reshape image as unet inp 
        img = modulo_padded(inp)
        #print('i',img.shape)
        #print('o',inp.shape)
        img_shape = img.shape #NOTE grayscale!
        img_bat = img.reshape((1,) + img_shape) # size 1 batch

        # get segmentation map
        segmap = segnet.predict(img_bat, batch_size=1)#, verbose=1)

        # crop and reshape output
        segmap = segmap[:,:org_h,:org_w,:].reshape((org_h,org_w))
        #print(inp.shape, answer.shape, segmap.shape)
        result_tuples.append( (inp.reshape([org_h,org_w]), 
                               answer.reshape([org_h,org_w]),  
                               segmap.reshape([org_h,org_w])) )
        '''
        io.imshow_collection([inp.reshape((org_h,org_w)), 
                              answer.reshape((org_h,org_w)),  
                              segmap.reshape((org_h,org_w)),]); io.show() # output
                              '''
        # calculate iou
        iou_score = iou(answer,segmap)
        #print(iou_score, np.mean(iou_score))
        iou_arr.append( np.asscalar(np.mean(iou_score)) )
    return iou_arr, result_tuples

#def save_eval_results(result_tuples, result_dir):

dataset_dir = 'data/Benigh_74sep/'
train_dir = os.path.join(dataset_dir,'train')
valid_dir = os.path.join(dataset_dir,'valid')
test_dir = os.path.join(dataset_dir,'test')

train_inputs = load_imgs(os.path.join(train_dir,'image'))
train_answers = load_imgs(os.path.join(train_dir,'label'))
valid_inputs = load_imgs(os.path.join(valid_dir,'image'))
valid_answers = load_imgs(os.path.join(valid_dir,'label'))
test_inputs = load_imgs(os.path.join(test_dir,'image'))
test_answers = load_imgs(os.path.join(test_dir,'label'))

eval_result_dir = os.path.join(dataset_dir,'evaluation')
eval_train_dir = os.path.join(eval_result_dir,'train')
eval_valid_dir = os.path.join(eval_result_dir,'valid')
eval_test_dir = os.path.join(eval_result_dir,'test')
os.makedirs(eval_train_dir, exist_ok=True)
os.makedirs(eval_valid_dir, exist_ok=True)
os.makedirs(eval_test_dir, exist_ok=True)

eval_summary = './data/Benigh_74sep/benigh_eval_summary.yml'
#eval_summary = 'data/seg_data/eval_summary.yml'
model_path = './benigh.h5'
#model_path = './seg_data.h5'
segnet = model.unet(model_path, (None,None,1)) # img h,w must be x16(multiple of 16)

train_iou_arr, train_result_tuples = evaluate(segnet, train_inputs, train_answers)
valid_iou_arr, valid_result_tuples = evaluate(segnet, valid_inputs, valid_answers)
test_iou_arr, test_result_tuples = evaluate(segnet, test_inputs, test_answers)

with open(eval_summary,'w') as f:
    f.write(yaml.dump(dict( 
        train_iou_arr = train_iou_arr,
        train_mean_iou = np.asscalar(np.mean(train_iou_arr)),
        valid_iou_arr = valid_iou_arr,
        valid_mean_iou = np.asscalar(np.mean(valid_iou_arr)),
        test_iou_arr = test_iou_arr,
        test_mean_iou = np.asscalar(np.mean(test_iou_arr))
    )))#,
                           #valid_iou_arr = valid_iou_arr,
                           #test_iou_arr = test_iou_arr)))

for idx,(org,ans,pred) in enumerate(train_result_tuples):
    cv2.imwrite(os.path.join(eval_train_dir,"%d.png"%idx),
                (org * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(eval_train_dir,"%dans.png"%idx),
                (ans * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(eval_train_dir,"%dpred.png"%idx),
                (pred * 255).astype(np.uint8))

for idx,(org,ans,pred) in enumerate(valid_result_tuples):
    cv2.imwrite(os.path.join(eval_valid_dir,"%d.png"%idx),
                (org * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(eval_valid_dir,"%dans.png"%idx),
                (ans * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(eval_valid_dir,"%dpred.png"%idx),
                (pred * 255).astype(np.uint8))

for idx,(org,ans,pred) in enumerate(test_result_tuples):
    cv2.imwrite(os.path.join(eval_test_dir,"%d.png"%idx),
                (org * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(eval_test_dir,"%dans.png"%idx),
                (ans * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(eval_test_dir,"%dpred.png"%idx),
                (pred * 255).astype(np.uint8))
