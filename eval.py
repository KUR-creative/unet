import re
import os

import cv2
import yaml
import numpy as np
#import skimage.io as io
from tqdm import tqdm
from sklearn.metrics import confusion_matrix 

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
    return I / U.astype(np.float32)

def modulo_padded(img, modulo=16):
    h,w = img.shape[:2]
    h_padding = (modulo - (h % modulo)) % modulo
    w_padding = (modulo - (w % modulo)) % modulo
    if len(img.shape) == 3:
        return np.pad(img, [(0,h_padding),(0,w_padding),(0,0)], mode='reflect')
    elif len(img.shape) == 2:
        return np.pad(img, [(0,h_padding),(0,w_padding)], mode='reflect')

def evaluate(segnet, inputs, answers):
    result_tuples = []
    iou_arr = []
    for inp, answer in tqdm( zip(inputs,answers), total=len(inputs) ):
        org_h,org_w = inp.shape[:2]

        img = modulo_padded(inp)
        img_shape = img.shape #NOTE grayscale!
        img_bat = img.reshape((1,) + img_shape) # size 1 batch

        segmap = segnet.predict(img_bat, batch_size=1)#, verbose=1)
        segmap = segmap[:,:org_h,:org_w,:].reshape((org_h,org_w))

        result_tuples.append( (inp.reshape([org_h,org_w]), 
                               answer.reshape([org_h,org_w]),  
                               segmap.reshape([org_h,org_w])) )
        '''
        io.imshow_collection([inp.reshape((org_h,org_w)), 
                              answer.reshape((org_h,org_w)),  
                              segmap.reshape((org_h,org_w)),]); io.show() # output
                              '''
        iou_score = iou(answer,segmap)
        iou_arr.append( np.asscalar(np.mean(iou_score)) )
    return iou_arr, result_tuples

def save_eval_results(result_tuples, result_dir):
    for idx,(org,ans,pred) in enumerate(result_tuples):
        cv2.imwrite(os.path.join(result_dir, '%d.png' % idx),
                    (org * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(result_dir, '%dans.png' % idx),
                    (ans * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(result_dir, '%dpred.png' % idx),
                    (pred * 255).astype(np.uint8))


if __name__ == '__main__':
    ## set source directories
    #dataset_dir = 'data/Benigh_74sep/'
    dataset_dir = 'data/Malignant_91sep/'
    train_dir = os.path.join(dataset_dir,'train')
    valid_dir = os.path.join(dataset_dir,'valid')
    test_dir = os.path.join(dataset_dir,'test')

    #eval_summary = 'data/Benigh_74sep/benigh_eval_summary.yml'
    eval_summary = 'data/Malignant_91sep/eval_summary.yml'
    #eval_summary = 'data/seg_data/eval_summary.yml'

    model_path = './malignant.h5'
    #model_path = './seg_data.h5'

    ## load images
    train_inputs = load_imgs(os.path.join(train_dir,'image'))
    train_answers = load_imgs(os.path.join(train_dir,'label'))
    valid_inputs = load_imgs(os.path.join(valid_dir,'image'))
    valid_answers = load_imgs(os.path.join(valid_dir,'label'))
    test_inputs = load_imgs(os.path.join(test_dir,'image'))
    test_answers = load_imgs(os.path.join(test_dir,'label'))

    ## ready to save results
    eval_result_dir = os.path.join(dataset_dir,'evaluation') # experiment result directory...
    eval_train_dir = os.path.join(eval_result_dir,'train')
    eval_valid_dir = os.path.join(eval_result_dir,'valid')
    eval_test_dir = os.path.join(eval_result_dir,'test')
    os.makedirs(eval_train_dir, exist_ok=True)
    os.makedirs(eval_valid_dir, exist_ok=True)
    os.makedirs(eval_test_dir, exist_ok=True)

    segnet = model.unet(model_path, (None,None,1)) # img h,w must be x16(multiple of 16)

    train_iou_arr, train_result_tuples = evaluate(segnet, train_inputs, train_answers)
    valid_iou_arr, valid_result_tuples = evaluate(segnet, valid_inputs, valid_answers)
    test_iou_arr, test_result_tuples = evaluate(segnet, test_inputs, test_answers)
    print('Evaluation completed!')

    with open(eval_summary,'w') as f:
        f.write(yaml.dump(dict( 
            train_iou_arr = train_iou_arr,
            train_mean_iou = np.asscalar(np.mean(train_iou_arr)),
            valid_iou_arr = valid_iou_arr,
            valid_mean_iou = np.asscalar(np.mean(valid_iou_arr)),
            test_iou_arr = test_iou_arr,
            test_mean_iou = np.asscalar(np.mean(test_iou_arr))
        )))#,
    print('Evaluation summary is saved!')

    save_eval_results(train_result_tuples, eval_train_dir)
    save_eval_results(valid_result_tuples, eval_valid_dir)
    save_eval_results(test_result_tuples, eval_test_dir)
    print('Evaluation result images are saved!')
