import re
import os

import cv2
import yaml
import numpy as np
#import skimage.io as io
from tqdm import tqdm
from sklearn.metrics import confusion_matrix 

import model
from utils import bgr_float32, load_imgs
from utils import assert_exists, assert_not_exists

def iou(y_true,y_pred,thr=0.5):
    y_true = (y_true.flatten() >= thr).astype(np.uint8)
    y_pred = (y_pred.flatten() >= thr).astype(np.uint8)
    cnfmat = confusion_matrix(y_true, y_pred, labels=[0, 1])
    intersection = np.diag(cnfmat)
    prediction = cnfmat.sum(axis=0) # 
    ground_truth = cnfmat.sum(axis=1)
    union = ground_truth + prediction - intersection
    return intersection / union.astype(np.float32)

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

def save_eval_summary(eval_summary_path,
                      train_iou_arr, valid_iou_arr, test_iou_arr):
    with open(eval_summary_path,'w') as f:
        train_mean_iou = np.asscalar(np.mean(train_iou_arr))
        valid_mean_iou = np.asscalar(np.mean(valid_iou_arr))
        test_mean_iou = np.asscalar(np.mean(test_iou_arr))
        f.write(yaml.dump(dict( 
            train_iou_arr = train_iou_arr,
            train_mean_iou = train_mean_iou,
            valid_iou_arr = valid_iou_arr,
            valid_mean_iou = valid_mean_iou,
            test_iou_arr = test_iou_arr,
            test_mean_iou = test_mean_iou
        )))#,
        print('------------ Mean IoUs ------------')
        print('train mean iou =',train_mean_iou)
        print('valid mean iou =',valid_mean_iou)
        print(' test mean iou =', test_mean_iou)
        print('-----------------------------------')

def save_img_tuples(result_tuples, result_dir):
    for idx,(org,ans,pred) in enumerate(result_tuples):
        cv2.imwrite(os.path.join(result_dir, '%d.png' % idx),
                    (org * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(result_dir, '%dans.png' % idx),
                    (ans * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(result_dir, '%dpred.png' % idx),
                    (pred * 255).astype(np.uint8))

def make_eval_directory(eval_dirpath, eval_summary_name='summary.yml',
                        train_dir='train',valid_dir='valid',test_dir='test'):
    ''' TODO: input = dictionary that express directory structure. return paths. '''
    assert_not_exists(eval_dirpath)
    eval_summary_path = os.path.join(eval_dirpath, eval_summary_name)
    eval_train_dirpath = os.path.join(eval_dirpath, train_dir)
    eval_valid_dirpath = os.path.join(eval_dirpath, valid_dir)
    eval_test_dirpath = os.path.join(eval_dirpath, test_dir)
    os.makedirs(eval_train_dirpath, exist_ok=True)
    os.makedirs(eval_valid_dirpath, exist_ok=True)
    os.makedirs(eval_test_dirpath, exist_ok=True)
    return eval_summary_path, eval_train_dirpath, eval_valid_dirpath, eval_test_dirpath

def eval_and_save_result(dataset_dir, model_path, eval_result_dirpath,
                         eval_summary_name='summary.yml'):
    #---- load ----
    train_dir = os.path.join(dataset_dir,'train')
    valid_dir = os.path.join(dataset_dir,'valid')
    test_dir = os.path.join(dataset_dir,'test')
    train_img_dir = os.path.join(train_dir,'image')
    train_label_dir = os.path.join(train_dir,'label')
    valid_img_dir = os.path.join(valid_dir,'image')
    valid_label_dir = os.path.join(valid_dir,'label')
    test_img_dir = os.path.join(test_dir, 'image')
    test_label_dir = os.path.join(test_dir, 'label')
    assert_exists(train_img_dir)
    assert_exists(train_label_dir)
    assert_exists(valid_img_dir)
    assert_exists(valid_label_dir)
    assert_exists(test_img_dir)
    assert_exists(test_label_dir)
    train_inputs = list(load_imgs(train_img_dir))
    train_answers = list(load_imgs(train_label_dir))
    valid_inputs = list(load_imgs(valid_img_dir))
    valid_answers = list(load_imgs(valid_label_dir))
    test_inputs = list(load_imgs(test_img_dir))
    test_answers = list(load_imgs(test_label_dir))

    #---- eval ----
    segnet = model.unet(model_path, (None,None,1)) # img h,w must be x16(multiple of 16)
    train_iou_arr, train_result_tuples = evaluate(segnet, train_inputs, train_answers)
    valid_iou_arr, valid_result_tuples = evaluate(segnet, valid_inputs, valid_answers)
    test_iou_arr, test_result_tuples = evaluate(segnet, test_inputs, test_answers)
    print('Evaluation completed!')

    #---- save ----
    summary_path, train_path, valid_path, test_path = make_eval_directory(eval_result_dirpath,
                                                                          eval_summary_name)
    save_eval_summary(summary_path, train_iou_arr, valid_iou_arr, test_iou_arr)
    print('Evaluation summary is saved!')

    save_img_tuples(train_result_tuples, train_path)
    save_img_tuples(valid_result_tuples, valid_path)
    save_img_tuples(test_result_tuples, test_path)
    print('Evaluation result images are saved!')

if __name__ == '__main__':
    #dataset_dir = 'data'
    #dataset_dir = 'data/Malignant_91sep/'
    dataset_dir = 'data/Benigh_74sep/'

    #model_path = './malignant.h5'
    model_path = './benigh.h5'
    #model_path = './seg_data.h5'

    eval_result_dirpath = 'data/Benigh_74sep/eval_results'
    eval_and_save_result(dataset_dir, model_path, 
                         os.path.join(eval_result_dirpath,'eval10'),
                         'eval_summary.yml')
