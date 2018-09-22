import re
import os
import shutil

import cv2
import yaml
import numpy as np
#import skimage.io as io
from tqdm import tqdm
from sklearn.metrics import confusion_matrix 
from keras import backend as K

import model
import utils
from utils import bgr_float32, load_imgs
from utils import assert_exists, assert_not_exists
from metric import advanced_metric
from data_gen import bgrk2bgr
import traceback

np.set_printoptions(threshold=np.nan, linewidth=np.nan)

def binarization(img, threshold=100):
    #print(img[100:152,100:160])
    binarized = (img >= threshold).astype(np.uint8) * 255
    #print(binarized[100:152,100:160])
    #cv2.imshow('i',img)
    #cv2.imshow('b',binarized);cv2.waitKey(0)    
    return binarized

def get_segmap(segnet, img_batch, batch_size=1):
    segmap = segnet.predict(img_batch, batch_size)
    if segmap.shape[-1] == 4:
        segmap = bgrk2bgr(segmap)
    return segmap

def iou(y_true,y_pred,thr=0.5):
    y_true = (y_true.flatten() >= thr).astype(np.uint8)
    y_pred = (y_pred.flatten() >= thr).astype(np.uint8)
    cnfmat = confusion_matrix(y_true, y_pred, labels=[0, 1])
    intersection = np.diag(cnfmat)
    prediction = cnfmat.sum(axis=0) # 
    ground_truth = cnfmat.sum(axis=1)
    union = ground_truth + prediction - intersection
    return ((intersection + 0.001) / (union.astype(np.float32) + 0.001)).tolist()

def modulo_padded(img, modulo=16):
    h,w = img.shape[:2]
    h_padding = (modulo - (h % modulo)) % modulo
    w_padding = (modulo - (w % modulo)) % modulo
    if len(img.shape) == 3:
        return np.pad(img, [(0,h_padding),(0,w_padding),(0,0)], mode='reflect')
    elif len(img.shape) == 2:
        return np.pad(img, [(0,h_padding),(0,w_padding)], mode='reflect')

def segment_or_oom(segnet, inp, modulo=16):
    ''' If image is too big, return None '''
    org_h,org_w = inp.shape[:2]

    img = modulo_padded(inp, modulo) 
    img_shape = img.shape #NOTE grayscale!
    img_bat = img.reshape((1,) + img_shape) # size 1 batch
    try:
        segmap = get_segmap(segnet, img_bat)#segnet.predict(img_bat, batch_size=1)#, verbose=1)
        segmap = segmap[:,:org_h,:org_w,:]#.reshape((org_h,org_w)) # remove padding
        return segmap
    except Exception as e: # ResourceExhaustedError:
        print(traceback.print_exc()); exit()
        print(img_shape,'OOM error: image is too big. (in segnet)')
        return None

size_limit = 4000000 # dev-machine
def segment(segnet, inp, modulo=16):
    ''' oom-free segmentation '''
    global size_limit
    
    h,w = inp.shape[:2]
    result = None
    if h*w < size_limit:
        result = segment_or_oom(segnet, inp, modulo)
        if result is None: # size_limit: Ok but OOM occur!
            size_limit = h*w
            print('segmentation size_limit =', size_limit, 'updated!')
    else:
        print('segmentation size_limit exceed! img_size =', 
              h*w, '>', size_limit, '= size_limit')

    if result is None: # exceed size_limit or OOM
        if h > w:
            upper = segment(segnet, inp[:h//2,:], modulo) 
            downer= segment(segnet, inp[h//2:,:], modulo)
            return np.concatenate((upper,downer), axis=0)
        else:
            left = segment(segnet, inp[:,:w//2], modulo)
            right= segment(segnet, inp[:,w//2:], modulo)
            return np.concatenate((left,right), axis=1)
    else:
        return result # image inpainted successfully!

from tensorflow.errors import ResourceExhaustedError
def evaluate_manga(segnet, inputs, answers, modulo=16):
    result_tuples = []
    iou_arr = []
    for inp, answer in tqdm( zip(inputs,answers), total=len(inputs) ):
        ans_bgr = np.copy(answer)
        # flatten
        if answer.shape[-1] != 1:
            answer = np.sum(answer, axis=-1)
        if inp.shape <= answer.shape:
            org_h,org_w = inp.shape[:2]
        else:
            org_h,org_w = answer.shape[:2]

        img = modulo_padded(inp,modulo)
        #img_shape = img.shape #NOTE grayscale!
        #img_bat = img.reshape((1,) + img_shape) # size 1 batch

        segmap = segment(segnet, img, modulo=modulo) # not batch, just use img!
        segmap_bgr = np.copy(segmap).reshape(segmap.shape[1:])
        segmap_bgr = (segmap_bgr * 255).astype(np.uint8)
        #print(np.unique(segmap_bgr))
        segmap_bgr01 = binarization(np.copy(segmap_bgr))
        #print(np.unique(segmap_bgr01))
        #cv2.imshow('s',segmap_bgr);cv2.waitKey(0)

        #print('org',org_h,org_w)
        if segmap.shape[-1] != 1:
            segmap = np.sum(segmap, axis=-1)
        segmap = segmap.reshape(segmap.shape[1:])
        #print(inp.shape)
        #print(answer.shape)
        #print(segmap.shape)
        inp = inp[:org_h,:org_w].reshape((org_h,org_w))
        answer = answer[:org_h,:org_w].reshape((org_h,org_w))  
        segmap = segmap[:org_h,:org_w].reshape((org_h,org_w))

        result_tuples.append( (inp,ans_bgr,segmap_bgr,segmap_bgr01) )
        iou_score = iou(answer,segmap)
        iou_arr.append( np.asscalar(np.mean(iou_score)) )

    return {'ious':iou_arr}, result_tuples

def evaluate(segnet, inputs, answers, modulo=16):
    result_tuples = []
    iou_arr = []
    f1_score_arr = []
    dice_obj_arr = []
    for inp, answer in tqdm( zip(inputs,answers), total=len(inputs) ):
        org_h,org_w = inp.shape[:2]

        img = modulo_padded(inp,modulo)
        img_shape = img.shape #NOTE grayscale!
        img_bat = img.reshape((1,) + img_shape) # size 1 batch

        try:
            segmap = get_segmap(segnet, img_bat)#segnet.predict(img_bat, batch_size=1)#, verbose=1)
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

            f1_score, dice_obj = advanced_metric(answer,segmap)
            f1_score_arr.append(f1_score)
            dice_obj_arr.append(dice_obj)

        except: #ResourceExhaustedError:
            print(img_shape,'OOM error')
            # Now it just skip very big image, but you can implement OOM free code.
            # For instance, if OOM happen, divide image into 4 pieces, modulo_pad them,
            # predict 4 segmaps, and then merge them. And also save image shape that
            # caused OOM error, that would be used later.
    return {'ious':iou_arr, 'f1s':f1_score_arr, 'dice_objs':dice_obj_arr}, result_tuples
def inference(segnet, inputs, modulo=16):
    outputs = []
    for inp in inputs:
        org_h,org_w = inp.shape[:2]

        img = modulo_padded(inp, modulo)
        img_shape = img.shape #NOTE grayscale!
        img_bat = img.reshape((1,) + img_shape) # size 1 batch
        try:
            segmap = get_segmap(segnet, img_bat)#segnet.predict(img_bat, batch_size=1)#, verbose=1)
            segmap = segmap[:,:org_h,:org_w,:].reshape((org_h,org_w))
            outputs.append(segmap)
        except ResourceExhaustedError:
            print(img_shape,'OOM error: image is too big.')
    return outputs

def save_manga_eval_summary(eval_summary_path,
                      train_metrics, valid_metrics, test_metrics):
    train_iou_arr = train_metrics['ious']
    valid_iou_arr = valid_metrics['ious']
    test_iou_arr = test_metrics['ious']

    with open(eval_summary_path,'w') as f:
        train_mean_iou = float(np.mean(train_iou_arr))
        valid_mean_iou = float(np.mean(valid_iou_arr))
        test_mean_iou  = float(np.mean(test_iou_arr))

        f.write(yaml.dump(dict( 
            train_iou_arr = train_iou_arr,
            valid_iou_arr = valid_iou_arr,
            test_iou_arr  = test_iou_arr,
            train_mean_iou = train_mean_iou,
            valid_mean_iou = valid_mean_iou,
            test_mean_iou  = test_mean_iou,
        )))#,
        print('------------ Mean IoUs ------------')
        print('train mean iou =',train_mean_iou)
        print('valid mean iou =',valid_mean_iou)
        print(' test mean iou =', test_mean_iou)
        print('-----------------------------------')

def save_eval_summary(eval_summary_path,
                      train_metrics, valid_metrics, test_metrics):
    train_iou_arr = train_metrics['ious']
    train_f1_arr = train_metrics['f1s']
    train_dice_arr = train_metrics['dice_objs']

    valid_iou_arr = valid_metrics['ious']
    valid_f1_arr = valid_metrics['f1s']
    valid_dice_arr = valid_metrics['dice_objs']

    test_iou_arr = test_metrics['ious']
    test_f1_arr = test_metrics['f1s']
    test_dice_arr = test_metrics['dice_objs']

    with open(eval_summary_path,'w') as f:
        train_mean_iou      = float(np.mean(train_iou_arr))
        train_mean_f1_score = float(np.mean(train_f1_arr))
        train_mean_dice_obj = float(np.mean(train_dice_arr))

        valid_mean_iou      = float(np.mean(valid_iou_arr))
        valid_mean_f1_score = float(np.mean(valid_f1_arr))
        valid_mean_dice_obj = float(np.mean(valid_dice_arr))

        test_mean_iou      = float(np.mean(test_iou_arr))
        test_mean_f1_score = float(np.mean(test_f1_arr))
        test_mean_dice_obj = float(np.mean(test_dice_arr))

        f.write(yaml.dump(dict( 
            train_iou_arr = train_iou_arr,
            train_f1_arr  = train_f1_arr, 
            train_dice_arr= train_dice_arr,
                                            
            valid_iou_arr = valid_iou_arr,
            valid_f1_arr  = valid_f1_arr,
            valid_dice_arr= valid_dice_arr,
                                            
            test_iou_arr  = test_iou_arr,
            test_f1_arr   = test_f1_arr,
            test_dice_arr = test_dice_arr,

            train_mean_iou      = train_mean_iou,
            train_mean_f1_score = train_mean_f1_score,
            train_mean_dice_obj = train_mean_dice_obj,
                                                       
            valid_mean_iou      = valid_mean_iou,
            valid_mean_f1_score = valid_mean_f1_score,
            valid_mean_dice_obj = valid_mean_dice_obj,
                                                       
            test_mean_iou       = test_mean_iou,
            test_mean_f1_score  = test_mean_f1_score,
            test_mean_dice_obj  = test_mean_dice_obj
        )))#,
        print('------------ Mean IoUs ------------')
        print('train mean iou =',train_mean_iou)
        print('valid mean iou =',valid_mean_iou)
        print(' test mean iou =', test_mean_iou)
        print('-----------------------------------')

def save_img_tuples(result_tuples, result_dir):
        for idx,result_tuple in enumerate(result_tuples):
            org,ans,pred = result_tuple[:3]
            cv2.imwrite(os.path.join(result_dir, '%d.png' % idx),
                        (org * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(result_dir, '%dans.png' % idx),
                        (ans * 255).astype(np.uint8))
            #print('p',np.unique(pred))
            #print('pt',pred.dtype)
            cv2.imwrite(os.path.join(result_dir, '%dpred.png' % idx), pred)

            if len(result_tuple) == 4:
                pred01 = result_tuple[3]
                cv2.imwrite(os.path.join(result_dir, '%dpred01.png' % idx), pred01)
                #print('p',np.unique(pred01))
                #print('pt',pred01.dtype)

def make_eval_directory(eval_dirpath, eval_summary_name='summary.yml',
                        train_dir='train',valid_dir='valid',test_dir='test'):
    ''' TODO: input = dictionary that express directory structure. return paths. '''
    #assert_not_exists(eval_dirpath)
    eval_summary_path = os.path.join(eval_dirpath, eval_summary_name)
    eval_train_dirpath = os.path.join(eval_dirpath, train_dir)
    eval_valid_dirpath = os.path.join(eval_dirpath, valid_dir)
    eval_test_dirpath = os.path.join(eval_dirpath, test_dir)
    os.makedirs(eval_train_dirpath, exist_ok=True)
    os.makedirs(eval_valid_dirpath, exist_ok=True)
    os.makedirs(eval_test_dirpath, exist_ok=True)
    return eval_summary_path, eval_train_dirpath, eval_valid_dirpath, eval_test_dirpath

def eval_and_save_result2(dataset_dir, model_path, eval_result_dirpath,
                         eval_summary_name='eval_summary.yml',
                         files_2b_copied=None,
                         num_filters=64,num_maxpool=4, filter_vec=(3,3,1),
                         num_classes=1, last_activation='sigmoid',
                         modulo=16):
    '''
    for manga!
    '''
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
    mask_type = cv2.IMREAD_GRAYSCALE if num_classes == 1 else cv2.IMREAD_COLOR
    train_inputs = list(load_imgs(train_img_dir))
    train_answers = list(load_imgs(train_label_dir, mask_type))
    valid_inputs = list(load_imgs(valid_img_dir))
    valid_answers = list(load_imgs(valid_label_dir, mask_type))
    test_inputs = list(load_imgs(test_img_dir))
    test_answers = list(load_imgs(test_label_dir, mask_type))

    #---- eval ----
    segnet = model.unet(model_path, (None,None,1), 
                        num_classes=num_classes, last_activation=last_activation,
                        num_filters=num_filters, num_maxpool=num_maxpool,
                        filter_vec=filter_vec)
    train_metrics, train_result_tuples = evaluate_manga(segnet, train_inputs, train_answers, modulo)
    valid_metrics, valid_result_tuples = evaluate_manga(segnet, valid_inputs, valid_answers, modulo)
    test_metrics, test_result_tuples = evaluate_manga(segnet, test_inputs, test_answers, modulo)
    K.clear_session()
    print('Evaluation completed!')

    #---- save ----
    summary_path, train_path, valid_path, test_path = make_eval_directory(eval_result_dirpath,
                                                                          eval_summary_name)
    save_manga_eval_summary(summary_path, train_metrics, valid_metrics, test_metrics)
    print('Evaluation summary is saved!')

    save_img_tuples(train_result_tuples, train_path)
    save_img_tuples(valid_result_tuples, valid_path)
    save_img_tuples(test_result_tuples, test_path)
    print('Evaluation result images are saved!')

    if files_2b_copied is None:
        files_2b_copied = [model_path]
    else:
        files_2b_copied.append(model_path)

    for file_path in files_2b_copied:
        file_name = os.path.basename(file_path)
        shutil.copyfile(file_path, os.path.join(eval_result_dirpath, file_name))
        print("file '%s' is copyed into '%s'" % (file_name,eval_result_dirpath))

def eval_and_save_result(dataset_dir, model_path, eval_result_dirpath,
                         eval_summary_name='eval_summary.yml',
                         files_2b_copied=None,
                         num_filters=64,num_maxpool=4,
                         modulo=16):
    '''
    '''
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
    segnet = model.unet(model_path, (None,None,1), 
                        num_filters=num_filters, num_maxpool=num_maxpool)
    train_metrics, train_result_tuples = evaluate(segnet, train_inputs, train_answers, modulo)
    valid_metrics, valid_result_tuples = evaluate(segnet, valid_inputs, valid_answers, modulo)
    test_metrics, test_result_tuples = evaluate(segnet, test_inputs, test_answers, modulo)
    K.clear_session()
    print('Evaluation completed!')

    #---- save ----
    summary_path, train_path, valid_path, test_path = make_eval_directory(eval_result_dirpath,
                                                                          eval_summary_name)
    save_eval_summary(summary_path, train_metrics, valid_metrics, test_metrics)
    print('Evaluation summary is saved!')

    save_img_tuples(train_result_tuples, train_path)
    save_img_tuples(valid_result_tuples, valid_path)
    save_img_tuples(test_result_tuples, test_path)
    print('Evaluation result images are saved!')

    if files_2b_copied is None:
        files_2b_copied = [model_path]
    else:
        files_2b_copied.append(model_path)

    for file_path in files_2b_copied:
        file_name = os.path.basename(file_path)
        shutil.copyfile(file_path, os.path.join(eval_result_dirpath, file_name))
        print("file '%s' is copyed into '%s'" % (file_name,eval_result_dirpath))

import sys,pathlib
from fp import pipe,cmap,cfilter
if __name__ == '__main__':
    benigh_data = './data/Benigh_74sep/'
    malignant_data = './data/Malignant_91sep/'

    eval_and_save_result(benigh_data, './mixed_models/mixed_cnum32_depth4.h5',        'mixed_benigh/mixed_cnum32_depth4', num_filters=32, num_maxpool=4, modulo=16)
    eval_and_save_result(benigh_data, './mixed_models/mixed_cnum32_depth5.h5',        'mixed_benigh/mixed_cnum32_depth5', num_filters=32, num_maxpool=5, modulo=32)
    eval_and_save_result(benigh_data, './mixed_models/mixed_cnum64_depth4.h5',        'mixed_benigh/mixed_cnum64_depth4', num_filters=64, num_maxpool=4, modulo=16)
    eval_and_save_result(benigh_data, './mixed_models/mixed_cnum64_depth5.h5',        'mixed_benigh/mixed_cnum64_depth5', num_filters=64, num_maxpool=5, modulo=32)

    eval_and_save_result(benigh_data, './mixed_models/mixed_xavier_cnum32_depth4.h5', 'mixed_benigh/mixed_xavier_cnum32_depth4', num_filters=32, num_maxpool=4, modulo=16)
    eval_and_save_result(benigh_data, './mixed_models/mixed_xavier_cnum32_depth5.h5', 'mixed_benigh/mixed_xavier_cnum32_depth5', num_filters=32, num_maxpool=5, modulo=32)
    eval_and_save_result(benigh_data, './mixed_models/mixed_xavier_cnum64_depth4.h5', 'mixed_benigh/mixed_xavier_cnum64_depth4', num_filters=64, num_maxpool=4, modulo=16)
    eval_and_save_result(benigh_data, './mixed_models/mixed_xavier_cnum64_depth5.h5', 'mixed_benigh/mixed_xavier_cnum64_depth5', num_filters=64, num_maxpool=5, modulo=32)

    eval_and_save_result(benigh_data, './mixed_models/mixed_RandomNormal.h5',         'mixed_benigh/mixed_RandomNormal.', num_filters=64, num_maxpool=4, modulo=16)
    eval_and_save_result(benigh_data, './mixed_models/mixed_glorot_normal.h5',        'mixed_benigh/mixed_glorot_normal', num_filters=64, num_maxpool=4, modulo=16)
    eval_and_save_result(benigh_data, './mixed_models/mixed_he_normal.h5',            'mixed_benigh/mixed_he_normal', num_filters=64, num_maxpool=4, modulo=16)


    eval_and_save_result(malignant_data, './mixed_models/mixed_cnum32_depth4.h5',        'mixed_malignant/mixed_cnum32_depth4', num_filters=32, num_maxpool=4, modulo=16)
    eval_and_save_result(malignant_data, './mixed_models/mixed_cnum32_depth5.h5',        'mixed_malignant/mixed_cnum32_depth5', num_filters=32, num_maxpool=5, modulo=32)
    eval_and_save_result(malignant_data, './mixed_models/mixed_cnum64_depth4.h5',        'mixed_malignant/mixed_cnum64_depth4', num_filters=64, num_maxpool=4, modulo=16)
    eval_and_save_result(malignant_data, './mixed_models/mixed_cnum64_depth5.h5',        'mixed_malignant/mixed_cnum64_depth5', num_filters=64, num_maxpool=5, modulo=32)

    eval_and_save_result(malignant_data, './mixed_models/mixed_xavier_cnum32_depth4.h5', 'mixed_malignant/mixed_xavier_cnum32_depth4', num_filters=32, num_maxpool=4, modulo=16)
    eval_and_save_result(malignant_data, './mixed_models/mixed_xavier_cnum32_depth5.h5', 'mixed_malignant/mixed_xavier_cnum32_depth5', num_filters=32, num_maxpool=5, modulo=32)
    eval_and_save_result(malignant_data, './mixed_models/mixed_xavier_cnum64_depth4.h5', 'mixed_malignant/mixed_xavier_cnum64_depth4', num_filters=64, num_maxpool=4, modulo=16)
    eval_and_save_result(malignant_data, './mixed_models/mixed_xavier_cnum64_depth5.h5', 'mixed_malignant/mixed_xavier_cnum64_depth5', num_filters=64, num_maxpool=5, modulo=32)

    eval_and_save_result(malignant_data, './mixed_models/mixed_RandomNormal.h5',         'mixed_malignant/mixed_RandomNormal', num_filters=64, num_maxpool=4, modulo=16)
    eval_and_save_result(malignant_data, './mixed_models/mixed_glorot_normal.h5',        'mixed_malignant/mixed_glorot_normal', num_filters=64, num_maxpool=4, modulo=16)
    eval_and_save_result(malignant_data, './mixed_models/mixed_he_normal.h5',            'mixed_malignant/mixed_he_normal', num_filters=64, num_maxpool=4, modulo=16)
    '''
    benigh_noaug = './data/Benigh_74sep/eval_results/no_aug_benigh'
    benigh_aug = './data/Benigh_74sep/eval_results/aug_benigh'

    # benigh no augmentation
    eval_and_save_result(benigh_74sep, os.path.join(benigh_noaug,'benigh_RandomNormal/benigh.h5'), './data/eval2/noaug/benigh_RandomNormal',  num_filters=64, num_maxpool=4, modulo=16)
    eval_and_save_result(benigh_74sep, os.path.join(benigh_noaug,'benigh_glorot_normal/benigh.h5'),'./data/eval2/noaug/benigh_glorot_normal', num_filters=64, num_maxpool=4, modulo=16)
    eval_and_save_result(benigh_74sep, os.path.join(benigh_noaug,'benigh_he_normal/benigh.h5'),    './data/eval2/noaug/benigh_he_normal',     num_filters=64, num_maxpool=4, modulo=16)

    eval_and_save_result(benigh_74sep, os.path.join(benigh_noaug,'benigh_cnum32_depth4/benigh.h5'),'./data/eval2/noaug/benigh_cnum32_depth4', num_filters=32, num_maxpool=4, modulo=16)
    eval_and_save_result(benigh_74sep, os.path.join(benigh_noaug,'benigh_cnum32_depth5/benigh.h5'),'./data/eval2/noaug/benigh_cnum32_depth5', num_filters=32, num_maxpool=5, modulo=32)
    eval_and_save_result(benigh_74sep, os.path.join(benigh_noaug,'benigh_cnum64_depth4/benigh.h5'),'./data/eval2/noaug/benigh_cnum64_depth4', num_filters=64, num_maxpool=4, modulo=16)
    eval_and_save_result(benigh_74sep, os.path.join(benigh_noaug,'benigh_cnum64_depth5/benigh.h5'),'./data/eval2/noaug/benigh_cnum64_depth5', num_filters=64, num_maxpool=5, modulo=32)

    # benigh augmentation
    eval_and_save_result(benigh_74sep, os.path.join(benigh_aug,'benigh_RandomNormal/benigh.h5'), './data/eval2/aug/benigh_RandomNormal',  num_filters=64, num_maxpool=4, modulo=16)
    eval_and_save_result(benigh_74sep, os.path.join(benigh_aug,'benigh_glorot_normal/benigh.h5'),'./data/eval2/aug/benigh_glorot_normal', num_filters=64, num_maxpool=4, modulo=16)
    eval_and_save_result(benigh_74sep, os.path.join(benigh_aug,'benigh_he_normal/benigh.h5'),    './data/eval2/aug/benigh_he_normal',     num_filters=64, num_maxpool=4, modulo=16)

    eval_and_save_result(benigh_74sep, os.path.join(benigh_aug,'benigh_cnum32_depth4/benigh.h5'),'./data/eval2/aug/benigh_cnum32_depth4', num_filters=32, num_maxpool=4, modulo=16)
    eval_and_save_result(benigh_74sep, os.path.join(benigh_aug,'benigh_cnum32_depth5/benigh.h5'),'./data/eval2/aug/benigh_cnum32_depth5', num_filters=32, num_maxpool=5, modulo=32)
    eval_and_save_result(benigh_74sep, os.path.join(benigh_aug,'benigh_cnum64_depth4/benigh.h5'),'./data/eval2/aug/benigh_cnum64_depth4', num_filters=64, num_maxpool=4, modulo=16)
    eval_and_save_result(benigh_74sep, os.path.join(benigh_aug,'benigh_cnum64_depth5/benigh.h5'),'./data/eval2/aug/benigh_cnum64_depth5', num_filters=64, num_maxpool=5, modulo=32)
    #python evaluator.py segnet.h5 imgs_dir output_dir
    segnet_model_path = sys.argv[1]
    imgs_dir = sys.argv[2]
    output_dir = sys.argv[3]
    utils.safe_copytree(imgs_dir, output_dir,['*.*'])

    segnet = model.unet(segnet_model_path, (None,None,1))

    f = pipe(utils.file_paths, 
             cmap(lambda path: (cv2.imread(path,0), path)),
             cfilter(lambda img_path: img_path[0] is not None),
             cmap(lambda img_path: (utils.bgr_float32(img_path[0]), img_path[1]) ),
             cmap(lambda im_p: (im_p[0].reshape((1,)+im_p[0].shape), im_p[1]) ),
             cmap(lambda im_p: (inference(segnet,im_p[0]), im_p[1])))
    old_parent_dir = pathlib.Path(imgs_dir).parts[-1]
    
    for segmap_list, img_path in f(imgs_dir):
        new_path = utils.make_dstpath(img_path, old_parent_dir, output_dir)
        segmap = segmap_list[0]
        segmap = (segmap.reshape(segmap.shape[:2]) * 255).astype(np.uint8)
        #cv2.imshow('segmap',segmap); cv2.waitKey(0)
        cv2.imwrite(new_path, segmap)
    '''
