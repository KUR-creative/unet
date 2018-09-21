from model import *
from data import *
import yaml
import os, sys
import numpy as np
import skimage.io as io
import skimage.transform as trans
import cv2
from skimage.viewer import ImageViewer
from itertools import cycle, islice
from data_gen import augmenter, bgr2bgrk, bgrk2bgr
from imgaug import augmenters as iaa
from utils import file_paths, bgr_float32, load_imgs, human_sorted, ElapsedTimer
import evaluator
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

def bgrk_weights(masks):
    n_all, n_bgrk = 0, [0,0,0,0]
    for mask in masks:
        bgrk_mask = bgr2bgrk(mask)
        class_map = np.argmax(bgrk_mask, axis=-1)
        classes,counts = np.unique(class_map,return_counts=True)
        for color,count in zip(classes,counts):
            n_bgrk[color] += count
        n_all += sum(n_bgrk)
    w_b = n_all / n_bgrk[0]
    w_g = 0 # no green!
    w_r = n_all / n_bgrk[2]
    w_k = n_all / n_bgrk[3]
    #print(n_bgrk, w_b,w_g,w_r,w_k)
    return w_b,w_g,w_r,w_k

def weight01(imgs):
    num_all,num0,num1 = 0,0,0
    for img in imgs:
        h,w = img.shape[:2]
        num_all += h*w
        num1 += np.count_nonzero(img)
        num0 += (num_all - num1)
    weight0 = num_all / num0
    weight1 = num_all / num1 
    return weight0, weight1

def batch_gen(imgs, masks, batch_size, 
              both_aug=None, img_aug=None, mask_aug=None,
              num_classes=1):
    assert len(imgs) == len(masks)
    img_flow = cycle(imgs)
    mask_flow = cycle(masks)
    while True:
        img_batch = list(islice(img_flow,batch_size))
        mask_batch = list(islice(mask_flow,batch_size))

        #print(img_batch)
        if both_aug:
            aug_det = both_aug.to_deterministic()
            img_batch = aug_det.augment_images(img_batch)
            mask_batch = aug_det.augment_images(mask_batch)
        else: # no random crop - use square crop dataset
            img_batch = np.array(img_batch, np.float32)
            mask_batch = np.array(mask_batch, np.float32)

        if img_aug:
            img_batch = img_aug.augment_images(img_batch)
        if mask_aug:
            mask_batch = mask_aug.augment_images(mask_batch)

        if num_classes == 4:
            mask_batch = bgr2bgrk(mask_batch)

        yield img_batch, mask_batch

def modulo_ceil(x, mod):
    ''' return multiple of 'mod' greater than x '''
    return x + (mod - (x % mod)) % mod

def main(experiment_yml_path):
    with open(experiment_yml_path,'r') as f:
        print(experiment_yml_path)
        config = yaml.load(f)
    experiment_name,_ = os.path.splitext(os.path.basename(experiment_yml_path))
    print('->',experiment_name)
    for k,v in config.items():
        print(k,'=',v)
    #----------------------- experiment config ------------------------
    IMG_SIZE = config['IMG_SIZE']
    BATCH_SIZE = config['BATCH_SIZE']
    NUM_EPOCHS = config['NUM_EPOCHS']

    data_augmentation = config['data_augmentation'] # string

    dataset_dir = config['dataset_dir']
    save_model_path = config['save_model_path']## NOTE
    history_path = config['history_path']## NOTE

    eval_result_dirpath = os.path.join(config['eval_result_parent_dir'], 
                                       experiment_name)
    # optional config
    kernel_init = config.get('kernel_init')
    num_maxpool = config.get('num_maxpool')
    num_filters = config.get('num_filters')
    num_sample = config.get('num_sample')
    filter_vec = config.get('filter_vec')
    sqr_crop_dataset = config.get('sqr_crop_dataset')
    num_classes = config.get('num_classes')

    loss = config.get('loss')
    optimizer = config.get('optimizer')
    learning_rate = config.get('learning_rate')

    loaded_model = None
    #--------------------------------------------------------------------

    #--------------------------------------------------------------------
    train_dir = os.path.join(dataset_dir,'train')
    valid_dir = os.path.join(dataset_dir,'valid')
    test_dir = os.path.join(dataset_dir,'test')

    output_dir = os.path.join(dataset_dir,'output')
    origin_dir = os.path.join(output_dir,'image')
    answer_dir = os.path.join(output_dir,'label')
    result_dir = os.path.join(output_dir,'result')
    #--------------------------------------------------------------------

    #-------------------- ready to generate batch -----------------------
    mask_type = cv2.IMREAD_GRAYSCALE if num_classes == 1 else cv2.IMREAD_COLOR
    train_imgs = list(load_imgs(os.path.join(train_dir,'image')))
    train_masks =list(load_imgs(os.path.join(train_dir,'label'), mask_type))
    valid_imgs = list(load_imgs(os.path.join(valid_dir,'image')))
    valid_masks =list(load_imgs(os.path.join(valid_dir,'label'), mask_type))
    test_imgs =  list(load_imgs(os.path.join(test_dir, 'image')))
    test_masks = list(load_imgs(os.path.join(test_dir, 'label'), mask_type))

    w0,w1 = weight01(train_masks + valid_masks + test_masks)
    print('weight0 = %f, weight1 = %f' % (w0,w1))
    weights = bgrk_weights(train_masks + valid_masks + test_masks)
    print('b, g, r, k',weights)

    if num_sample is None: num_sample = 4 #2
    #calc mean h,w of dataset
    tr_h, tr_w = sum(map(lambda img: np.array(img.shape[:2]),train_imgs)) / len(train_imgs)
    vl_h, vl_w = sum(map(lambda img: np.array(img.shape[:2]),valid_imgs)) / len(valid_imgs)
    te_h, te_w = sum(map(lambda img: np.array(img.shape[:2]),test_imgs))  / len(test_imgs)
    #print(tr_h,tr_w, '|', vl_h,vl_w, '|', te_h,te_w)
    train_num_sample = num_sample #1 # int((tr_h/IMG_SIZE) * (tr_w/IMG_SIZE) * overlap_factor)
    valid_num_sample = num_sample #1 # int((vl_h/IMG_SIZE) * (vl_w/IMG_SIZE) * overlap_factor)
    test_num_sample  = num_sample #1 # int((te_h/IMG_SIZE) * (te_w/IMG_SIZE) * overlap_factor)
    #print(train_num_sample,valid_num_sample,test_num_sample)
    train_steps_per_epoch = modulo_ceil(len(train_imgs),BATCH_SIZE) // BATCH_SIZE * train_num_sample
    valid_steps_per_epoch = modulo_ceil(len(valid_imgs),BATCH_SIZE) // BATCH_SIZE * valid_num_sample
    test_steps_per_epoch  = modulo_ceil(len(test_imgs),BATCH_SIZE)  // BATCH_SIZE * test_num_sample
    print('# train images =', len(train_imgs), '| train steps/epoch =', train_steps_per_epoch)
    print('# valid images =', len(valid_imgs), '| valid steps/epoch =', valid_steps_per_epoch)
    print('#  test images =', len(test_imgs),  '|  test steps/epoch =', test_steps_per_epoch)

    aug,img_aug,mask_aug = None,None,None
    if data_augmentation == 'bioseg':
        aug = augmenter(BATCH_SIZE, IMG_SIZE, 1, 
                crop_before_augs=[
                  iaa.Fliplr(0.5),
                  iaa.Flipud(0.5),
                  iaa.Affine(rotate=(-180,180),mode='reflect'),
                ],
                crop_after_augs=[
                  iaa.ElasticTransformation(alpha=(100,200),sigma=14,mode='reflect'),
                ]
              )
    elif data_augmentation == 'manga':
        aug = augmenter(BATCH_SIZE, IMG_SIZE, 1, 
                crop_before_augs=[
                  iaa.Affine(
                    rotate=(-3,3), shear=(-3,3), 
                    scale={'x':(0.8,1.5), 'y':(0.8,1.5)},
                    mode='reflect'),
                ]
              )
    elif data_augmentation == 'manga2':
        aug = augmenter(BATCH_SIZE, IMG_SIZE, 1, 
                crop_before_augs=[
                  iaa.Affine(
                    rotate=(-3,3), shear=(-3,3), 
                    scale={'x':(0.8,1.5), 'y':(0.8,1.5)},
                    mode='reflect'),
                ]
              )
        img_aug = iaa.Sequential([
            iaa.ContrastNormalization((0.998, 1.002)),
            iaa.Sharpen(alpha=(0.0, 0.75), lightness=(0.5, 1.5)),
        ],random_order=True)
    elif data_augmentation == 'too_hard(prev manga2)':
        aug = augmenter(BATCH_SIZE, IMG_SIZE, 1, 
                crop_before_augs=[
                  iaa.Fliplr(0.5),
                  iaa.Flipud(0.5),
                  iaa.Affine(
                    rotate=(-170,170), shear=(-3,3), 
                    scale={'x':(0.8,1.5), 'y':(0.8,1.5)},
                    mode='reflect'),
                ]
              )
        img_aug = iaa.Sequential([
            iaa.ContrastNormalization((0.998, 1.002)),
            iaa.Sharpen(alpha=(0.0, 0.75), lightness=(0.75, 3.0)),
        ],random_order=True)
    elif data_augmentation == 'no_aug':
        aug = augmenter(BATCH_SIZE, IMG_SIZE, 1)

    if sqr_crop_dataset:
        aug = None

    my_gen = batch_gen(train_imgs, train_masks, BATCH_SIZE, aug, img_aug, num_classes=4)
    valid_gen = batch_gen(valid_imgs, valid_masks, BATCH_SIZE, aug, img_aug, num_classes=4)
    test_gen = batch_gen(test_imgs, test_masks, BATCH_SIZE, aug, img_aug, num_classes=4)
    #--------------------------------------------------------------------
    '''
    # DEBUG
    for ims,mas in my_gen:
        for im,ma in zip(ims,mas):
            cv2.imshow('i',im)
            if num_classes == 4:
                print(ma.shape)
                cv2.imshow('m',bgrk2bgr(ma)); cv2.waitKey(0)
            else:
                cv2.imshow('m',ma); cv2.waitKey(0)
    '''
    #---------------------------- train model ---------------------------
    if kernel_init is None: kernel_init = 'he_normal'
    if num_maxpool is None: num_maxpool = 4 
    if num_filters is None: num_filters = 64
    if filter_vec is None: filter_vec = (3,3,1)
    if loss is None: 'jaccard'
    if (num_classes is None) or num_classes == 1:
        last_activation = 'sigmoid'
    else:
        last_activation = 'softmax'

    if optimizer is None: 
        optimizer = 'Adadelta'
        if learning_rate is None: learning_rate = 1.0
    elif optimizer == 'Adam':
        if learning_rate is None: learning_rate = 0.001
        optimizer = Adam(lr=learning_rate)

    print('filter_vec = ', filter_vec)

    print('nc:',num_classes, 'la:',last_activation)
    model = unet(pretrained_weights=loaded_model,
                 input_size=(IMG_SIZE,IMG_SIZE,1),
                 kernel_init=kernel_init, 
                 num_classes=num_classes, last_activation=last_activation,
                 num_filters=num_filters, num_maxpool=num_maxpool, filter_vec=filter_vec,
                 loss=loss, optimizer=optimizer,
                 weight_0=w0, weight_1=w1, weights=weights)
    #model.summary()

    model_checkpoint = ModelCheckpoint(save_model_path, monitor='val_loss',
                                        verbose=1, save_best_only=True)
    train_timer = ElapsedTimer(experiment_yml_path + ' training')
    history = model.fit_generator(
        my_gen, epochs=NUM_EPOCHS,
        steps_per_epoch=train_steps_per_epoch, 
        validation_steps=valid_steps_per_epoch,
        validation_data=valid_gen, 
        callbacks=[model_checkpoint,
                   TensorBoard(log_dir=experiment_name+'_logs',
                               batch_size=BATCH_SIZE, write_graph=False)])
    train_time_str = train_timer.elapsed_time()
    #--------------------------------------------------------------------

    #--------------------------- save results ---------------------------
    origins = list(load_imgs(origin_dir))
    answers = list(load_imgs(answer_dir))
    assert len(origins) == len(answers)

    num_imgs = len(origins)

    if not sqr_crop_dataset:
        aug_det = augmenter(num_imgs,IMG_SIZE,1).to_deterministic() # no augmentation!
        origins = aug_det.augment_images(origins)
        answers = aug_det.augment_images(answers)

    predictions = model.predict_generator((img.reshape(1,IMG_SIZE,IMG_SIZE,1) for img in origins), 
                                          num_imgs, verbose=1)
    evaluator.save_img_tuples(zip(origins,answers, [bgrk2bgr(m) for m in predictions]),result_dir)

    test_metrics = model.evaluate_generator(test_gen, steps=test_steps_per_epoch)
    K.clear_session()
    #print(model.metrics_names)
    #print(test_metrics)
    print('test set: loss =', test_metrics[0], '| IoU =', test_metrics[1])
    #--------------------------------------------------------------------

    #------------------- evaluation and save results --------------------
    with open(history_path,'w') as f:
        f.write(yaml.dump(dict(
            loss = list(map(np.asscalar,history.history['loss'])),
             acc = list(map(np.asscalar,history.history['mean_iou'])),
        val_loss = list(map(np.asscalar,history.history['val_loss'])),
         val_acc = list(map(np.asscalar,history.history['val_mean_iou'])),
       test_loss = np.asscalar(test_metrics[0]),
        test_acc = np.asscalar(test_metrics[1]),
        train_time = train_time_str,
        )))

    modulo = 2**num_maxpool
    evaluator.eval_and_save_result2(dataset_dir, save_model_path, eval_result_dirpath,
                                    files_2b_copied=[history_path, experiment_yml_path],
                                    num_classes=num_classes, last_activation=last_activation,
                                    num_filters=num_filters, num_maxpool=num_maxpool, 
                                    filter_vec=filter_vec,modulo=modulo)
    #--------------------------------------------------------------------

if __name__ == '__main__':
    with open('experiment_log','w') as log:
        for experiment_path in human_sorted(file_paths(sys.argv[1])):
            try:
                timer = ElapsedTimer(experiment_path)
                main(experiment_path)
                log.write(timer.elapsed_time())
            except AssertionError as error:
                print(str(error))
                log.write(str(error))

