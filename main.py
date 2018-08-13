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
from data_gen import augmenter
from imgaug import augmenters as iaa
from utils import file_paths, bgr_float32, load_imgs, human_sorted, ElapsedTimer
import evaluator
from keras import backend as K

def batch_gen(imgs, masks, batch_size, augmentater):
    assert len(imgs) == len(masks)
    img_flow = cycle(imgs)
    mask_flow = cycle(masks)
    while True:
        img_batch = list(islice(img_flow,batch_size))
        mask_batch = list(islice(mask_flow,batch_size))
        if augmentater:
            aug_det = augmentater.to_deterministic()
            img_batch = aug_det.augment_images(img_batch)
            mask_batch = aug_det.augment_images(mask_batch)
        else: # no random crop - use square crop dataset
            img_batch = np.array(img_batch, np.float32)
            mask_batch = np.array(mask_batch, np.float32)
        yield img_batch, mask_batch

def modulo_ceil(x, mod):
    ''' return multiple of 'mod' greater than x '''
    return x + (mod - (x % mod)) % mod

def main(experiment_yml_path):
    with open(experiment_yml_path,'r') as f:
        settings = yaml.load(f)
    experiment_name,_ = os.path.splitext(os.path.basename(experiment_yml_path))
    print('->',experiment_name)
    for k,v in settings.items():
        print(k,'=',v)
    #----------------------- experiment settings ------------------------
    IMG_SIZE = settings['IMG_SIZE']
    BATCH_SIZE = settings['BATCH_SIZE']
    NUM_EPOCHS = settings['NUM_EPOCHS']

    data_augmentation = settings['data_augmentation'] # string

    dataset_dir = settings['dataset_dir']
    save_model_path = settings['save_model_path']## NOTE
    history_path = settings['history_path']## NOTE

    eval_result_dirpath = os.path.join(settings['eval_result_parent_dir'], 
                                       experiment_name)
    # optional settings
    sqr_crop_dataset = settings.get('sqr_crop_dataset') 
    kernel_init = settings.get('kernel_init')
    num_maxpool = settings.get('num_maxpool')
    num_filters = settings.get('num_filters')
    #loaded_model = save_model_path ## NOTE
    loaded_model = None
    #--------------------------------------------------------------------

    #--------------------------------------------------------------------
    train_dir = os.path.join(dataset_dir,'train')
    valid_dir = os.path.join(dataset_dir,'valid')
    test_dir = os.path.join(dataset_dir,'test')

    num_train_imgs = len(os.listdir( os.path.join(train_dir,'image')) )
    num_valid_imgs = len(os.listdir( os.path.join(valid_dir,'image')) )
    TRAIN_STEPS_PER_EPOCH = modulo_ceil(num_train_imgs,BATCH_SIZE) // BATCH_SIZE  # num images: 37 = (10 step) * (4 BATCH_SIZE)
    VALID_STEPS_PER_EPOCH = modulo_ceil(num_valid_imgs,BATCH_SIZE) // BATCH_SIZE  # num images: 19 = (5 step) * (4 BATCH_SIZE)
    print('# train images =', num_train_imgs, '| train steps/epoch =', TRAIN_STEPS_PER_EPOCH)
    print('# valid images =', num_valid_imgs, '| valid steps/epoch =', VALID_STEPS_PER_EPOCH)

    output_dir = os.path.join(dataset_dir,'output')
    origin_dir = os.path.join(output_dir,'image')
    answer_dir = os.path.join(output_dir,'label')
    result_dir = os.path.join(output_dir,'result')
    #--------------------------------------------------------------------

    #-------------------- ready to generate batch -----------------------
    train_imgs = list(load_imgs(os.path.join(train_dir,'image')))
    train_masks =list(load_imgs(os.path.join(train_dir,'label')))
    valid_imgs = list(load_imgs(os.path.join(valid_dir,'image')))
    valid_masks =list(load_imgs(os.path.join(valid_dir,'label')))
    test_imgs =  list(load_imgs(os.path.join(test_dir, 'image')))
    test_masks = list(load_imgs(os.path.join(test_dir, 'label')))

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
    elif data_augmentation == 'manga_gb':
        aug = augmenter(BATCH_SIZE, IMG_SIZE, 1, 
                crop_before_augs=[
                  iaa.Affine(
                    rotate=(-3,3), shear=(-3,3), 
                    scale={'x':(0.8,1.5), 'y':(0.8,1.5)},
                    mode='reflect'),
                ]
              )
    elif data_augmentation == 'no_aug':
        aug = augmenter(BATCH_SIZE, IMG_SIZE, 1)

    if sqr_crop_dataset:
        aug = None

    my_gen = batch_gen(train_imgs, train_masks, BATCH_SIZE, aug)
    valid_gen = batch_gen(valid_imgs, valid_masks, BATCH_SIZE, aug)
    test_gen = batch_gen(test_imgs, test_masks, BATCH_SIZE, aug)
    #--------------------------------------------------------------------
    '''
    '''
    # DEBUG
    for ims,mas in my_gen:
        for im,ma in zip(ims,mas):
            cv2.imshow('i',im)
            cv2.imshow('m',ma); cv2.waitKey(0)
    #---------------------------- train model ---------------------------
    if kernel_init is None: kernel_init = 'he_normal'
    if num_maxpool is None: num_maxpool = 4 
    if num_filters is None: num_filters = 64

    LEARNING_RATE = 1.0
    model = unet(pretrained_weights=loaded_model,
                 input_size=(IMG_SIZE,IMG_SIZE,1),
                 kernel_init=kernel_init,
                 num_filters=num_filters,
                 num_maxpool=num_maxpool,
                 lr=LEARNING_RATE)

    model_checkpoint = ModelCheckpoint(save_model_path, monitor='val_loss',
                                        verbose=1, save_best_only=True)
    train_timer = ElapsedTimer(experiment_yml_path + ' training')
    history = model.fit_generator(my_gen, epochs=NUM_EPOCHS,
                                  steps_per_epoch=TRAIN_STEPS_PER_EPOCH, 
                                  validation_steps=VALID_STEPS_PER_EPOCH,
                                  validation_data=valid_gen, 
                                  callbacks=[model_checkpoint])
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

    evaluator.save_img_tuples(zip(origins,answers,predictions),result_dir)

    test_metrics = model.evaluate_generator(test_gen, steps=3)
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
    evaluator.eval_and_save_result(dataset_dir, save_model_path, eval_result_dirpath,
                                   files_2b_copied=[history_path, experiment_yml_path],
                                   num_filters=num_filters, num_maxpool=num_maxpool, modulo=modulo)
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

