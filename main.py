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
from data_gen import augmenter
from utils import bgr_float32, load_imgs, human_sorted
import evaluator

import re

def batch_gen(imgs, masks, batch_size, augmentater):
    assert len(imgs) == len(masks)
    img_flow = cycle(imgs)
    mask_flow = cycle(masks)
    while True:
        aug_det = augmentater.to_deterministic()
        img_batch = aug_det.augment_images( list(islice(img_flow,batch_size)) )
        mask_batch = aug_det.augment_images( list(islice(mask_flow,batch_size)) )
        yield img_batch, mask_batch

def main():
    IMG_SIZE = 256
    batch_size = 4 
    num_epochs = 2#4000
    learning_rate = 1.0 # for Adadelta

    #train_dir = 'data/seg_data/train'
    #valid_dir = 'data/seg_data/valid/'
    #test_dir = 'data/seg_data/test/'
    #output_dir = 'data/seg_data/output/'
    #steps_per_epoch = 8 # 32 = 8step * 4batch
    #save_model_path = 'seg_data.h5' ## NOTE
    #history_path = 'seg_data_history.yml' ## NOTE

    dataset_dirpath = 'data/Benigh_74sep'
    train_dir = os.path.join(dataset_dirpath,'train')
    valid_dir = os.path.join(dataset_dirpath,'valid')
    test_dir = os.path.join(dataset_dirpath,'test')
    output_dir = os.path.join(dataset_dirpath,'output')

    save_model_path = 'benigh_t.h5' ## NOTE
    history_path = 'benigh_history_t.yml' ## NOTE
    steps_per_epoch = 10 # num images: 37 = (10 step) * (4 batch_size)

    #train_dir = 'data/Malignant_91sep/train'
    #valid_dir = 'data/Malignant_91sep/valid'
    #test_dir = 'data/Malignant_91sep/test'
    #output_dir = 'data/Malignant_91sep/output/'
    #save_model_path = 'malignant.h5' ## NOTE
    #history_path = 'malignant_history.yml' ## NOTE
    #steps_per_epoch = 12 # num images: 48 = (12 step) * (4 batch_size)

    '''
    dataset_dir = 'data/Benigh_74/'
    label_str = '_anno'
    all_paths = list(file_paths(dataset_dir))
    img_paths = human_sorted(filter(lambda p: label_str not in p, all_paths))
    mask_paths = human_sorted(filter(lambda p: label_str in p, all_paths))
    img_mask_pairs = list(zip(img_paths, mask_paths))
    for ip, mp in img_mask_pairs:
        print(ip,mp)
        im = io.imread(ip, as_gray=True)
        m = io.imread(mp, as_gray=True)
        io.imshow_collection([im,m]); io.show()
        #io.imshow(m); io.show()
    '''
    train_imgs = list(load_imgs(train_dir+'/image'))
    train_masks = list(load_imgs(train_dir+'/label'))
    valid_imgs = list(load_imgs(valid_dir+'/image'))
    valid_masks = list(load_imgs(valid_dir+'/label'))
    test_imgs = list(load_imgs(test_dir+'/image'))
    test_masks = list(load_imgs(test_dir+'/label'))

    aug = augmenter(batch_size, 256, 1)

    my_gen = batch_gen(train_imgs, train_masks, batch_size, aug)
    valid_gen = batch_gen(valid_imgs, valid_masks, batch_size, aug)
    test_gen = batch_gen(test_imgs, test_masks, batch_size, aug)

    #loaded_model = save_model_path ## NOTE
    loaded_model = None
    if loaded_model:
        model = unet(pretrained_weights=loaded_model,
                     input_size=(IMG_SIZE,IMG_SIZE,1),
                     lr=learning_rate)#, decay=decay#, weight_0=weight_0, weight_1=weight_1) 
    else:
        model = unet(input_size=(IMG_SIZE,IMG_SIZE,1), 
                     lr=learning_rate)#, decay=decay)#, weight_0=weight_0, weight_1=weight_1)

    model_checkpoint = ModelCheckpoint(save_model_path, monitor='val_loss',
                                        verbose=1, save_best_only=True)
    history = model.fit_generator(my_gen, steps_per_epoch=steps_per_epoch, epochs=num_epochs, ## NOTE
                                  validation_data=valid_gen, validation_steps=3,#)
                                  callbacks=[model_checkpoint])
    #--------------------------------------------------------------------

    #--------------------------- save results ---------------------------
    origin_dir = output_dir + '/image'
    answer_dir = output_dir + '/label'
    result_dir = output_dir + '/result'

    origins = list(load_imgs(origin_dir))
    answers = list(load_imgs(answer_dir))
    assert len(origins) == len(answers)

    num_imgs = len(origins)
    aug_det = augmenter(num_imgs,256,1).to_deterministic()

    origins = aug_det.augment_images(origins)
    answers = aug_det.augment_images(answers)
    predictions = model.predict_generator((img.reshape(1,256,256,1) for img in origins), 
                                          num_imgs, verbose=1)

    evaluator.save_img_tuples(zip(origins,answers,predictions),result_dir)

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

if __name__ == '__main__':
    main()
