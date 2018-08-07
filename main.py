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

def batch_gen(imgs, masks, batch_size, augmentater):
    assert len(imgs) == len(masks)
    img_flow = cycle(imgs)
    mask_flow = cycle(masks)
    while True:
        aug_det = augmentater.to_deterministic()
        img_batch = aug_det.augment_images( list(islice(img_flow,batch_size)) )
        mask_batch = aug_det.augment_images( list(islice(mask_flow,batch_size)) )
        yield img_batch, mask_batch

def modulo_ceil(x, mod):
    ''' return multiple of 'mod' greater than x '''
    return x + (mod - (x % mod)) % mod

def main():
    #settings = 
    #---------------------- experiment setting --------------------------
    IMG_SIZE = 256
    BATCH_SIZE = 4 
    NUM_EPOCHS = 2#4000
    LEARNING_RATE = 1.0 # for Adadelta

    #train_dir = 'data/seg_data/train'
    #valid_dir = 'data/seg_data/valid/'
    #test_dir = 'data/seg_data/test/'
    #output_dir = 'data/seg_data/output/'
    #STEPS_PER_EPOCH = 8 # 32 = 8step * 4batch
    #save_model_path = 'seg_data.h5' ## NOTE
    #history_path = 'seg_data_history.yml' ## NOTE

    dataset_dir = 'data/Benigh_74sep'
    save_model_path = 'benigh_t.h5' ## NOTE
    history_path = 'benigh_history_t.yml' ## NOTE

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

    #train_dir = 'data/Malignant_91sep/train'
    #valid_dir = 'data/Malignant_91sep/valid'
    #test_dir = 'data/Malignant_91sep/test'
    #output_dir = 'data/Malignant_91sep/output/'
    #save_model_path = 'malignant.h5' ## NOTE
    #history_path = 'malignant_history.yml' ## NOTE
    #steps_per_epoch = 12 # num images: 48 = (12 step) * (4 BATCH_SIZE)

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
    #--------------------------------------------------------------------

    #-------------------- ready to generate batch -----------------------
    train_imgs = list(load_imgs(os.path.join(train_dir,'image')))
    train_masks =list(load_imgs(os.path.join(train_dir,'label')))
    valid_imgs = list(load_imgs(os.path.join(valid_dir,'image')))
    valid_masks =list(load_imgs(os.path.join(valid_dir,'label')))
    test_imgs =  list(load_imgs(os.path.join(test_dir, 'image')))
    test_masks = list(load_imgs(os.path.join(test_dir, 'label')))

    aug = augmenter(BATCH_SIZE, IMG_SIZE, 1)

    my_gen = batch_gen(train_imgs, train_masks, BATCH_SIZE, aug)
    valid_gen = batch_gen(valid_imgs, valid_masks, BATCH_SIZE, aug)
    test_gen = batch_gen(test_imgs, test_masks, BATCH_SIZE, aug)
    #--------------------------------------------------------------------

    #---------------------------- train model ---------------------------
    #loaded_model = save_model_path ## NOTE
    loaded_model = None
    model = unet(pretrained_weights=loaded_model,
                 input_size=(IMG_SIZE,IMG_SIZE,1),
                 lr=LEARNING_RATE)

    model_checkpoint = ModelCheckpoint(save_model_path, monitor='val_loss',
                                        verbose=1, save_best_only=True)
    history = model.fit_generator(my_gen, epochs=NUM_EPOCHS,
                                  steps_per_epoch=TRAIN_STEPS_PER_EPOCH, 
                                  validation_steps=VALID_STEPS_PER_EPOCH,
                                  validation_data=valid_gen, 
                                  callbacks=[model_checkpoint])
    #--------------------------------------------------------------------

    #--------------------------- save results ---------------------------
    origins = list(load_imgs(origin_dir))
    answers = list(load_imgs(answer_dir))
    assert len(origins) == len(answers)

    num_imgs = len(origins)
    aug_det = augmenter(num_imgs,IMG_SIZE,1).to_deterministic()

    origins = aug_det.augment_images(origins)
    answers = aug_det.augment_images(answers)
    predictions = model.predict_generator((img.reshape(1,IMG_SIZE,IMG_SIZE,1) for img in origins), 
                                          num_imgs, verbose=1)

    evaluator.save_img_tuples(zip(origins,answers,predictions),result_dir)

    test_metrics = model.evaluate_generator(test_gen, steps=3)
    #print(model.metrics_names)
    #print(test_metrics)
    print('test set: loss =', test_metrics[0], '| IoU =', test_metrics[1])
    #--------------------------------------------------------------------

    #------------------- evaluation and save results --------------------
    import yaml
    with open(history_path,'w') as f:
        f.write(yaml.dump(dict(
            loss = list(map(np.asscalar,history.history['loss'])),
             acc = list(map(np.asscalar,history.history['mean_iou'])),
        val_loss = list(map(np.asscalar,history.history['val_loss'])),
         val_acc = list(map(np.asscalar,history.history['val_mean_iou'])),
       test_loss = np.asscalar(test_metrics[0]),
        test_acc = np.asscalar(test_metrics[1]) 
        )))

    eval_result_dirpath = 'data/Benigh_74sep/eval_results/eval0'
    evaluator.eval_and_save_result(dataset_dir, save_model_path, eval_result_dirpath)
    '''
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
    '''
    #--------------------------------------------------------------------

if __name__ == '__main__':
    main()
