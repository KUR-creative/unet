from model import *
from data import *
import os
import numpy as np
import skimage.io as io
from skimage.viewer import ImageViewer
from utils import file_paths

train_dir = 'data/seg_data/train/'
valid_dir = 'data/seg_data/valid/'
test_dir = 'data/seg_data/test/'
#output_dir = 'data/seg_data/output'
#save_model_path = 'unet_glends.hdf5'
output_dir = 'data/seg_data/modified_output'
save_model_path = 'modified_unet_glends.h5'
history_path = 'modified_history.yml'
'''
train_dir = 'data/membrane/train/'
valid_dir = 'data/membrane/valid/'
test_dir = 'data/membrane/test/'
output_dir = 'data/membrane/output'
save_model_path = 'unet_membrane.hdf5'
'''
'''
ratio_0 = 0
ratio_1 = 0
num_img = 0
for path in file_paths(os.path.join(train_dir,'label')):
    #print(path)
    img = io.imread(path, as_gray=True)
    h,w = img.shape[:2]

    num_all = h*w
    num_1 = np.count_nonzero(img) 
    num_0 = num_all - num_1

    ratio_1 += num_1 / num_all
    ratio_0 += num_0 / num_all
    num_img += 1

    #io.imshow(img); io.show()
    #viewer = ImageViewer(img)
    #viewer.show()
ratio_0 /= num_img
ratio_1 /= num_img
weight_0 = ratio_1
weight_1 = ratio_0
print(ratio_0, ratio_1)
'''

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
batch_size = 4
#learning_rate = 0.1e-10# 1.0e-9: best -> now, lower(slower). 
#learning_rate = 200 # for jaccard loss
decay = 0.1
learning_rate = 1.0 # for Adadelta
#0 ~ 10000: 1.0e-7 later: 1.0e-8

my_gen = dataGenerator(batch_size, train_dir,'image','label',data_gen_args,save_to_dir = None)
valid_gen = dataGenerator(batch_size, valid_dir,'image','label',data_gen_args,save_to_dir = None)
test_gen = dataGenerator(batch_size, test_dir,'image','label',data_gen_args,save_to_dir = None)

#loaded_model= save_model_path
loaded_model = None
if loaded_model:
    model = unet(pretrained_weights=loaded_model,
                 lr=learning_rate, decay=decay)#, weight_0=weight_0, weight_1=weight_1) 
else:
    model = unet(lr=learning_rate, decay=decay)#, weight_0=weight_0, weight_1=weight_1)

model_checkpoint = ModelCheckpoint(save_model_path, monitor='val_loss',
                                    verbose=1, save_best_only=True)
history = model.fit_generator(my_gen, steps_per_epoch=10, epochs=100, 
                              validation_data=valid_gen, validation_steps=3,#)
                              callbacks=[model_checkpoint])
'''
print('    train loss:', history.history['loss'])
print('train accuracy:', history.history['acc'])
print('    valid loss:', history.history['val_loss'])
print('valid accuracy:', history.history['val_acc'])
'''
import yaml
with open(history_path,'w') as f:
    f.write(yaml.dump(dict(loss = list(map(np.asscalar,history.history['loss'])),
                            acc = list(map(np.asscalar,history.history['mean_iou'])),
                       val_loss = list(map(np.asscalar,history.history['val_loss'])),
                        val_acc = list(map(np.asscalar,history.history['val_mean_iou'])))))

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

output_gen = outputGenerator(output_dir,4)#,(512,512))
results = model.predict_generator(output_gen,4,verbose=1)
saveResult(output_dir,results)

test = model.evaluate_generator(test_gen, steps=3)
print(model.metrics_names)
print(test)

