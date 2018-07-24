from model import *
from data import *

train_dir = 'data/seg_data/train/'
valid_dir = 'data/seg_data/valid/'
test_dir = 'data/seg_data/test/'
output_dir = 'data/seg_data/output'
#save_model_path = 'unet_glends.hdf5'
save_model_path = 'modified_unet_glends.hdf5'
history_path = 'modified_history.yml'
'''
train_dir = 'data/membrane/train/'
valid_dir = 'data/membrane/valid/'
test_dir = 'data/membrane/test/'
output_dir = 'data/membrane/output'
save_model_path = 'unet_membrane.hdf5'
'''

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
batch_size = 16
learning_rate = 0.1e-7# 1.0e-7: best -> now, lower(slower). 
#0 ~ 10000: 1.0e-7 later: 1.0e-8

my_gen = dataGenerator(batch_size, train_dir,'image','label',data_gen_args,save_to_dir = None)
valid_gen = dataGenerator(batch_size, valid_dir,'image','label',data_gen_args,save_to_dir = None)
test_gen = dataGenerator(batch_size, test_dir,'image','label',data_gen_args,save_to_dir = None)

#loaded_model_path = save_model_path
loaded_model = None
if loaded_model:
    model = unet(pretrained_weights=loaded_model_path,lr=learning_rate) 
else:
    model = unet(lr=learning_rate)
model_checkpoint = ModelCheckpoint(save_model_path, monitor='val_loss',
                                    verbose=1, save_best_only=True)
history = model.fit_generator(my_gen, steps_per_epoch=3, epochs=10, 
                              validation_data=valid_gen, validation_steps=3#)
                              ,callbacks=[model_checkpoint])
'''
print('    train loss:', history.history['loss'])
print('train accuracy:', history.history['acc'])
print('    valid loss:', history.history['val_loss'])
print('valid accuracy:', history.history['val_acc'])
'''
import yaml
with open(history_path,'w') as f:
    f.write(yaml.dump(dict(loss = list(map(np.asscalar,history.history['loss'])),
                            acc = list(map(np.asscalar,history.history['acc'])),
                       val_loss = list(map(np.asscalar,history.history['val_loss'])),
                        val_acc = list(map(np.asscalar,history.history['val_acc'])))))

import matplotlib.pyplot as plt
plt.clf()
plt.plot(history.history['loss'], 'b', label='train loss')
plt.plot(history.history['val_loss'], 'r', label='valid loss')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.legend(fontsize=10)
plt.draw()
plt.show()

output_gen = outputGenerator(output_dir,4)#,(512,512))
results = model.predict_generator(output_gen,4,verbose=1)
saveResult(output_dir,results)

test = model.evaluate_generator(test_gen, steps=3)
print(model.metrics_names)
print(test)

