from model import *
from data import *

train_dir = 'data/seg_data/train/'
valid_dir = 'data/seg_data/valid/'
test_dir = 'data/seg_data/test/'
output_dir = 'data/seg_data/output'
save_model_path = 'unet_glends.hdf5'
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
my_gen = dataGenerator(2, train_dir,'image','label',data_gen_args,save_to_dir = None)
valid_gen = dataGenerator(2, valid_dir,'image','label',data_gen_args,save_to_dir = None)
test_gen = dataGenerator(2, test_dir,'image','label',data_gen_args,save_to_dir = None)

print(my_gen)
print(valid_gen)
model = unet(lr=1.0e-7) # 1.3e-7 ~ e-7 ~ 0.7e-7
model_checkpoint = ModelCheckpoint(save_model_path, monitor='val_loss',
                                    verbose=1, save_best_only=True)
history = model.fit_generator(my_gen, steps_per_epoch=30, epochs=1, 
                              validation_data=eval_gen, validation_steps=3,
                              callbacks=[model_checkpoint])
#print(dir(history))
#print(history)
print(history.history.keys())
#print(history.epoch)
#print(history.model)
print('    train loss:', history.history['loss'])
print('train accuracy:', history.history['acc'])
print('    valid loss:', history.history['val_loss'])
print('valid accuracy:', history.history['val_acc'])

testGene = testGenerator("data/membrane/output")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/membrane/output",results)

test = model.evaluate_generator(test_gen, steps=3)
print(model.metrics_names)
print(test)
