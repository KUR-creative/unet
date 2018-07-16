from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
my_gen = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
eval_gen = trainGenerator(2,'data/membrane/eval','image','label',data_gen_args,save_to_dir = None)
test_gen = trainGenerator(2,'data/membrane/test','image','label',data_gen_args,save_to_dir = None)

print(my_gen)
print(eval_gen)
model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',
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
