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
myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
eval_gen = trainGenerator(2,'data/membrane/eval','image','label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',
                                    verbose=1, save_best_only=True)
history = model.fit_generator(myGene,steps_per_epoch=30,epochs=5,callbacks=[model_checkpoint])
#print(dir(history))
#print(history)
#print(history.history)
#print(history.epoch)
#print(history.model)
print('    train loss:', history.history['loss'])
print('train accuracy:', history.history['acc'])

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/membrane/test",results)
