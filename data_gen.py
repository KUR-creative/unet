import numpy as np
from imgaug import augmenters as iaa

def augmenter(batch_size = 4, crop_size=256):
    n_img = batch_size
    size = crop_size
    #n_ch = num_channels
    def func_images(images, random_state, parents, hooks):
        _,_,n_ch = images[0].shape
        ret_imgs = np.empty((n_img,size,size,n_ch))
        for idx,img in enumerate(images):
            h,w,_ = img.shape
            y = random_state.randint(0, h - size)
            x = random_state.randint(0, w - size)
            ret_imgs[idx] = img[y:y+size,x:x+size].reshape((size,size,n_ch))
        return ret_imgs
        
    def func_heatmaps(heatmaps, random_state, parents, hooks):
        return heatmaps
    def func_keypoints(keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    return iaa.Sequential([
             iaa.Lambda(
               func_images=func_images,
               func_heatmaps=func_heatmaps,
               func_keypoints=func_keypoints),
           ])

'''
import re
def human_sorted(iterable):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(iterable, key = alphanum_key)

def preprocess(img):
    #h,w = img.shape[:2]
    img = img / 255
    #img = trans.resize(img, (256,256))
    #return img.reshape((h,w,3))
    return img

load_imgs = (lambda img_dir: 
               list(map(lambda path: preprocess(io.imread(path)),
                        human_sorted(file_paths(img_dir)))))
imgs = load_imgs('data')
ms = load_imgs('data')

# Standard scenario: You have N RGB-images and additionally 21 heatmaps per image.
# You want to augment each image and its heatmaps identically.
iaa.Crop(
    px=(0, 128),
    keep_size=False), # crop images from each side by 0 to 16px (randomly chosen)
iaa.Fliplr(0.5), # horizontally flip 50% of the images
iaa.Affine(
    rotate=(-90, 90),
    mode='reflect',
)
seq = iaa.Sequential([
    sqr_crop,
])

io.imshow_collection(imgs); io.show()

# Convert the stochastic sequence of augmenters to a deterministic one.
# The deterministic sequence will always apply the exactly same effects to the images.
seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
aug_imgs = seq_det.augment_images(imgs)
aug_ms = seq_det.augment_images(ms)

print(1, type(aug_imgs))
io.imshow_collection(aug_imgs); io.show()
io.imshow_collection(aug_ms); io.show()

seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
aug_imgs = seq_det.augment_images(imgs)
aug_ms = seq_det.augment_images(ms)

print(2)
io.imshow_collection(aug_imgs); io.show()
io.imshow_collection(aug_ms); io.show()
'''
