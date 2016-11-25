
# coding: utf-8

# In[7]:

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
import theano
import os
import matplotlib.pyplot as plt
from matplotlib import offsetbox


# In[8]:

class VGG_16(object):

    def __init__(self,weights_path=None):
        model = Sequential()
        l1 = ZeroPadding2D((1,1),input_shape=(3,224,224))
        model.add(l1)
        l2 = Convolution2D(64, 3, 3, activation='relu')
        model.add(l2)
        l3 = ZeroPadding2D((1,1))
        model.add(l3)
        l4 = Convolution2D(64, 3, 3, activation='relu')
        model.add(l4)
        l5 = MaxPooling2D((2,2), strides=(2,2))
        model.add(l5)

        l6 = ZeroPadding2D((1,1))
        model.add(l6)
        l7 = Convolution2D(128, 3, 3, activation='relu')
        model.add(l7)
        l8 = ZeroPadding2D((1, 1))
        model.add(l8)
        l9 = Convolution2D(128, 3, 3, activation='relu')
        model.add(l9)
        l10 = MaxPooling2D((2,2), strides=(2,2))
        model.add(l10)

        l11 = ZeroPadding2D((1,1))
        model.add(l11)
        l12 = Convolution2D(256, 3, 3, activation='relu')
        model.add(l12)
        l13 = ZeroPadding2D((1,1))
        model.add(l13)
        l14 = Convolution2D(256, 3, 3, activation='relu')
        model.add(l14)
        l15 = ZeroPadding2D((1,1))
        model.add(l15)
        l16 = Convolution2D(256, 3, 3, activation='relu')
        model.add(l16)
        l17 = MaxPooling2D((2,2), strides=(2,2))
        model.add(l17)

        l18 = ZeroPadding2D((1,1))
        model.add(l18)
        l19 = Convolution2D(512, 3, 3, activation='relu')
        model.add(l19)
        l20 = ZeroPadding2D((1,1))
        model.add(l20)
        l21 = Convolution2D(512, 3, 3, activation='relu')
        model.add(l21)
        l22 = ZeroPadding2D((1,1))
        model.add(l22)
        l23 = Convolution2D(512, 3, 3, activation='relu')
        model.add(l23)
        l24 = MaxPooling2D((2,2), strides=(2,2))
        model.add(l24)

        l25 = ZeroPadding2D((1,1))
        model.add(l25)
        l26 = Convolution2D(512, 3, 3, activation='relu')
        model.add(l26)
        l27 = ZeroPadding2D((1,1))
        model.add(l27)
        l28 = Convolution2D(512, 3, 3, activation='relu')
        model.add(l28)
        l29 = ZeroPadding2D((1,1))
        model.add(l29)
        l30 = Convolution2D(512, 3, 3, activation='relu')
        model.add(l30)
        l31 = MaxPooling2D((2,2), strides=(2,2))
        model.add(l31)
        l32 = Flatten()
        model.add(l32)
        l33 = Dense(4096, activation='relu')
        model.add(l33)
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        l34 = Dense(1000, activation='softmax')
        model.add(l34)

        model.summary()

        self.get_representation = theano.function([l1.input], l33.output,
                                          allow_input_downcast=True)


        if weights_path:
            model.load_weights(weights_path)

        self.model = model


# In[16]:

import os
from src.util import *
import cv2


# In[17]:

im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
im[:,:,0] -= 103.939
im[:,:,1] -= 116.779
im[:,:,2] -= 123.68
im = im.transpose((2,0,1))
im = np.expand_dims(im, axis=0)

    # Test pretrained model
vgg = VGG_16('vgg16_weights.h5')

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
vgg.model.compile(optimizer=sgd, loss='categorical_crossentropy')
out = vgg.model.predict(im)
print(len(vgg.get_representation(im)[0]))


# In[ ]:

from src.util import *
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

def load_images_from_folder(folder):
    images = []
    thumb_images = []
    for filename in os.listdir(folder):
        img = cv2.resize(cv2.imread(os.path.join(folder,filename)), (224, 224)).astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        if img is not None:
            images.append(img)

        img2 = cv2.resize(cv2.imread(os.path.join(folder, filename)), (20, 20)).astype(np.float32)
        thumb_images.append(img2)


    return images, thumb_images


images,thumb_images = load_images_from_folder("red_square")[0:100]
images,thumb_images = load_images_from_folder("green_square")[0:100]


representations = []
for image in images:
    representations.append(vgg.get_representation(image)[0])

print(len(images))

print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(representations)

plot_embedding(X_tsne, thumb_images,
                   "t-SNE embedding of the digits (time %.2fs)" %
                   (time() - t0))
plt.show()


# In[ ]:



