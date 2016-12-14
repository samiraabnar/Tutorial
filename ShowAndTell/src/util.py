
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import os
import numpy as np
from PIL import Image

# Scale and visualize the embedding vectors
def plot_embedding(X, image_tags, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            #if np.min(dist) < 4e-3:
                # don't show points that are too close
            #    continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(image_tags[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def load_images_from_folder(folder):
    images = []
    thumb_images = []
    for filename in os.listdir(folder):
        
        img = Image.open(os.path.join(folder,filename))
        img = img.resize((224,224))
        img = np.asarray(img, dtype='float32') / 256.
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        if img is not None:
            images.append(img)

        img2 = Image.open(os.path.join(folder,filename))
        img2 = img2.resize((224,224))
        img2 = np.asarray(img, dtype='float32') / 256.
        thumb_images.append(img2)


    return images, thumb_images



def get_string(list_of_vec,VOCAB):
    str = " "
    for i in np.arange(len(list_of_vec)):
        str += " "+VOCAB[np.argmax(list_of_vec[i])]

    return str.strip()



