from skimage.io import imread, imshow
from skimage.transform import resize

import numpy as np

import os

import torch
import torchvision

def load_images(directory, imdir='./'):
    imgs = []
    labels = []
    for i in os.listdir(os.path.join(imdir, directory)):
        img = resize(imread(os.path.join(imdir, directory, i)), (224, 224), anti_aliasing=True)
        imgs.append(np.moveaxis(img, 2, 0))
        if 'cat' in i:
            labels.append(0)
        else:
            labels.append(1)
    imgs = np.array(imgs)
    return imgs, labels

def data_gen(train_images, train_labels, batch_size, eval=True):
    all_idxs = np.arange(len(train_labels))
    idxs = np.random.shuffle(all_idxs)
    i = 0
    while i * batch_size + batch_size < train_images.shape[0]:
        samples = train_images[i*batch_size: (i+1)*batch_size]
        sample_labels = train_labels[i*batch_size: (i+1)*batch_size]
        i += 1
        yield torch.tensor(samples, dtype=torch.float), torch.tensor(sample_labels, dtype=torch.long)
    if eval:
        samples = train_images[(i)*batch_size:]
        sample_labels = train_labels[(i)*batch_size:]
        yield torch.tensor(samples, dtype=torch.float), torch.tensor(sample_labels, dtype=torch.long)
def postproc_output(out):
  sm = torch.nn.Softmax(dim=1)
  return sm(out).detach().numpy()

def evaluate_model(model_path, data_path):
    '''takes in a path to a model and a set of images 
    evaluates the model on that set of images''' 
    pass