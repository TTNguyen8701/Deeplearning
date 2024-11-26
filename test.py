#Setup Enviroment
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd
import numpy as np

def preprocess(image, label):
    image = tf.transpose(image)
    #image = tf.image.flip_left_right(image)
    return image, label

labels_to_unicode = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 0-9
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',  # A-Z
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'   # a-z
]

ds_train , ds_test = tfds.load('emnist/byclass', split=['train', 'test'], as_supervised=True)

ds_train = ds_train.map(preprocess)
ds_test = ds_test.map(preprocess)

image, label = list(ds_train.skip(5).take(1))[0]

image_np = image.numpy().squeeze()

print(labels_to_unicode[label.numpy()])

df = pd.DataFrame(np.squeeze(image_np))
df.to_csv('test.txt', sep='\t', header=False, index=False)