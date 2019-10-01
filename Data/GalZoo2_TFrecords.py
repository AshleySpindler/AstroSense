import numpy as np
import tensorflow as tf
from astropy.io import fits
import matplotlib.image as mpimg
from skimage.transform import rescale
from skimage.color import rgb2gray
from numpy.random import choice
from skimage.exposure import rescale_intensity
import sys

def load_image(ID):
    img = mpimg.imread('/data/astroml/aspindler/GalaxyZoo/GalaxyZooImages/galaxy_'+str(ID)+'.png')
    img_crop = img[84:340,84:340]
    img_grey = rgb2gray(img_crop)
    img_64 = rescale(img_grey, scale=4, preserve_range=True)
    img_uint = rescale_intensity(img_128, out_range=(0,255)).astype('uint8')
    return img_uint

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#load IDs from fits
IDs = fits.open('/data/astroml/aspindler/GalaxyZoo/gz2sample.fits.gz')[1].data['OBJID']

train, valid, test = 10000, 10000, 100000

gals = choice(IDs.shape[0], train+valid+test)

gals_train = IDs[gals[0:train]]
gals_valid = IDs[gals[train:train+valid]]
gals_test = IDs[gals[train+valid:train+valid+test]]

train_filename = '/data/astroml/aspindler/GalaxyZoo/galzoo_train.tfrecords'  # address to save the TFRecords file

# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)

for i in range(train):
    if not i % 100:
        print('Train data: {}/{}'.format(i, train))

        sys.stdout.flush()

    img = load_image(gals_train[i])

    feature = {'train/label': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()

valid_filename = '/data/astroml/aspindler/GalaxyZoo/galzoo_valid.tfrecords'  # address to save the TFRecords file

# open the TFRecords file
writer = tf.python_io.TFRecordWriter(valid_filename)

for i in range(valid):
    if not i % 1000:
        print('Valid data: {}/{}'.format(i, valid))
        sys.stdout.flush()

    img = load_image(gals_valid[i])

    feature = {'image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()

test_filename = '/data/astroml/aspindler/GalaxyZoo/galzoo_test.tfrecords'  # address to save the TFRecords file

# open the TFRecords file
writer = tf.python_io.TFRecordWriter(test_filename)

for i in range(test):
    if not i % 1000:
        print('Test data: {}/{}'.format(i, test))
        sys.stdout.flush()

    img = load_image(gals_test[i])

    feature = {'image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()
