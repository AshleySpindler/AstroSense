import numpy as np
import tensorflow as tf
from astropy.io import fits
import matplotlib.image as mpimg
from skimage.transform import rescale
from skimage.color import rgb2gray
from numpy.random import choice
from skimage.exposure import rescale_intensity
import sys
from multiprocessing import Pool
import tqdm

import warnings
warnings.filterwarnings("ignore")

#TODO tidy this shit up

def load_image(ID):
    img = mpimg.imread('/data/astroml/aspindler/GalaxyZoo/GalaxyZooImages/galaxy_'+str(ID)+'.png')
    img_crop = img[84:340,84:340]
    img_grey = rgb2gray(img_crop)
    img_64 = rescale(img_grey, scale=0.25, preserve_range=True)
    img_uint = rescale_intensity(img_64, out_range=(0,255)).astype('uint8')
    return img_uint

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def grab_data(ID):
    img = load_image(ID)

    feature = {'image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example#writer.write(example.SerializeToString())

#load IDs from fits
IDs = fits.open('/data/astroml/aspindler/GalaxyZoo/gz2sample.fits.gz')[1].data['OBJID']

train, valid, test = 10000, 10000, 100000

gals = choice(IDs.shape[0], train+valid+test)

gals_train = IDs[gals[0:train]]
gals_valid = IDs[gals[train:train+valid]]
gals_test = IDs[gals[train+valid:train+valid+test]]

train_filename = 'Data/Scratch/GalZoo2/galzoo_train.tfrecords'  # address to save the TFRecords file

# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)

pool = Pool(processes=20)       #Set pool of processors
for result in tqdm.tqdm(pool.imap_unordered(grab_data, gals_train), total=len(gals_train)):
    writer.write(result.SerializeToString())

writer.close()
sys.stdout.flush()

valid_filename = 'Data/Scratch/GalZoo2/galzoo_valid.tfrecords'  # address to save the TFRecords file

# open the TFRecords file
writer = tf.python_io.TFRecordWriter(valid_filename)

pool = Pool(processes=20)       #Set pool of processors
for result in tqdm.tqdm(pool.imap_unordered(grab_data, gals_valid), total=len(gals_valid)):
    writer.write(result.SerializeToString())

writer.close()
sys.stdout.flush()

test_filename = 'Data/Scratch/GalZoo2/galzoo_test.tfrecords'  # address to save the TFRecords file

# open the TFRecords file
writer = tf.python_io.TFRecordWriter(test_filename)

pool = Pool(processes=20)       #Set pool of processors
for result in tqdm.tqdm(pool.imap_unordered(grab_data, gals_test), total=len(gals_test)):
    writer.write(result.SerializeToString())

writer.close()
sys.stdout.flush()
