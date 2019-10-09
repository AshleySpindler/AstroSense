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

def grab_data(i):
    ID = IDs[0][i]
    condition = (IDs[1][i]).astype('uint8')
    img = load_image(ID)

    feature = {'image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
               'condition': _bytes_feature(tf.compat.as_bytes(condition.tostring()))
              }

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example#writer.write(example.SerializeToString())

#load IDs from fits
data = fits.open('Data/Scratch/GalZoo2/gz2_hart16.fits.gz')[1].data

spiral = (data['t02_edgeon_a05_no_flag'] & (data['t02_edgeon_a05_no_count'] > 20)).astype(bool)
data = data[spiral]

global IDs

IDs = [data['dr7objid'],
      np.array([data['t03_bar_a06_bar_flag'],
                data['t03_bar_a07_no_bar_flag'],
                data['t04_spiral_a08_spiral_flag'],
                data['t04_spiral_a09_no_spiral_flag'],
                data['t05_bulge_prominence_a10_no_bulge_flag'],
                data['t05_bulge_prominence_a11_just_noticeable_flag'],
                data['t05_bulge_prominence_a12_obvious_flag'],
                data['t05_bulge_prominence_a13_dominant_flag']
               ])]
IDs[1] = np.swapaxes(IDs[1],0,1)

train, valid, test = 10000, 10000, 25000

gals = choice(len(IDs[0]), train+valid+test)

global gals_train, gals_valid, gals_test
gals_train = gals[0:train]
gals_valid = gals[train:train+valid]
gals_test = gals[train+valid:train+valid+test]

train_filename = 'Data/Scratch/GalZoo2/galzoo_spiral_flags_train.tfrecords'  # address to save the TFRecords file

# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)

pool = Pool(processes=20)       #Set pool of processors
for result in tqdm.tqdm(pool.imap_unordered(grab_data, gals_train), total=len(gals_train)):
    writer.write(result.SerializeToString())

writer.close()
sys.stdout.flush()

valid_filename = 'Data/Scratch/GalZoo2/galzoo_spiral_flags_valid.tfrecords'  # address to save the TFRecords file

# open the TFRecords file
writer = tf.python_io.TFRecordWriter(valid_filename)

pool = Pool(processes=20)       #Set pool of processors
for result in tqdm.tqdm(pool.imap_unordered(grab_data, gals_valid), total=len(gals_valid)):
    writer.write(result.SerializeToString())

writer.close()
sys.stdout.flush()

test_filename = 'Data/Scratch/GalZoo2/galzoo_spiral_flags_test.tfrecords'  # address to save the TFRecords file

# open the TFRecords file
writer = tf.python_io.TFRecordWriter(test_filename)

pool = Pool(processes=20)       #Set pool of processors
for result in tqdm.tqdm(pool.imap_unordered(grab_data, gals_test), total=len(gals_test)):
    writer.write(result.SerializeToString())

writer.close()
sys.stdout.flush()

