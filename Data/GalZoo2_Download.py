from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import wget
import os
from PIL import Image
import numpy as np
from multiprocessing import Pool

def galaxyget(i):
    objid = gz['OBJID'][i]
    if os.path.isfile('GalaxyZooImages/galaxy_'+str(objid)+'.png')==True:
        return
    print(i)
    c = SkyCoord(gz['RA'][i], gz['DEC'][i], unit=(u.hourangle, u.deg))
    ra, dec = str(c.ra.deg), str(c.dec.deg)
    wget.download(url='http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?ra='+ra+'&dec='+dec+'&scale=0.2&width=424&height=424', out='Scratch/GalaxyZooImages/galaxy_'+str(objid)+'.jpg')
    im = Image.open('Scratch/GalaxyZooImages/galaxy_'+str(objid)+'.jpg')
    im.save('Scratch/GalaxyZooImages/galaxy_'+str(objid)+'.png', 'PNG')
    im.close()
    os.remove('Scratch/GalaxyZooImages/galaxy_'+str(objid)+'.jpg')

#Initiate Parallel Processes
if __name__ == '__main__':
    if os.path.isfile('Scratch/gz2sample.fits.gz')==False:
        wget.download(url='http://gz2hart.s3.amazonaws.com/gz2_hart16.fits.gz', out='Scratch/gz2sample.fits.gz')
    gz = fits.open('Scratch/gz2sample.fits.gz')[1].data
    vals = np.where(gz)[0]
    pool = Pool(processes=10)       #Set pool of processors
    pool.map(galaxyget, vals)      #Call function over iterable

