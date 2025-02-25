import os
from osgeo import gdal
import numpy as np
import scipy.ndimage.interpolation as sc
import math
from concurrent.futures import ProcessPoolExecutor

aspect_output_path = 'your path'
wall_output_path = "your path"
dem_folder_path = 'your path'

walllimit = 3.

def findwalls(a, walllimit):
    col = a.shape[0]
    row = a.shape[1]
    walls = np.zeros((col, row))
    domain = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    index = 0
    for i in np.arange(1, row-1):
        for j in np.arange(1, col-1):
            dom = a[j-1:j+2, i-1:i+2]
            walls[j, i] = np.max(dom[np.where(domain == 1)])
            index = index + 1

    walls = np.copy(walls - a)
    walls[(walls < walllimit)] = 0

    walls[0:walls.shape[0], 0] = 0
    walls[0:walls.shape[0], walls.shape[1] - 1] = 0
    walls[0, 0:walls.shape[0]] = 0
    walls[walls.shape[0] - 1, 0:walls.shape[1]] = 0

    return walls

def filter1Goodwin_as_aspect_v3(walls, scale, a):
    row = a.shape[0]
    col = a.shape[1]

    filtersize = np.floor((scale + 0.0000000001) * 9)
    if filtersize <= 2:
        filtersize = 3
    else:
        if filtersize != 9:
            if filtersize % 2 == 0:
                filtersize = filtersize + 1

    filthalveceil = int(np.ceil(filtersize / 2.))
    filthalvefloor = int(np.floor(filtersize / 2.))

    filtmatrix = np.zeros((int(filtersize), int(filtersize)))
    buildfilt = np.zeros((int(filtersize), int(filtersize)))

    filtmatrix[:, filthalveceil - 1] = 1
    n = filtmatrix.shape[0] - 1
    buildfilt[filthalveceil - 1, 0:filthalvefloor] = 1
    buildfilt[filthalveceil - 1, filthalveceil: int(filtersize)] = 2

    y = np.zeros((row, col))
    z = np.zeros((row, col))
    x = np.zeros((row, col))
    walls[walls > 0] = 1

    for h in range(0, 180):
        filtmatrix1temp = sc.rotate(filtmatrix, h, order=1, reshape=False, mode='nearest')
        filtmatrix1 = np.round(filtmatrix1temp)
        filtmatrixbuildtemp = sc.rotate(buildfilt, h, order=0, reshape=False, mode='nearest')
        filtmatrixbuild = np.round(filtmatrixbuildtemp)
        index = 270 - h
        if h == 150:
            filtmatrixbuild[:, n] = 0
        if h == 30:
            filtmatrixbuild[:, n] = 0
        if index == 225:
            filtmatrix1[0, 0] = 1
            filtmatrix1[n, n] = 1
        if index == 135:
            filtmatrix1[0, n] = 1
            filtmatrix1[n, 0] = 1

        for i in range(int(filthalveceil) - 1, row - int(filthalveceil) - 1):
            for j in range(int(filthalveceil) - 1, col - int(filthalveceil) - 1):
                if walls[i, j] == 1:
                    wallscut = walls[i - filthalvefloor:i + filthalvefloor + 1,
                               j - filthalvefloor:j + filthalvefloor + 1] * filtmatrix1
                    dsmcut = a[i - filthalvefloor:i + filthalvefloor + 1, j - filthalvefloor:j + filthalvefloor + 1]
                    if z[i, j] < wallscut.sum():
                        z[i, j] = wallscut.sum()
                        if np.sum(dsmcut[filtmatrixbuild == 1]) > np.sum(dsmcut[filtmatrixbuild == 2]):
                            x[i, j] = 1
                        else:
                            x[i, j] = 2

                        y[i, j] = index

    y[(x == 1)] = y[(x == 1)] - 180
    y[(y < 0)] = y[(y < 0)] + 360

    grad, asp = get_ders(a, scale)

    y = y + ((walls == 1) * 1) * ((y == 0) * 1) * (asp / (math.pi / 180.))

    dirwalls = y

    return dirwalls

def cart2pol(x, y, units='deg'):
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if units in ['deg', 'degs']:
        theta = theta * 180 / np.pi
    return theta, radius

def get_ders(dsm, scale):
    dx = 1/scale
    fy, fx = np.gradient(dsm, dx, dx)
    asp, grad = cart2pol(fy, fx, 'rad')
    grad = np.arctan(grad)
    asp = asp * -1
    asp = asp + (asp < 0) * (np.pi * 2)
    return grad, asp

def process_file(filename):
    dem_path = os.path.join(dem_folder_path, filename)
    dataset = gdal.Open(dem_path)
    band = dataset.GetRasterBand(1)
    a = band.ReadAsArray().astype(np.float32)

    geotransform = dataset.GetGeoTransform()
    scale = 1 / geotransform[1]

    rows = a.shape[0]
    cols = a.shape[1]

    walls = findwalls(a, walllimit)
    dirwalls = filter1Goodwin_as_aspect_v3(walls, scale, a)

    driver = gdal.GetDriverByName('GTiff')
    wall_output_file = os.path.join(wall_output_path, f'walls_{filename[13:-3]}.tif')
    aspect_output_file = os.path.join(aspect_output_path, f'aspect_{filename[13:-3]}.tif')

    out_dataset = driver.Create(wall_output_file, cols, rows, 1, gdal.GDT_Float32)
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(walls.astype(np.float32))
    out_band.FlushCache()
    out_dataset = None


    out_dataset = driver.Create(aspect_output_file, cols, rows, 1, gdal.GDT_Float32)
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(dirwalls.astype(np.float32))
    out_band.FlushCache()
    out_dataset = None
    dataset = None
    print(f'{filename} completed')

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        dem_files = [f for f in os.listdir(dem_folder_path) if f.endswith('.tif')]
        executor.map(process_file, dem_files)
