from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import numpy as np
from math import radians
from copy import deepcopy
from osgeo import gdal, osr
import datetime
import calendar
import scipy.ndimage.interpolation as sc
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate
import time
from sun_position_cpu import Solweig_2015a_metdata_noload
from shadow_gpu import svf_calculator, create_patches
from solweig_gpu import Solweig_2022a_calc, clearnessindex_2013b
from UTCI_gpu import utci_calculator
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Always change these inputs
base_output_path = '/scratch/09295/naveens/Austin_test/UTCI/'
inputMet = '/scratch/09295/naveens/Austin_test/met_files'
building_dsm_dir = '/scratch/09295/naveens/Austin_test/Building_DSM/'
tree_dir = '/scratch/09295/naveens/Austin_test/Only_trees/'
dem_dir = '/scratch/09295/naveens/Austin_test/DEM/'
walls_dir = '/scratch/09295/naveens/Austin_test/walls/'
aspect_dir = '/scratch/09295/naveens/Austin_test/aspect/'

utc = 6.
#wind_speed = torch.tensor(1.5)  # Wind speed = 1.5 m/s

# Wall and ground emissivity and albedo
albedo_b = 0.2
albedo_g = 0.15
ewall = 0.9
eground = 0.95
absK = 0.7
absL = 0.95

# Standing position
Fside = 0.22
Fup = 0.06
Fcyl = 0.28

cyl = True
elvis = 0
usevegdem = 1
onlyglobal = 1

firstdayleaf = 97
lastdayleaf = 300
conifer_bool = False

def load_raster_to_tensor(dem_path):
    dataset = gdal.Open(dem_path)
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray().astype(np.float32)
    return torch.tensor(array, device=device), dataset

# Function to list matching files in a directory
def get_matching_files(directory, extension):
    return sorted([f for f in os.listdir(directory) if f.endswith(extension)])

def extract_number_from_filename(filename):
    number = filename[13:-4] # change according to the naming of building DSM files
    return number


def process_file_set(building_dsm_path, tree_path, dem_path, walls_path, aspect_path, met_file, output_path):
    a, dataset = load_raster_to_tensor(building_dsm_path)
    temp1, dataset2 = load_raster_to_tensor(tree_path)
    temp2, dataset3 = load_raster_to_tensor(dem_path)
    walls, dataset4 = load_raster_to_tensor(walls_path)
    dirwalls, dataset5 = load_raster_to_tensor(aspect_path)
    rows, cols = a.shape
    geotransform = dataset.GetGeoTransform()
    scale = 1 / geotransform[1]
    projection_wkt = dataset.GetProjection()
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(projection_wkt)
    wgs84_wkt = """GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""
    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)
    transform = osr.CoordinateTransformation(old_cs, new_cs)
    widthx = dataset.RasterXSize
    heightx = dataset.RasterYSize
    geotransform = dataset.GetGeoTransform()
    minx = geotransform[0]
    miny = geotransform[3] + widthx * geotransform[4] + heightx * geotransform[5]
    lonlat = transform.TransformPoint(minx, miny)
    gdalver = float(gdal.__version__[0])
    if gdalver == 3.:
        lon = lonlat[1]  # changed to gdal 3
        lat = lonlat[0]  # changed to gdal 3
    else:
        lon = lonlat[0]  # changed to gdal 2
        lat = lonlat[1]  # changed to gdal 2
    alt = torch.median(temp2)
    if alt > 0:
        alt = 3.
    location = {'longitude': lon, 'latitude': lat, 'altitude': alt}
    YYYY, altitude, azimuth, zen, jday, leafon, dectime, altmax = Solweig_2015a_metdata_noload(met_file, location, utc)
    temp1[temp1 < 0.] = 0.
    vegdem = temp1 + temp2
    vegdem2 = torch.add(temp1 * 0.25, temp2)
    bush = torch.logical_not(vegdem2 * vegdem) * vegdem
    vegdsm = temp1 + a
    vegdsm[vegdsm == a] = 0
    vegdsm2 = temp1 * 0.25 + a
    vegdsm2[vegdsm2 == a] = 0
    amaxvalue = torch.maximum(a.max(), vegdem.max())
    buildings = a - temp2
    buildings[buildings < 2.] = 1.
    buildings[buildings >= 2.] = 0.
    valid_mask = (buildings == 1)
    Knight = torch.zeros((rows, cols), device=device)
    Tgmap1 = torch.zeros((rows, cols), device=device)
    Tgmap1E = torch.zeros((rows, cols), device=device)
    Tgmap1S = torch.zeros((rows, cols), device=device)
    Tgmap1W = torch.zeros((rows, cols), device=device)
    Tgmap1N = torch.zeros((rows, cols), device=device)
    TgOut1 = torch.zeros((rows, cols), device=device)
    TgK = Knight + 0.37
    Tstart = Knight - 3.41
    alb_grid = Knight + albedo_g
    emis_grid = Knight + eground
    TgK_wall = 0.37
    Tstart_wall = -3.41
    TmaxLST = 15.
    TmaxLST_wall = 15.
    transVeg = 3. / 100.
    landcover = 0
    lcgrid = False
    anisotropic_sky = 1
    patch_option = 2
    DOY = torch.tensor(met_file[:, 1], device=device)
    hours = torch.tensor(met_file[:, 2], device=device)
    minu = torch.tensor(met_file[:, 3], device=device)
    Ta = torch.tensor(met_file[:, 11], device=device)
    RH = torch.tensor(met_file[:, 10], device=device)
    radG = torch.tensor(met_file[:, 14], device=device)
    radD = torch.tensor(met_file[:, 21], device=device)
    radI = torch.tensor(met_file[:, 22], device=device)
    P = torch.tensor(met_file[:, 12], device=device)
    Ws = torch.tensor(met_file[:, 9], device=device)
    # Prepare leafon based on vegetation type
    if conifer_bool:
        leafon = torch.ones((1, DOY.shape[0]), device=device)
    else:
        leafon = torch.zeros((1, DOY.shape[0]), device=device)
        if firstdayleaf > lastdayleaf:
            leaf_bool = ((DOY > firstdayleaf) | (DOY < lastdayleaf))
        else:
            leaf_bool = ((DOY > firstdayleaf) & (DOY < lastdayleaf))
        leafon[0, leaf_bool] = 1
    psi = leafon * transVeg
    psi[leafon == 0] = 0.5
    Twater = []
    height = 1.1
    height = torch.tensor(height, device=device)
    first = torch.round(torch.tensor(height, device=device))
    if first == 0.:
        first = torch.tensor(1., device=device)
    second = torch.round(height * 20.)

    if len(Ta) == 1:
        timestepdec = 0
    else:
        timestepdec = dectime[1] - dectime[0]
    timeadd = 0.
    firstdaytime = 1.
    start_time = time.time()
    # Calculate SVF and related parameters (remains unchanged)
    svf, svfaveg, svfE, svfEaveg, svfEveg, svfN, svfNaveg, svfNveg, svfS, svfSaveg, svfSveg, svfveg, svfW, svfWaveg, svfWveg, vegshmat, vbshvegshmat, shmat = svf_calculator(patch_option, amaxvalue, a, vegdsm, vegdsm2, bush, scale)
    svfbuveg = svf - (1.0 - svfveg) * (1.0 - transVeg)
    asvf = torch.acos(torch.sqrt(svf))
    diffsh = torch.zeros((rows, cols, shmat.shape[2]), device=device)
    for i in range(shmat.shape[2]):
        diffsh[:, :, i] = shmat[:, :, i] - (1 - vegshmat[:, :, i]) * (1 - transVeg)
    tmp = svf + svfveg - 1.0
    tmp[tmp < 0.0] = 0.0
    svfalfa = torch.asin(torch.exp(torch.log(1.0 - tmp) / 2.0))
    # Prepare lists to store results for all time steps
    UTCI_all = []
    TMRT_all = []
    for i in np.arange(0, Ta.__len__()):
        if landcover == 1:
            if ((dectime[i] - np.floor(dectime[i]))) == 0 or (i == 0):
                Twater = np.mean(Ta[jday[0] == np.floor(dectime[i])])
        if (dectime[i] - np.floor(dectime[i])) == 0:
            daylines = np.where(np.floor(dectime) == dectime[i])
            if daylines.__len__() > 1:
                alt = altitude[0][daylines]
                alt2 = np.where(alt > 1)
                rise = alt2[0][0]
                [_, CI, _, _, _] = clearnessindex_2013b(zen[0, i + rise + 1], jday[0, i + rise + 1], Ta[i + rise + 1],
                                                        RH[i + rise + 1] / 100., radG[i + rise + 1], location, P[i + rise + 1])
                if (CI > 1.) or (CI == np.inf):
                    CI = 1.
            else:
                CI = 1.
        Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, shadow, firstdaytime, timestepdec, timeadd, \
        Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, Keast, Ksouth, Kwest, Knorth, Least, Lsouth, Lwest, Lnorth, \
        KsideI, TgOut1, TgOut, radIout, radDout, Lside, Lsky_patch_characteristics, CI_Tg, CI_TgG, KsideD, dRad, Kside = Solweig_2022a_calc(
            i, a, scale, rows, cols, svf, svfN, svfW, svfE, svfS, svfveg, svfNveg, svfEveg, svfSveg, svfWveg, svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg, vegdsm, vegdsm2, albedo_b, absK, absL, ewall, Fside, Fup, Fcyl,
            altitude[0][i], azimuth[0][i], zen[0][i], jday[0][i], usevegdem, onlyglobal, buildings, location, psi[0][i], landcover, lcgrid, dectime[i], altmax[0][i], dirwalls, walls, cyl, elvis, Ta[i], RH[i], radG[i], radD[i], radI[i], P[i],
            amaxvalue, bush, Twater, TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST, TmaxLST_wall, first, second, svfalfa, svfbuveg, firstdaytime, timeadd, timestepdec, Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N,
            CI, TgOut1, diffsh, shmat, vegshmat, vbshvegshmat, anisotropic_sky, asvf, patch_option)
        # Create matrices for meteorological parameters for the current time step
        Ta_mat = torch.zeros((rows, cols), device='cuda') + Ta[i]
        RH_mat = torch.zeros((rows, cols), device='cuda') + RH[i]
        Tmrt_mat = torch.zeros((rows, cols), device='cuda') + Tmrt
        va10m_mat = torch.zeros((rows, cols), device='cuda') + Ws[i]
        UTCI_mat = utci_calculator(Ta_mat, RH_mat, Tmrt_mat, va10m_mat)
        UTCI = torch.full(UTCI_mat.shape, float('nan'), device='cuda')
        UTCI[valid_mask] = UTCI_mat[valid_mask]
        # Append results (converted to CPU numpy arrays) to the lists
        UTCI_all.append(UTCI.cpu().numpy())
        TMRT_all.append(Tmrt.cpu().numpy())
    # Convert the lists to numpy arrays with shape (time_steps, rows, cols)
    UTCI_all = np.array(UTCI_all)
    TMRT_all = np.array(TMRT_all)
    # Write a multi-band GeoTIFF for UTCI (each band corresponds to one time step)
    driver = gdal.GetDriverByName('GTiff')
    out_file_path = os.path.join(output_path, 'UTCI_{number}.tif')
    num_bands = UTCI_all.shape[0]
    out_dataset = driver.Create(out_file_path, cols, rows, num_bands, gdal.GDT_Float32)
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())
    for band in range(num_bands):
        out_band = out_dataset.GetRasterBand(band + 1)
        out_band.WriteArray(UTCI_all[band])
        out_band.FlushCache()
    out_dataset = None
    # Optionally, you can similarly write TMRT to a single multi-band file:
    out_file_path_tmrt = os.path.join(output_path, 'TMRT_{number}.tif')
    num_bands_tmrt = TMRT_all.shape[0]
    out_dataset_tmrt = driver.Create(out_file_path_tmrt, cols, rows, num_bands_tmrt, gdal.GDT_Float32)
    out_dataset_tmrt.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset_tmrt.SetProjection(dataset.GetProjection())
    for band in range(num_bands_tmrt):
        out_band = out_dataset_tmrt.GetRasterBand(band + 1)
        out_band.WriteArray(TMRT_all[band])
        out_band.FlushCache()
    out_dataset_tmrt = None
    # Clean up datasets
    dataset = None
    dataset2 = None
    dataset3 = None
    dataset4 = None
    dataset5 = None
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken to execute the function for {building_dsm_path}: {time_taken:.2f} seconds")

# Load met files and create a mapping
met_files = get_matching_files(inputMet, ".txt")
# Load DSM and corresponding datasets
building_dsm_files = get_matching_files(building_dsm_dir, ".tif")
tree_files = get_matching_files(tree_dir, ".tif")
dem_files = get_matching_files(dem_dir, ".tif")
walls_files = get_matching_files(walls_dir, ".tif")
aspect_files = get_matching_files(aspect_dir, ".tif")

for i in range(len(building_dsm_files)):
    building_dsm_path = os.path.join(building_dsm_dir, building_dsm_files[i])
    tree_path = os.path.join(tree_dir, tree_files[i])
    dem_path = os.path.join(dem_dir, dem_files[i])
    walls_path = os.path.join(walls_dir, walls_files[i])
    aspect_path = os.path.join(aspect_dir, aspect_files[i])
    number = extract_number_from_filename(building_dsm_files[i])
    output_folder = os.path.join(base_output_path, number)
    os.makedirs(output_folder, exist_ok=True)
    met_file_path = os.path.join(inputMet, met_files[i])
    met_file_data = np.loadtxt(met_file_path, skiprows=1, delimiter=' ')
    output_folder = os.path.join(base_output_path, number)
    os.makedirs(output_folder, exist_ok=True)
    process_file_set(building_dsm_path, tree_path, dem_path, walls_path, aspect_path, met_file_data, output_folder)
    torch.cuda.empty_cache()

