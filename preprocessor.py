import os
import re
import glob
import datetime
import pytz
import numpy as np
import pandas as pd
import netCDF4 as nc
import shutil
from osgeo import gdal, ogr, osr
from shapely.geometry import box
from timezonefinder import TimezoneFinder

# =============================================================================
# Function to check that all raster files have matching dimensions, pixel size, and CRS.
# =============================================================================
def check_rasters(files):
    if not files:
        raise ValueError("No raster files provided.")

    ref_file = files[0]
    ds = gdal.Open(ref_file)
    if ds is None:
        raise FileNotFoundError(f"Could not open {ref_file}")
    ref_width = ds.RasterXSize
    ref_height = ds.RasterYSize
    ref_gt = ds.GetGeoTransform()  # (originX, pixelWidth, rot, originY, rot, pixelHeight)
    ref_pixel_width = ref_gt[1]
    ref_pixel_height = ref_gt[5]  # typically negative
    ref_crs = ds.GetProjection()
    ds = None

    for f in files[1:]:
        ds = gdal.Open(f)
        if ds is None:
            raise FileNotFoundError(f"Could not open {f}")
        if ds.RasterXSize != ref_width or ds.RasterYSize != ref_height:
            raise ValueError("Error: Raster dimensions do not match.")
        gt = ds.GetGeoTransform()
        pixel_width = gt[1]
        pixel_height = gt[5]
        if pixel_width != ref_pixel_width or pixel_height != ref_pixel_height:
            raise ValueError("Error: Pixel sizes do not match.")
        if ds.GetProjection() != ref_crs:
            raise ValueError("Error: CRS does not match.")
        ds = None

    return True

# =============================================================================
# Function to tile a raster file into smaller chunks.
# =============================================================================
def create_tiles(infile, tilesize, tile_type):
    ds = gdal.Open(infile)
    if ds is None:
        raise FileNotFoundError(f"Could not open {infile}")

    width = ds.RasterXSize
    height = ds.RasterYSize

    out_folder = os.path.join(os.path.dirname(infile), tile_type)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if tilesize >= width and tilesize >= height:
        outfile = os.path.join(out_folder, f"{tile_type}_0_0.tif")
        options = gdal.TranslateOptions(format='GTiff', srcWin=[0, 0, width, height])
        gdal.Translate(outfile, ds, options=options)
        print(f"Created single tile (original file): {outfile}")
        ds = None
        return

    for i in range(0, width, tilesize):
        for j in range(0, height, tilesize):
            tile_width = min(tilesize, width - i)
            tile_height = min(tilesize, height - j)
            outfile = os.path.join(out_folder, f"{tile_type}_{i}_{j}.tif")
            options = gdal.TranslateOptions(format='GTiff', srcWin=[i, j, tile_width, tile_height])
            gdal.Translate(outfile, ds, options=options)
            print(f"Created tile: {outfile}")
    
    ds = None

# =============================================================================
# Function to process the NetCDF file and create metfiles based on a set of raster tiles.
# =============================================================================
def process_metfiles(netcdf_file, raster_folder, base_path, selected_date_str):
    metfiles_folder = os.path.join(base_path, "metfiles")
    os.makedirs(metfiles_folder, exist_ok=True)
    
    tf = TimezoneFinder()
    dataset = nc.Dataset(netcdf_file, "r")
    mem_driver = gdal.GetDriverByName('MEM')
    
    tif_files = glob.glob(os.path.join(raster_folder, "*.tif"))
    if not tif_files:
        print(f"No TIFF files found in {raster_folder}.")
        dataset.close()
        return

    var_map = {
        "Wind": "WIND",     
        "RH": "RH2",
        "Td": "T2",         # Temperature in Kelvin (to be converted to °C)
        "press": "PSFC",    # Pressure in Pascals (to be converted to kPa)
        "Kdn": "SWDOWN",    
        "ldown": "GLW"      
    }
    fixed_values = {
        "Q*": -999, "QH": -999, "QE": -999, "Qs": -999, "Qf": -999,
        "snow": -999, "fcld": -999, "wuh": -999, "xsmd": -999, "lai_hr": -999,
        "Kdiff": -999, "Kdir": -999, "Wd": -999,
        "rain": 0
    }
    
    time_var = dataset.variables["time"][:]
    time_units = dataset.variables["time"].units
    time_base_date = nc.num2date(time_var, units=time_units, only_use_cftime_datetimes=False)
    
    selected_local_date = datetime.datetime.strptime(selected_date_str, "%Y-%m-%d").date()
    
    latitudes = dataset.variables["lat"][:]
    longitudes = dataset.variables["lon"][:]
    min_lon_nc = np.min(longitudes)
    max_lon_nc = np.max(longitudes)
    min_lat_nc = np.min(latitudes)
    max_lat_nc = np.max(latitudes)
    width_nc = len(longitudes)
    height_nc = len(latitudes)
    netcdf_gt = (min_lon_nc, (max_lon_nc - min_lon_nc) / width_nc, 0, max_lat_nc, 0, -(max_lat_nc - min_lat_nc) / height_nc)
    
    columns = [
        'iy', 'id', 'it', 'imin',
        'Q*', 'QH', 'QE', 'Qs', 'Qf',
        'Wind', 'RH', 'Td', 'press',
        'Kdn', 'ldown','rain', 'snow', 'fcld', 'wuh', 'xsmd', 'lai_hr',
        'Kdiff', 'Kdir', 'Wd'
    ]
    columns_out = [
        "iy", "id", "it", "imin",
        "Q*", "QH", "QE", "Qs", "Qf",
        "Wind", "RH", "Td", "press",
        "rain",
        "Kdn",
        "snow",
        "ldown",
        "fcld",
        "wuh",
        "xsmd",
        "lai_hr",
        "Kdiff",
        "Kdir",
        "Wd"
    ]
    
    for tif_file in tif_files:
        ds_tif = gdal.Open(tif_file)
        if ds_tif is None:
            print(f"❌ Could not open {tif_file}. Skipping.")
            continue
        gt_tif = ds_tif.GetGeoTransform()
        xsize = ds_tif.RasterXSize
        ysize = ds_tif.RasterYSize

        proj_tif = ds_tif.GetProjection()
        srs_tif = osr.SpatialReference()
        srs_tif.ImportFromWkt(proj_tif)
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(4326)
        transform = osr.CoordinateTransformation(srs_tif, target_srs)

        left = gt_tif[0]
        top = gt_tif[3]
        right = left + gt_tif[1] * xsize
        bottom = top + gt_tif[5] * ysize
        corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
        latlon_corners = [transform.TransformPoint(x, y) for x, y in corners]
        lats = [pt[0] for pt in latlon_corners]
        lons = [pt[1] for pt in latlon_corners]

        min_lon_tif, max_lon_tif = min(lons), max(lons)
        min_lat_tif, max_lat_tif = min(lats), max(lats)
        shape = box(min_lon_tif, min_lat_tif, max_lon_tif, max_lat_tif)
        
        shape_name = os.path.splitext(os.path.basename(tif_file))[0]
        shape_name_clean = re.sub(r'\W+', '_', shape_name).replace("DEM", "metfile", 1)
        output_text_file = os.path.join(metfiles_folder, f"{shape_name_clean}_{selected_date_str}.txt")
        
        lat_center, lon_center = shape.centroid.y, shape.centroid.x
        timezone_name = tf.timezone_at(lng=lon_center, lat=lat_center) or "UTC"
        local_tz = pytz.timezone(timezone_name)
        
        local_start = local_tz.localize(datetime.datetime.combine(selected_local_date, datetime.time(0, 0)))
        local_end = local_tz.localize(datetime.datetime.combine(selected_local_date, datetime.time(23, 59)))
        utc_start = local_start.astimezone(pytz.utc)
        utc_end = local_end.astimezone(pytz.utc)
        
        print(
            f"📌 {shape_name_clean}: Local {selected_date_str} ({timezone_name}) → "
            f"UTC {utc_start.date()} {utc_start.hour}:00 to {utc_end.date()} {utc_end.hour}:59"
        )
        
        time_indices = [
            idx for idx, dt in enumerate(time_base_date)
            if utc_start <= dt.replace(tzinfo=pytz.utc) <= utc_end
        ]
        if not time_indices:
            print(f"❌ No UTC data found for local date {selected_date_str} in {tif_file}.")
            ds_tif = None
            continue
        print(f"✅ Processing {len(time_indices)} time steps for {shape_name_clean}")
        
        met_new = []
        for t in time_indices:
            utc_time = time_base_date[t].replace(tzinfo=pytz.utc)
            local_time = utc_time.astimezone(local_tz)
            year = local_time.year
            doy = local_time.timetuple().tm_yday
            hour = local_time.hour
            minute = local_time.minute

            row = [year, doy, hour, minute]
            row.extend([fixed_values[key] for key in ["Q*", "QH", "QE", "Qs", "Qf"]])
            
            for key in ["Wind", "RH", "Td", "press", "Kdn", "ldown"]:
                var_name = var_map[key]
                if var_name in dataset.variables:
                    try:
                        data_array = dataset.variables[var_name][t, :, :].T
                        rows_arr, cols_arr = data_array.shape

                        data_ds = mem_driver.Create('', cols_arr, rows_arr, 1, gdal.GDT_Float32)
                        data_gt = (min_lon_nc, (max_lon_nc - min_lon_nc) / cols_arr, 0, max_lat_nc, 0, -(max_lat_nc - min_lat_nc) / rows_arr)
                        data_ds.SetGeoTransform(data_gt)
                        srs = osr.SpatialReference()
                        srs.ImportFromEPSG(4326)
                        data_ds.SetProjection(srs.ExportToWkt())
                        data_ds.GetRasterBand(1).WriteArray(data_array)

                        mask_ds = mem_driver.Create('', cols_arr, rows_arr, 1, gdal.GDT_Byte)
                        mask_ds.SetGeoTransform(data_gt)
                        mask_ds.SetProjection(srs.ExportToWkt())
                        mask_ds.GetRasterBand(1).Fill(0)

                        ogr_driver = ogr.GetDriverByName('Memory')
                        ogr_ds = ogr_driver.CreateDataSource('temp')
                        layer = ogr_ds.CreateLayer('poly', srs, ogr.wkbPolygon)
                        field_defn = ogr.FieldDefn('id', ogr.OFTInteger)
                        layer.CreateField(field_defn)
                        feature_defn = layer.GetLayerDefn()
                        feature = ogr.Feature(feature_defn)
                        geom = ogr.CreateGeometryFromWkt(shape.wkt)
                        feature.SetGeometry(geom)
                        feature.SetField('id', 1)
                        layer.CreateFeature(feature)
                        gdal.RasterizeLayer(mask_ds, [1], layer, burn_values=[1])

                        mask_array = mask_ds.GetRasterBand(1).ReadAsArray()
                        masked_data = np.where(mask_array == 1, data_array, np.nan)
                        if np.any(~np.isnan(masked_data)):
                            mean_value = np.nanmean(masked_data)
                        else:
                            mean_value = -999

                        if key == "Td" and mean_value != -999:
                            mean_value -= 273.15
                        if key == "press" and mean_value != -999:
                            mean_value /= 1000.0
                        row.append(mean_value)

                        data_ds = None
                        mask_ds = None
                        ogr_ds = None
                    except IndexError:
                        print(
                            f"IndexError: Variable {var_name} has incorrect dimensions {dataset.variables[var_name].shape}"
                        )
                        row.append(-999)
                else:
                    row.append(-999)

            row.append(fixed_values["rain"])
            row.extend([fixed_values[key] for key in ["snow", "fcld", "wuh", "xsmd", "lai_hr", "Kdiff", "Kdir", "Wd"]])
            met_new.append(row)

        df = pd.DataFrame(met_new, columns=columns)
        df = df[columns_out]
        with open(output_text_file, "w") as f:
            f.write(" ".join(df.columns) + "\n")
            for _, row in df.iterrows():
                f.write('{:d} {:d} {:d} {:d} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.5f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {: .2f} {: .2f}\n'.format(
                    int(row["iy"]), int(row["id"]), int(row["it"]), int(row["imin"]),
                    row["Q*"], row["QH"], row["QE"], row["Qs"], row["Qf"],
                    row["Wind"], row["RH"], row["Td"], row["press"], row["rain"],
                    row["Kdn"], row["snow"], row["ldown"], row["fcld"], row["wuh"],
                    row["xsmd"], row["lai_hr"], row["Kdiff"], row["Kdir"], row["Wd"]
                ))
        print(f"Metfile saved: {output_text_file}")
        ds_tif = None

    dataset.close()
    print(f"All raster extents processed and metfiles saved in {metfiles_folder}")

# =============================================================================
# Function to process own met file: copies the source met file into new files
# renaming each copy based on the numeric suffix extracted from .tif files.
# =============================================================================
def create_met_files(base_path, source_met_file):
    raster_folder = os.path.join(base_path, 'Building_DSM')
    target_folder = os.path.join(base_path, 'metfiles')

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    for file in os.listdir(raster_folder):
        if file.lower().endswith('.tif'):
            name_without_ext = os.path.splitext(file)[0]
            prefix = 'Building_DSM_'
            if name_without_ext.startswith(prefix):
                digits = name_without_ext[len(prefix):]
                new_filename = f'metfile_{digits}.txt'
                target_met_file = os.path.join(target_folder, new_filename)
                shutil.copy(source_met_file, target_met_file)
                print(f"Copied to {target_met_file}")

# =============================================================================
# Main function: checks rasters, creates tiles, and creates metfiles using either a
# user-supplied met file or a netCDF file. Only the parameters required for the chosen
# method need to be provided.
# =============================================================================
def ppr(base_path, building_dsm_filename, dem_filename, trees_filename,
         tile_size, selected_date_str, use_own_met=False, netcdf_filename=None, own_met_file=None):
    
    building_dsm_path = os.path.join(base_path, building_dsm_filename)
    dem_path = os.path.join(base_path, dem_filename)
    trees_path = os.path.join(base_path, trees_filename)

    # Check that all rasters have matching dimensions, pixel size, and CRS.
    try:
        check_rasters([building_dsm_path, dem_path, trees_path])
    except ValueError as error:
        print(error)
        exit(1)

    rasters = {
        "Building_DSM": building_dsm_path,
        "DEM": dem_path,
        "Trees": trees_path
    }
    for tile_type, raster in rasters.items():
        print(f"Creating tiles for {tile_type}...")
        create_tiles(raster, tile_size, tile_type)
    
    # For metfiles processing, we use the DEM tiles folder.
    dem_tiles_folder = os.path.join(os.path.dirname(dem_path), "DEM")
    
    # Choose between own met file or .nc file based on the flag.
    if use_own_met:
        if own_met_file is None:
            print("Error: Please provide the path to your own met file.")
            exit(1)
        create_met_files(base_path, own_met_file)
    else:
        if netcdf_filename is None:
            print("Error: Please provide the netCDF file path.")
            exit(1)
        netcdf_path = os.path.join(base_path, netcdf_filename)
        process_metfiles(netcdf_path, dem_tiles_folder, base_path, selected_date_str)

# =============================================================================
# Example usage:
# For processing a netCDF file (use_own_met = False), do not provide own_met_file.
# =============================================================================
#base_path = 'C:/Users/hk25639/Desktop/Austin/'
#building_dsm_filename = 'Building_DSM.tif'
#dem_filename = 'DEM.tif'
#trees_filename = 'Trees.tif'
#tile_size = 3600
#selected_date_str = '2020-08-13'

# When processing a netCDF file:
#use_own_met = False
# netcdf_filename = 'Control.nc'   # Required in this case

# When using your own met file:
#use_own_met = True
#own_met_file = os.path.join(base_path, 'ownmet.txt')   # Required in this case

#if __name__ == "__main__":
    # For netCDF processing, call main without an own_met_file argument.
    # main(base_path, building_dsm_filename, dem_filename, trees_filename,
    #      tile_size, selected_date_str, use_own_met, netcdf_filename=netcdf_filename)
    
    # For using a custom met file, uncomment the lines below and comment the above call.
   # main(base_path, building_dsm_filename, dem_filename, trees_filename,
   #      tile_size, selected_date_str, use_own_met, own_met_file=own_met_file)
