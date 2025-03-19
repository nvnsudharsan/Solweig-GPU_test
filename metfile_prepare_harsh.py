import netCDF4 as nc
import numpy as np
import os
import datetime
import re
import pytz
import pandas as pd
import glob
from shapely.geometry import box
from timezonefinder import TimezoneFinder
from osgeo import gdal, ogr, osr

# Input files
netcdf_file = "C:/Users/hk25639/Desktop/Austin/Control.nc"
raster_folder = "C:/Users/hk25639/Desktop/Austin/DEM/"  
output_dir = "C:/Users/hk25639/Desktop/Austin/metfiles/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize TimezoneFinder
tf = TimezoneFinder()

# Load the NetCDF dataset
dataset = nc.Dataset(netcdf_file, "r")

# Initialize GDAL in-memory driver
mem_driver = gdal.GetDriverByName('MEM')

# List all .tif files in the raster folder
tif_files = glob.glob(os.path.join(raster_folder, "*.tif"))

# Define required variables (match NetCDF names)
var_map = {
    "Wind": "WIND",     
    "RH": "RH2",
    "Td": "T2",         # Temperature in Kelvin (to be converted to °C)
    "press": "PSFC",    # Pressure in Pascals (to be converted to kPa)
    "Kdn": "SWDOWN",    
    "ldown": "GLW"      
}

# Fixed values for missing variables
fixed_values = {
    "Q*": -999, "QH": -999, "QE": -999, "Qs": -999, "Qf": -999,
    "snow": -999, "fcld": -999, "wuh": -999, "xsmd": -999, "lai_hr": -999,
    "Kdiff": -999, "Kdir": -999, "Wd": -999,
    "rain": 0  # Fixed rain value for all hours
}

# Extract time information
time_var = dataset.variables["time"][:]
time_units = dataset.variables["time"].units
time_base_date = nc.num2date(time_var, units=time_units, only_use_cftime_datetimes=False)

# Prompt user for the local date
selected_date_str = input("Enter the LOCAL date to process (YYYY-MM-DD): ").strip()
selected_local_date = datetime.datetime.strptime(selected_date_str, "%Y-%m-%d").date()

# Extract lat/lon from the NetCDF dataset and create geotransform for the NetCDF grid
latitudes = dataset.variables["lat"][:]
longitudes = dataset.variables["lon"][:]
min_lon_nc = np.min(longitudes)
max_lon_nc = np.max(longitudes)
min_lat_nc = np.min(latitudes)
max_lat_nc = np.max(latitudes)
width_nc = len(longitudes)
height_nc = len(latitudes)
# GDAL geotransform for NetCDF: (top left x, pixel width, rotation, top left y, rotation, pixel height)
netcdf_gt = (min_lon_nc, (max_lon_nc - min_lon_nc) / width_nc, 0, max_lat_nc, 0, -(max_lat_nc - min_lat_nc) / height_nc)

# Columns in the final dataframe (matching the header order)
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

# Process each TIFF file in the folder
for tif_file in tif_files:
    # Open the TIFF using GDAL to get its bounds and projection
    ds_tif = gdal.Open(tif_file)
    if ds_tif is None:
        print(f"❌ Could not open {tif_file}. Skipping.")
        continue
    gt_tif = ds_tif.GetGeoTransform()
    xsize = ds_tif.RasterXSize
    ysize = ds_tif.RasterYSize

    # Get the TIFF's spatial reference from its projection
    proj_tif = ds_tif.GetProjection()
    srs_tif = osr.SpatialReference()
    srs_tif.ImportFromWkt(proj_tif)

    # Define target spatial reference system (WGS84: EPSG 4326)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(4326)

    # Create coordinate transformation from the TIFF's CRS to WGS84
    transform = osr.CoordinateTransformation(srs_tif, target_srs)

    # Compute the original bounds in the TIFF's coordinate system
    left = gt_tif[0]
    top = gt_tif[3]
    right = left + gt_tif[1] * xsize
    bottom = top + gt_tif[5] * ysize  # note: gt_tif[5] is usually negative

    # Define the four corners of the TIFF in its native CRS
    corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
    # Transform corners to lat-lon (WGS84)
    latlon_corners = [transform.TransformPoint(x, y) for x, y in corners]
    # Fix: swap coordinates so that the first element is latitude and the second is longitude.
    lons = [pt[1] for pt in latlon_corners]
    lats = [pt[0] for pt in latlon_corners]

    # Determine bounding box in lat-lon
    min_lon_tif, max_lon_tif = min(lons), max(lons)
    min_lat_tif, max_lat_tif = min(lats), max(lats)
    # Create a polygon using the lat-lon bounds
    shape = box(min_lon_tif, min_lat_tif, max_lon_tif, max_lat_tif)
    
    # Use the file name (without extension) as the shape attribute for the output file
    shape_name = os.path.splitext(os.path.basename(tif_file))[0]
    shape_name_clean = re.sub(r'\W+', '_', shape_name).replace("DEM", "metfile", 1)
    output_text_file = os.path.join(output_dir, f"{shape_name_clean}_{selected_date_str}.txt")

    # Get shape centroid for timezone conversion
    lat_center, lon_center = shape.centroid.y, shape.centroid.x
    timezone_name = tf.timezone_at(lng=lon_center, lat=lat_center) or "UTC"
    local_tz = pytz.timezone(timezone_name)

    # Convert local date to UTC range
    local_start = local_tz.localize(datetime.datetime.combine(selected_local_date, datetime.time(0, 0)))
    local_end = local_tz.localize(datetime.datetime.combine(selected_local_date, datetime.time(23, 59)))
    utc_start = local_start.astimezone(pytz.utc)
    utc_end = local_end.astimezone(pytz.utc)

    print(
        f"📌 {shape_name_clean}: Local {selected_date_str} ({timezone_name}) → "
        f"UTC {utc_start.date()} {utc_start.hour}:00 to {utc_end.date()} {utc_end.hour}:59"
    )

    # Filter NetCDF time indices for the UTC time range
    time_indices = [
        idx for idx, dt in enumerate(time_base_date)
        if utc_start <= dt.replace(tzinfo=pytz.utc) <= utc_end
    ]
    if not time_indices:
        print(f"❌ No UTC data found for local date {selected_date_str}.")
        continue
    print(f"✅ Processing {len(time_indices)} time steps for {shape_name_clean}")

    # Prepare a list to hold rows for the DataFrame
    met_new = []
    for t in time_indices:
        utc_time = time_base_date[t].replace(tzinfo=pytz.utc)
        local_time = utc_time.astimezone(local_tz)
        year = local_time.year
        doy = local_time.timetuple().tm_yday
        hour = local_time.hour
        minute = local_time.minute

        # Start row with time info
        row = [year, doy, hour, minute]
        # Fixed values first
        row.extend([fixed_values[key] for key in ["Q*", "QH", "QE", "Qs", "Qf"]])

        # Process each variable using GDAL for masking
        for key in ["Wind", "RH", "Td", "press", "Kdn", "ldown"]:
            var_name = var_map[key]
            if var_name in dataset.variables:
                try:
                    # Read the data; note: array is (lon, lat)
                    data_array = dataset.variables[var_name][t, :, :]
                    # Transpose to get (lat, lon) for proper geospatial orientation
                    data_array = data_array.T
                    rows_arr, cols_arr = data_array.shape  # rows_arr = number of lat, cols_arr = number of lon

                    # Create an in-memory raster for the data using the transposed shape
                    data_ds = mem_driver.Create('', cols_arr, rows_arr, 1, gdal.GDT_Float32)
                    data_gt = (min_lon_nc, (max_lon_nc - min_lon_nc) / cols_arr, 0, max_lat_nc, 0, -(max_lat_nc - min_lat_nc) / rows_arr)
                    data_ds.SetGeoTransform(data_gt)
                    srs = osr.SpatialReference()
                    srs.ImportFromEPSG(4326)
                    data_ds.SetProjection(srs.ExportToWkt())
                    data_ds.GetRasterBand(1).WriteArray(data_array)

                    # Create the matching mask raster with the same dimensions and geotransform
                    mask_ds = mem_driver.Create('', cols_arr, rows_arr, 1, gdal.GDT_Byte)
                    mask_ds.SetGeoTransform(data_gt)
                    mask_ds.SetProjection(srs.ExportToWkt())
                    mask_ds.GetRasterBand(1).Fill(0)

                    # Create an in-memory OGR layer to hold the polygon geometry
                    ogr_driver = ogr.GetDriverByName('Memory')
                    ogr_ds = ogr_driver.CreateDataSource('temp')
                    layer = ogr_ds.CreateLayer('poly', srs, ogr.wkbPolygon)
                    field_defn = ogr.FieldDefn('id', ogr.OFTInteger)
                    layer.CreateField(field_defn)
                    feature_defn = layer.GetLayerDefn()
                    feature = ogr.Feature(feature_defn)
                    # Create OGR geometry from the shapely polygon using its WKT
                    geom = ogr.CreateGeometryFromWkt(shape.wkt)
                    feature.SetGeometry(geom)
                    feature.SetField('id', 1)
                    layer.CreateFeature(feature)
                    # Rasterize the polygon onto the mask dataset: pixels inside the polygon become 1
                    gdal.RasterizeLayer(mask_ds, [1], layer, burn_values=[1])

                    # Read the mask array
                    mask_array = mask_ds.GetRasterBand(1).ReadAsArray()
                    # Apply mask: use data_array values where mask == 1; else np.nan
                    masked_data = np.where(mask_array == 1, data_array, np.nan)
                    if np.any(~np.isnan(masked_data)):
                        mean_value = np.nanmean(masked_data)
                    else:
                        mean_value = -999

                    # Conversion adjustments if needed
                    if key == "Td" and mean_value != -999:
                        mean_value -= 273.15
                    if key == "press" and mean_value != -999:
                        mean_value /= 1000.0
                    row.append(mean_value)

                    # Cleanup: explicitly close in-memory datasets if desired
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

        # Add "rain" (always 0 in this example)
        row.append(fixed_values["rain"])
        # Add the remaining fixed values in the correct order
        row.extend([fixed_values[key] for key in ["snow", "fcld", "wuh", "xsmd", "lai_hr", "Kdiff", "Kdir", "Wd"]])
        # Append this row to our data list
        met_new.append(row)

    # Create a pandas DataFrame and save to a text file
    df = pd.DataFrame(met_new, columns=columns)
    df = df[columns_out]
    # Write DataFrame to text file with headers and formatted numerical output
    with open(output_text_file, "w") as f:
        f.write(" ".join(df.columns) + "\n")
        for _, row in df.iterrows():
            f.write('{:d} {:d} {:d} {:d} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.5f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {: .2f}\n'.format(
                int(row["iy"]), int(row["id"]), int(row["it"]), int(row["imin"]),
                row["Q*"], row["QH"], row["QE"], row["Qs"], row["Qf"],
                row["Wind"], row["RH"], row["Td"], row["press"], row["rain"],
                row["Kdn"], row["snow"], row["ldown"], row["fcld"], row["wuh"],
                row["xsmd"], row["lai_hr"], row["Kdiff"], row["Kdir"], row["Wd"]
            ))

dataset.close()
print(f"All raster extents processed and metfiles saved in {output_dir}")