import netCDF4 as nc
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import os
import datetime
import re
import pytz
import pandas as pd
from timezonefinder import TimezoneFinder

# Input files
netcdf_file = "/Users/geo-ns36752/Downloads/Control.nc"
shapefile = "/Users/geo-ns36752/Downloads/DEM/raster_boundaries.shp"
output_dir = "/Users/geo-ns36752/Downloads/DEM/metfiles/"

# Attribute field to use as filename
attribute_column = "Raster"  # Change if needed

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize TimezoneFinder
tf = TimezoneFinder()

# Load the NetCDF dataset
dataset = nc.Dataset(netcdf_file, "r")

# Load the shapefile
gdf = gpd.read_file(shapefile)

# Define required variables (match NetCDF names)
var_map = {
    "Wind": "WIND",     # Changed to "WIND" from Control.nc
    "RH": "RH2",
    "Td": "T2",         # Will convert from Kelvin to Celsius
    "press": "PSFC",    # Will convert from Pascals to kPa
    "Kdn": "SWDOWN",    # Downward shortwave radiation
    "ldown": "GLW"      # Downward longwave radiation
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

# Extract lat/lon
latitudes = dataset.variables["lat"][:]
longitudes = dataset.variables["lon"][:]

# Create GeoTransform for raster alignment
transform = rasterio.transform.from_bounds(
    longitudes.min(), latitudes.min(), longitudes.max(), latitudes.max(),
    len(longitudes), len(latitudes)
)

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

# Process each shape in the shapefile
for i, shape in enumerate(gdf.geometry):
    shape_name = str(gdf.loc[i, attribute_column])
    shape_name_clean = re.sub(r'\W+', '_', shape_name)
    output_text_file = os.path.join(output_dir, f"{shape_name_clean}_{selected_date_str}.txt")
    # Get shape centroid for timezone conversion
    lon, lat = shape.centroid.x, shape.centroid.y
    timezone_name = tf.timezone_at(lng=lon, lat=lat) or "UTC"
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
        # Extracted variables from NetCDF
        for key in ["Wind", "RH", "Td", "press", "Kdn", "ldown"]:
            var_name = var_map[key]
            if var_name in dataset.variables:
                try:
                    data_array = dataset.variables[var_name][t, :, :]
                    # Write a temporary raster in memory
                    with rasterio.open(
                        "/vsimem/temp_raster.tif", "w",
                        driver="GTiff",
                        height=len(latitudes),
                        width=len(longitudes),
                        count=1,
                        dtype=str(data_array.dtype),
                        crs="EPSG:4326",
                        transform=transform
                    ) as temp_raster:
                        temp_raster.write(data_array, 1)
                    # Mask with the current shape to get the average
                    with rasterio.open("/vsimem/temp_raster.tif") as src:
                        out_image, _ = rasterio.mask.mask(src, [shape], crop=True, nodata=np.nan)
                        mean_value = np.nanmean(out_image) if np.any(~np.isnan(out_image)) else -999
                        # Convert temperature from K to °C
                        if key == "Td" and mean_value != -999:
                            mean_value -= 273.15
                        # Convert pressure from Pa to kPa
                        if key == "press" and mean_value != -999:
                            mean_value /= 1000.0
                        row.append(mean_value)
                except IndexError:
                    print(
                        f"IndexError: Variable {var_name} "
                        f"has incorrect dimensions {dataset.variables[var_name].shape}"
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
            f.write('{:d} {:d} {:d} {:d} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.5f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
                int(row["iy"]), int(row["id"]), int(row["it"]), int(row["imin"]), 
                row["Q*"], row["QH"], row["QE"], row["Qs"], row["Qf"], 
                row["Wind"], row["RH"], row["Td"], row["press"], row["rain"], 
                row["Kdn"], row["snow"], row["ldown"], row["fcld"], row["wuh"], 
                row["xsmd"], row["lai_hr"], row["Kdiff"], row["Kdir"], row["Wd"]
            ))
            
dataset.close()
print(f"All shape averages saved in {output_dir}")
