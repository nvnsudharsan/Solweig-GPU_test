import os
import numpy as np
import xarray as xr
from netCDF4 import Dataset, date2num
from datetime import datetime, timedelta

def saturation_vapor_pressure(T):
    """
    Calculate saturation vapor pressure (in hPa) given temperature T in Celsius.
    """
    return 6.112 * np.exp((17.67 * T) / (T + 243.5))

# Define file names for the ERA5 datasets
instant_file = 'data_stream-oper_stepType-instant.nc'
accum_file   = 'data_stream-oper_stepType-accum.nc'

# Open the datasets using xarray
ds_instant = xr.open_dataset(instant_file)
ds_accum   = xr.open_dataset(accum_file)

# Generate the correct time array using the provided start and end times (1-hour frequency)
start_time = datetime(2020, 8, 12, 0, 0, 0)
end_time   = datetime(2020, 8, 14, 23, 0, 0)
time_array = [start_time + timedelta(hours=i) for i in range(int((end_time - start_time).total_seconds() // 3600) + 1)]

# Process instantaneous variables:
# Convert temperatures from Kelvin to Celsius
temperatures = ds_instant['t2m'].values - 273.15
dew_points   = ds_instant['d2m'].values - 273.15

# Convert surface pressure from hPa to kPa
surface_pressures = ds_instant['sp'].values / 10.0

# Calculate wind speed from u and v components (m/s)
u10 = ds_instant['u10'].values
v10 = ds_instant['v10'].values
wind_speeds = np.sqrt(u10**2 + v10**2)

# Process accumulated radiation fields:
# Convert from J m^-2 (accumulated over 3 hours) to W m^-2 by dividing by 3600.
shortwave_radiation = ds_accum['ssrd'].values / 3600.0
longwave_radiation  = ds_accum['strd'].values / 3600.0

# Compute relative humidity (in %)
# First, compute the saturation vapor pressures (in hPa)
e_temp      = saturation_vapor_pressure(temperatures)
e_dew_point = saturation_vapor_pressure(dew_points)
relative_humidities = 100.0 * (e_dew_point / e_temp)

# Extract latitude and longitude.
# ERA5 files often include 1D coordinates for latitude and longitude.
latitudes = ds_instant['latitude'].values
longitudes = ds_instant['longitude'].values

# If lat and lon are 1D, convert them to 2D arrays
if latitudes.ndim == 1 and longitudes.ndim == 1:
    lon2d, lat2d = np.meshgrid(longitudes, latitudes)
else:
    lat2d = latitudes
    lon2d = longitudes

# Define the output NetCDF file
output_file = 'ERA5_processed.nc'
with Dataset(output_file, 'w', format='NETCDF4') as nc:
    # Define dimensions
    nc.createDimension('time', len(time_array))
    nc.createDimension('lat', lat2d.shape[0])
    nc.createDimension('lon', lon2d.shape[1])
    
    # Create coordinate variables
    time_var = nc.createVariable('time', 'f8', ('time',))
    lat_var  = nc.createVariable('lat', 'f4', ('lat', 'lon'))
    lon_var  = nc.createVariable('lon', 'f4', ('lat', 'lon'))
    
    # Create data variables with compression enabled
    t2_var   = nc.createVariable('T2', 'f4', ('time', 'lat', 'lon'), zlib=True)
    psfc_var = nc.createVariable('PSFC', 'f4', ('time', 'lat', 'lon'), zlib=True)
    rh2_var  = nc.createVariable('RH2', 'f4', ('time', 'lat', 'lon'), zlib=True)
    wind_var = nc.createVariable('WIND', 'f4', ('time', 'lat', 'lon'), zlib=True)
    swdown_var = nc.createVariable('SWDOWN', 'f4', ('time', 'lat', 'lon'), zlib=True)
    glw_var    = nc.createVariable('GLW', 'f4', ('time', 'lat', 'lon'), zlib=True)
    
    # Set attributes for coordinate variables
    # The time units are now defined relative to the start time.
    time_var.units = "hours since 2020-08-12 00:00:00"
    time_var.calendar = "gregorian"
    lat_var.units = "degrees_north"
    lon_var.units = "degrees_east"
    
    # Set attributes for data variables
    t2_var.units = "degC"
    psfc_var.units = "kPa"
    rh2_var.units = "%"
    wind_var.units = "m/s"
    swdown_var.units = "W/m^2"
    glw_var.units = "W/m^2"
    
    # Write coordinate data
    time_var[:] = date2num(time_array, units=time_var.units, calendar=time_var.calendar)
    lat_var[:, :] = lat2d
    lon_var[:, :] = lon2d
    
    # Write processed variable data.
    # It is assumed that the first dimension of each variable corresponds to time.
    t2_var[:, :, :]   = temperatures
    psfc_var[:, :, :] = surface_pressures
    rh2_var[:, :, :]  = relative_humidities
    wind_var[:, :, :] = wind_speeds
    swdown_var[:, :, :] = shortwave_radiation
    glw_var[:, :, :]    = longwave_radiation

print("New NetCDF file created:", output_file)