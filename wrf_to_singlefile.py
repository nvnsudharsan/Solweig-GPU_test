import os
import xarray as xr
import numpy as np
from netCDF4 import Dataset, date2num
from datetime import datetime, timedelta

# Directory containing WRF output files
control_dir = "/scratch/08457/h_kamath/Austin_heat/New_with_trees_USE_THIS/CR_plus_trees/"

# List of WRF output files
common_files = [
    'wrfout_d03_2020-08-12_18:00:00',
    'wrfout_d03_2020-08-13_00:00:00',
    'wrfout_d03_2020-08-13_06:00:00',
    'wrfout_d03_2020-08-13_12:00:00',
    'wrfout_d03_2020-08-13_18:00:00',
    'wrfout_d03_2020-08-14_00:00:00',
    'wrfout_d03_2020-08-14_06:00:00',
    'wrfout_d03_2020-08-14_12:00:00',
    'wrfout_d03_2020-08-14_18:00:00',
    'wrfout_d03_2020-08-15_00:00:00',
    'wrfout_d03_2020-08-15_06:00:00',
    'wrfout_d03_2020-08-15_12:00:00',
    'wrfout_d03_2020-08-15_18:00:00',
    'wrfout_d03_2020-08-16_00:00:00',
    'wrfout_d03_2020-08-16_06:00:00',
    'wrfout_d03_2020-08-16_12:00:00',
    'wrfout_d03_2020-08-16_18:00:00',
]

# Generate the correct time array for 96 hours (1-hour frequency)
start_time = datetime(2020, 8, 12, 18, 0, 0)
end_time = datetime(2020, 8, 16, 23, 0, 0)
time_array = [start_time + timedelta(hours=i) for i in range(int((end_time - start_time).total_seconds() // 3600) + 1)]

# Constants for RH calculation
Rd = 287.05  # Gas constant for dry air (J/kg/K)
Rv = 461.5   # Gas constant for water vapor (J/kg/K)
eps = Rd / Rv
Lv = 2.5e6   # Latent heat of vaporization (J/kg)

def calculate_rh(t2, q2, psfc):
    """Calculate relative humidity from temperature, mixing ratio, and surface pressure."""
    e_s = 6.112 * np.exp((17.67 * (t2 - 273.15)) / (t2 - 273.15 + 243.5))  # Saturation vapor pressure (hPa)
    e_s = e_s * 100  # Convert to Pa
    e = q2 * psfc / (eps + q2)  # Actual vapor pressure (Pa)
    rh = (e / e_s) * 100  # Relative humidity (%)
    rh = np.clip(rh, 0, 100)  # Ensure RH is within 0-100%
    return rh

# Initialize lists to store data for all variables
t2_list, wind_list, rh2_list, tsk_list = [], [], [], []
utci_list, ac_list, pv_list, swdown_list, glw_list, psfc_list = [], [], [], [], [], []

# Process each file
for file in common_files:
    file_path = os.path.join(control_dir, file)
    with xr.open_dataset(file_path) as ds:
        # Extract variables
        t2 = ds['T2'].values  # 2-meter temperature
        q2 = ds['Q2'].values  # Mixing ratio at 2 meters
        psfc = ds['PSFC'].values  # Surface pressure

        t2_list.append(t2)
        tsk_list.append(ds['TSK'].values)  # Land surface temperature
        utci_list.append(ds['COMF_50'].values)  # UTCI
        ac_list.append(ds['CM_AC_URB3D'].values)  # Air condition
        pv_list.append(ds['EP_PV_URB3D'].values)  # PV energy consumption
        swdown_list.append(ds['SWDOWN'].values)  # Downwelling shortwave radiation
        glw_list.append(ds['GLW'].values)  # Downwelling longwave radiation
        psfc_list.append(psfc)

        # Calculate wind speed from U and V components at the first model level
        u10 = ds['U10'].values  # U component of wind at 10 m
        v10 = ds['V10'].values  # V component of wind at 10 m
        wind_speed = np.sqrt(u10**2 + v10**2)
        wind_list.append(wind_speed)

        # Calculate relative humidity
        rh2 = calculate_rh(t2, q2, psfc)
        rh2_list.append(rh2)

        # Extract latitude and longitude (assuming they are the same for all files)
        if len(t2_list) == 1:  # Only extract once
            lat = ds['XLAT'].values[0, :, :]
            lon = ds['XLONG'].values[0, :, :]

# Combine all data arrays into single arrays
t2_array = np.concatenate(t2_list, axis=0)
wind_array = np.concatenate(wind_list, axis=0)
rh2_array = np.concatenate(rh2_list, axis=0)
tsk_array = np.concatenate(tsk_list, axis=0)
utci_array = np.concatenate(utci_list, axis=0)
ac_array = np.concatenate(ac_list, axis=0)
pv_array = np.concatenate(pv_list, axis=0)
swdown_array = np.concatenate(swdown_list, axis=0)
glw_array = np.concatenate(glw_list, axis=0)
psfc_array = np.concatenate(psfc_list, axis=0)

# Create a new NetCDF file
output_file = "/scratch/08457/h_kamath/Austin_heat/Simulations_processed/Cool_roof_realistic_plus_trees.nc"
with Dataset(output_file, 'w', format='NETCDF4') as nc:
    # Define dimensions
    time_dim = nc.createDimension('time', len(time_array))
    lat_dim = nc.createDimension('lat', lat.shape[0])
    lon_dim = nc.createDimension('lon', lon.shape[1])

    # Create variables
    time_var = nc.createVariable('time', 'f8', ('time',))
    lat_var = nc.createVariable('lat', 'f4', ('lat', 'lon'))
    lon_var = nc.createVariable('lon', 'f4', ('lat', 'lon'))
    t2_var = nc.createVariable('T2', 'f4', ('time', 'lat', 'lon'), zlib=True)
    wind_var = nc.createVariable('WIND', 'f4', ('time', 'lat', 'lon'), zlib=True)
    rh2_var = nc.createVariable('RH2', 'f4', ('time', 'lat', 'lon'), zlib=True)
    tsk_var = nc.createVariable('TSK', 'f4', ('time', 'lat', 'lon'), zlib=True)
    utci_var = nc.createVariable('UTCI', 'f4', ('time', 'lat', 'lon'), zlib=True)
    ac_var = nc.createVariable('AC_consumption', 'f4', ('time', 'lat', 'lon'), zlib=True)
    pv_var = nc.createVariable('PV_generation', 'f4', ('time', 'lat', 'lon'), zlib=True)
    swdown_var = nc.createVariable('SWDOWN', 'f4', ('time', 'lat', 'lon'), zlib=True)
    glw_var = nc.createVariable('GLW', 'f4', ('time', 'lat', 'lon'), zlib=True)
    psfc_var = nc.createVariable('PSFC', 'f4', ('time', 'lat', 'lon'), zlib=True)

    # Set attributes
    time_var.units = "hours since 1970-01-01 00:00:00"
    time_var.calendar = "gregorian"
    lat_var.units = "degrees_north"
    lon_var.units = "degrees_east"
    t2_var.units = "K"
    wind_var.units = "m/s"
    rh2_var.units = "%"
    tsk_var.units = "K"
    utci_var.units = "degrees_C"
    ac_var.units = "W/m^2"
    pv_var.units = "W/m^2"
    swdown_var.units = "W/m^2"
    glw_var.units = "W/m^2"
    psfc_var.units = "Pa"

    # Write data
    time_var[:] = date2num(time_array, units=time_var.units, calendar=time_var.calendar)
    lat_var[:, :] = lat
    lon_var[:, :] = lon
    t2_var[:, :, :] = t2_array
    wind_var[:, :, :] = wind_array
    rh2_var[:, :, :] = rh2_array 
    tsk_var[:, :, :] = tsk_array
    utci_var[:, :, :] = utci_array
    ac_var[:, :, :] = ac_array
    pv_var[:, :, :] = pv_array
    swdown_var[:, :, :] = swdown_array
    glw_var[:, :, :] = glw_array
    psfc_var[:, :, :] = psfc_array

print(f"New NetCDF file created: {output_file}")
