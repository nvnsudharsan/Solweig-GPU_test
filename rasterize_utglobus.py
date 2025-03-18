import geopandas as gpd
from osgeo import ogr, gdal
import sys
import os

def rasterize_gpkg(input_file, output_file, resolution):
    """
    Rasterizes a GeoPackage file using the 'height' attribute, setting no-data values to 0.
    
    Parameters:
        input_file (str): Path to the input .gpkg file.
        output_file (str): Path to the output raster (GeoTIFF) file.
        resolution (float): Grid spacing (pixel size) in meters.
    
    Returns:
        str: The output file path.
    """
    # Open the vector dataset using OGR
    vector_ds = ogr.Open(input_file)
    if vector_ds is None:
        print("Could not open vector dataset.")
        sys.exit(1)
    
    # Get the first layer (adjust if necessary)
    layer = vector_ds.GetLayer(0)
    
    # Get the extent of the vector layer (xmin, xmax, ymin, ymax)
    xmin, xmax, ymin, ymax = layer.GetExtent()
    print("Vector extent:", xmin, xmax, ymin, ymax)
    
    # Calculate raster dimensions based on the desired resolution
    x_pixels = int((xmax - xmin) / resolution)
    y_pixels = int((ymax - ymin) / resolution)
    print("Raster size (pixels):", x_pixels, "x", y_pixels)
    
    # Create the output raster dataset using the GTiff driver
    driver = gdal.GetDriverByName('GTiff')
    # Using Float32 to store the height values
    target_ds = driver.Create(output_file, x_pixels, y_pixels, 1, gdal.GDT_Float32)
    
    # Set the geotransform.
    # (top left x, pixel width, 0, top left y, 0, pixel height)
    # Note: The pixel height is negative as the origin is the top-left corner.
    geotransform = (xmin, resolution, 0, ymax, 0, -resolution)
    target_ds.SetGeoTransform(geotransform)
    
    # Set the projection from the vector layer to ensure CRS consistency
    spatial_ref = layer.GetSpatialRef()
    target_ds.SetProjection(spatial_ref.ExportToWkt())
    
    # Get the raster band and set no-data value to 0
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    
    # Rasterize the vector layer using the 'height' attribute
    gdal.RasterizeLayer(target_ds, [1], layer, options=["ATTRIBUTE=height"])
    
    # Flush changes and release dataset resources
    target_ds.FlushCache()
    del target_ds  # Release file handle so the file is not locked
    del vector_ds
    
    print(f"Rasterization complete. Output saved to {output_file}")
    return output_file

if __name__ == "__main__":
    # Define input and output paths
    input_file = "D:/UT-GLOBUS/Asia/Mumbai/Mumbai.gpkg"
    output_file = "rasterized_mumbai.tif"
    
    # Get grid spacing (resolution in meters) from the user
    try:
        resolution = float(input("Enter grid spacing (in meters): "))
    except ValueError:
        print("Invalid resolution input. Please enter a numerical value.")
        sys.exit(1)
    
    # Run the rasterization function
    rasterize_gpkg(input_file, output_file, resolution)