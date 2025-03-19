import os
from osgeo import gdal

def check_rasters(files):
    """
    Checks if all rasters in the 'files' list have the same dimensions,
    pixel size (from the geotransform), and CRS.
    
    Raises:
        ValueError: If any file's dimensions, pixel size, or CRS differ.
    """
    if not files:
        raise ValueError("No raster files provided.")

    # Open the first file as the reference
    ref_file = files[0]
    ds = gdal.Open(ref_file)
    if ds is None:
        raise FileNotFoundError(f"Could not open {ref_file}")
    ref_width = ds.RasterXSize
    ref_height = ds.RasterYSize
    ref_gt = ds.GetGeoTransform()  # (originX, pixelWidth, rot, originY, rot, pixelHeight)
    ref_pixel_width = ref_gt[1]
    ref_pixel_height = ref_gt[5]  # Note: typically a negative value
    ref_crs = ds.GetProjection()
    ds = None

    # Check each subsequent file against the reference
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

def create_tiles(infile, tilesize, tile_type):
    """
    Creates smaller raster chunks (tiles) from an input raster and saves them in a folder.
    
    The output file names follow the format:
        {base_name}_{tile_type}_{i}_{j}.tif
    where {base_name} is the input file's name without extension.
    
    The tiles are saved inside a folder named after the input file.
    
    Parameters:
        infile (str): Path to the input raster file.
        tilesize (int): Size of each tile (both width and height).
        tile_type (str): String to include in the output file names.
    """
    ds = gdal.Open(infile)
    if ds is None:
        raise FileNotFoundError(f"Could not open {infile}")

    width = ds.RasterXSize
    height = ds.RasterYSize

    # Create output folder based on the input file's base name
    infile_basename = os.path.splitext(os.path.basename(infile))[0]
    out_folder = infile_basename
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # If the tile size is greater than or equal to both dimensions,
    # create only one file that is the same as the original file.
    if tilesize >= width and tilesize >= height:
        outfile = os.path.join(out_folder, f"{infile_basename}_{tile_type}_0_0.tif")
        options = gdal.TranslateOptions(format='GTiff', srcWin=[0, 0, width, height])
        gdal.Translate(outfile, ds, options=options)
        print(f"Created single tile (original file): {outfile}")
        ds = None
        return

    # Loop over the raster and create tiles
    for i in range(0, width, tilesize):
        for j in range(0, height, tilesize):
            tile_width = min(tilesize, width - i)
            tile_height = min(tilesize, height - j)
            
            # Build the output filename in the respective folder
            outfile = os.path.join(out_folder, f"{tile_type}_{i}_{j}.tif")
            
            # Define translation options to create a tile using a subset (srcWin)
            options = gdal.TranslateOptions(format='GTiff', srcWin=[i, j, tile_width, tile_height])
            gdal.Translate(outfile, ds, options=options)
            print(f"Created tile: {outfile}")
    
    ds = None

if __name__ == "__main__":
    # List of input raster files
    raster_files = ["Building_DSM.tif", "DEM.tif", "Trees.tif"]
    
    # Check that all rasters have matching dimensions, pixel size, and CRS.
    try:
        check_rasters(raster_files)
    except ValueError as error:
        print(error)
        exit(1)

    # Tile size (in pixels) can be adjusted as needed
    tilesize = 3600
    
    # Process each raster file
    for raster in raster_files:
        # Use the base name (without extension) as the tile type
        tile_type = os.path.splitext(os.path.basename(raster))[0]
        create_tiles(raster, tilesize, tile_type)
