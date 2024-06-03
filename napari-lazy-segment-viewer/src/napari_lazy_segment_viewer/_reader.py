"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np
import openslide
from openslide import deepzoom
import dask
import dask.array as da
import napari as npr
from stardist.models import StarDist2D
from csbdeep.utils import normalize

model = StarDist2D.from_pretrained('2D_versatile_fluo')

def rgb2hed_custom(rgb_image):
    hed_from_rgb = np.linalg.inv(np.array([[0.65, 0.70, 0.29],
                                           [0.07, 0.99, 0.11],
                                           [0.27, 0.57, 0.78]]))
    rgb_array = np.array(rgb_image, dtype=np.float32) / 255.0
    np.maximum(rgb_array, 1e-6, out=rgb_array)
    log_adjust = np.log(1e-6)
    stains = (np.log(rgb_array) / log_adjust) @ hed_from_rgb
    np.maximum(stains, 0, out=stains)
    return stains

def process_tile(column, row, level, gen):

    tile = np.array(gen.get_tile(level, (column, row))).transpose((1, 0, 2))

    hed_tile = rgb2hed_custom(tile)
   
    labels= get_mask(hed_tile[:,:,0])
 
    # Create an empty 3D array with the same height and width, but with 3 channels
    # result = np.zeros(tile.shape)
 
    # Process the Hematoxylin channel and set it in the result array
    mask = (labels > 0).astype(np.uint8)
    # result[:, :, 0] = labels  # Set only the first channel with labels
 
    return mask*255
 
 
def get_mask(channel):
    

 
    # creates a pretrained model
    
   
    # np.save('channel.npy', channel)
    
    n_tiles = model._guess_n_tiles(channel)
    print(channel.shape)
    labels, _ = model.predict_instances(normalize(channel), n_tiles = n_tiles)
   
    return labels

def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    if not path.endswith(".svs"):
        return None

    
    # otherwise we return the *function* that can read ``path``.
    return reader_function


def svs2dask_array(
    svs_file,
    tile_size=256,
    overlap=0,
    remove_last=False,
    allow_unknown_chunksizes=False,
):
    """Convert SVS, TIF or TIFF to dask array.
    Parameters
    ----------
    svs_file : str
            The path to the image file.
    tile_size : int, optinal
            The size of the chunk to be read in.
    overlap : int, optional
            Do not modify, overlap between neighboring tiles.
    remove_last : bool, optional
            Whether to remove the last tile because it has a custom size.
    allow_unknown_chunksizes : bool, optional
            Whether to allow different chunk sizes. If True, flexibility
            increases, but this method becomes slower. The default is False.
    Returns
    -------
    pyramid: List[dask.array.Array]
            A list of Dask Array representing the contents of the image file.
    """
    img = openslide.open_slide(svs_file)
    gen = deepzoom.DeepZoomGenerator(
        img, tile_size=tile_size, overlap=overlap, limit_bounds=True
    )
    pyramid = []
    max_level = len(gen.level_dimensions)
    
    segment = dask.delayed(process_tile)
    
    for level in range(1,max_level):
        n_tiles_x, n_tiles_y = gen.level_tiles[level]

        print(f"Reading level {level} with {n_tiles_x}x{n_tiles_y} tiles")
        @dask.delayed(pure=True)
        def get_tile(level, column, row):
            tile = gen.get_tile(level, (column, row))
            return np.array(tile).transpose((1, 0, 2))
        
        
    
        sample_tile_shape = get_tile(level, 0, 0).shape.compute()
        rows = range(n_tiles_y - (0 if not remove_last else 1))
        cols = range(n_tiles_x - (0 if not remove_last else 1))
    
        print(f"Reading level {level} with {rows}x{cols} tiles")

        if level == max_level-1:

            label = da.concatenate(
            [
                da.concatenate(
                    [
                        da.from_delayed(
                            segment(col, row, level, gen), sample_tile_shape[:2], np.uint8
                        )
                        for row in rows
                    ],
                    allow_unknown_chunksizes=allow_unknown_chunksizes,
                    axis=1,
                )
                for col in cols
            ],
            allow_unknown_chunksizes=allow_unknown_chunksizes,
            ).transpose([1, 0])
        
            
        arr = da.concatenate(
            [
                da.concatenate(
                    [
                        da.from_delayed(
                            get_tile(level, col, row), sample_tile_shape, np.uint8
                        )
                        for row in rows
                    ],
                    allow_unknown_chunksizes=allow_unknown_chunksizes,
                    axis=1,
                )
                for col in cols
            ],
            allow_unknown_chunksizes=allow_unknown_chunksizes,
        ).transpose([1, 0, 2])
        
        print(f"Reading level {level} with {arr.shape} tiles")
    
        pyramid.insert(0, arr)
    print(label.shape)
    return pyramid, label

def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    """Take a path or list of paths and return a list of LayerData tuples."""
    paths = [path] if isinstance(path, str) else path
    layer_data = []


    
    for _path in paths:
        if _path.endswith('.svs'):
            pyramid, label = svs2dask_array(_path)
            # label_delayed = [dask.delayed(process_tile)(image) for image in pyramid]
            # label_array = [da.from_delayed(delayed_obj, shape=image.shape[:2], dtype=int) for delayed_obj, image in zip(label_delayed, pyramid)]
            labels = [da.zeros(arr.shape[:2]) for arr in pyramid]
            labels[0] = label
            add_kwargs = {}
            layer_type = "image"  # optional, default is "image"
            layer_data.append((pyramid, add_kwargs, layer_type))
            layer_data.append((labels, {'name':'label','opacity': 0.8, 'blending':'additive','colormap':'red'}, layer_type))
    
    return layer_data