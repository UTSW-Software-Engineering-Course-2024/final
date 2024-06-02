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
    return pyramid



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

def read_svs_slide(file_path):
    """Read a SVS slide using openslide

    Parameters
    ----------
    file_path : str
        Path to the SVS file

    Returns
    -------
    myPyramid : list
        List of numpy arrays representing the slide pyramid
    """
    slide = openslide.OpenSlide(file_path)
    myPyramid = []

    for level in range(slide.level_count):
        img = slide.read_region((0, 0), level, slide.level_dimensions[level])
        myPyramid.append(np.array(img))

    return myPyramid




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
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path

    layer_data = []
    for _path in paths:
        if _path.endswith('.svs'):
            data = svs2dask_array(_path)
            add_kwargs = {}
            layer_type = "image"  # optional, default is "image"
            layer_data.append((data, add_kwargs, layer_type))

    return layer_data
