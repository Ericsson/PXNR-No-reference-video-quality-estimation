import math
import numpy as np
from skimage import transform


__author__    = "Sebastian Emmot"
__contact__   = "david.lindero@ericsson.com"
__copyright__ = "Copyright (c) 2018 Ericsson"
__date__      = "2018-04-10"
__status__    = "ITU-Submission ready"


"""
This module contains two functions that deal with the slicing and scaling to subframes.

matrix_quadslice() returns smaller slices of the larger image

image_quadification() does the complete scaling and splitup of the complete full-size
input image.

"""


def matrix_quadslice(matrix, depth=1, axis_height=0, axis_width=1):
    '''
    @depth : the level of subdivision, will result in 4^depth quad-slices
    @axis_heigh/width : which axis represent height and width dimensions

    Returns a list of x evenly sliced quad submatrices from input matrix.
    '''
    slabs = np.split(matrix, pow(2, depth), axis=axis_width)  # split the matrix into rows.
    sub_matrices = np.array_split(slabs[0], pow(2, depth), axis=axis_height)  # splits the first slab to have somehting to append.

    slabs = slabs[1:]  # removes first slab to avoid data duplication.

    for slab in slabs:   # loop over each row in the matrix and chops them up.
        sub_matrices += np.array_split(slab, pow(2, depth), axis=axis_height)

    return sub_matrices


def image_quadification(nd_image, depth=1, color_space='rgb', image_dims=(216, 384), time=0):
    ''' Takes an image(np array) and subdivide into quads to recursive depth specified (default 1)
    e.g depth = 1,2,3 gives (1+4),(1+4+16),(1+4+16+64) quads respectively,
    formula: num of quads = sum(4^k) for k in [0,n], where n is depth.

    @color_space : What color space the image is in, 'rgb' assumed
    @image_dims  : The desired height x width dimesions of output image array.

    Returns a list of all quads resized to @image_dims.
    '''
    dtype = np.float32
    if color_space != 'rgb':
        raise NotImplementedError
    else:
        # resizing to desired img dims, forcing dtype since resize will transform with float64.
        resized = transform.resize(nd_image, image_dims, mode='constant', preserve_range=True).astype(dtype)

        frames = int(sum(math.pow(4, k) for k in range(0, depth+1)))
        resized_stack = np.empty(shape=(frames,)+image_dims+(3,))

        count = 0
        resized_stack[count] = resized
        resized = []
        count += 1

        for depth in range(1, depth+1):
            slice_list = matrix_quadslice(nd_image, depth=depth, axis_height=0, axis_width=1)

            for slc in slice_list:
                #  enforcing default mode='constant' to futureproof it (see skimage 0.15 doc).
                slc_resized = transform.resize(slc, image_dims, mode='constant', preserve_range=True).astype(dtype)
                resized_stack[count] = slc_resized
                count += 1
            slice_list = []

        resized_stack = np.asarray(resized_stack)/255
        return resized_stack
