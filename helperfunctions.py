import math
import numpy as np
import matrix_utils


__author__    = "Sebastian Emmot, Maciej Pedzisz, David Lindero, Samuel Chambers"
__contact__   = "david.lindero@ericsson.com"
__copyright__ = "Copyright (c) 2018 Ericsson"
__date__      = "2018-04-10"
__status__    = "ITU-Submission ready"


"""
This module contains three helper functions;

get_lr_metric() returns the metric for the image model.

yuv2RGB() inputs a yuv image and returns a RGB image. It can handle a number of conversions,
yuv411, 420, 410, 422, 444 and 440 to 8 bit rgb. This conversion scales 10-bit per
pixel input to 8-bit, performing the conversation in this space.

frame_extraction() extracts 96 frames from the video_object input, essentially a video
stream. This function, together with the image_quadification function from matrix_utils,
also takes care of the scaling and splitting up of the full-resolution frames into subframes.
"""


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


def yuv2RGB(yuv, depth, pixfmt):
    try:
        y, u, v = yuv
    except TypeError:
        # print('Type error too!')
        return False

    pixfmt = int(pixfmt)

    multdict = {411: (1, 4), 420: (2, 2), 410: (2, 4), 422: (1, 2), 444: (1, 1), 440: (2, 1)}

    y = np.array(y, dtype=np.int64)
    u = np.array(u, dtype=np.int64)
    v = np.array(v, dtype=np.int64)

    if depth == 10:
        # scale to 8 bit
        y = (y+2)/4
        u = (u+2)/4
        v = (v+2)/4

    u = u.repeat(multdict[pixfmt][0], axis=0).repeat(multdict[pixfmt][1], axis=1)
    v = v.repeat(multdict[pixfmt][0], axis=0).repeat(multdict[pixfmt][1], axis=1)

    B = 1.164 * (y-16) + 2.018 * (u - 128)
    G = 1.164 * (y-16) - 0.813 * (v - 128) - 0.391 * (u - 128)
    R = 1.164 * (y-16) + 1.596*(v - 128)

    R = R.clip(min=0.0, max=255.0)
    G = G.clip(min=0.0, max=255.0)
    B = B.clip(min=0.0, max=255.0)

    pix = np.dstack((R.astype(np.int32), G.astype(np.int32), B.astype(np.int32)))

    return pix


def frame_extraction(video_object, extracted_fps, video_duration, depth, pixfmt, nr_frames_to_extract=96):
    image_dims = (270, 480)
    quad_depth = 3
    num_slice_frames = int(sum(math.pow(4, k) for k in range(0, quad_depth+1)))
    total_number_of_frames = int(extracted_fps * video_duration)
    divisor = int(total_number_of_frames/nr_frames_to_extract)

    # Selecting which frames to use
    extraction_pattern = [ii % divisor for ii in range(0, total_number_of_frames)]
    framesNeeded = True
    frameiter = 0
    nr_added_frames = 0

    result = []
    while framesNeeded:
        yuv = video_object.loadFrame()

        if extraction_pattern[frameiter] == 0:
            # Going from yuv to rgb
            rgb = yuv2RGB(yuv, depth, pixfmt)
            yuv = []
            # Creating all subframes for this full-size-image
            result.extend(matrix_utils.image_quadification(rgb, quad_depth, "rgb", image_dims))
            rgb = []
            nr_added_frames += 1

        frameiter += 1
        if nr_added_frames == 96:
            yuv = []
            framesNeeded = False
            break
    result = np.reshape(np.asarray(result), newshape=(nr_added_frames*num_slice_frames, 270, 480, 3))

    return result
