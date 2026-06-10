import math
import numpy as np
import matrix_utils
import tensorflow as tf
import time
import h5py

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


image_model_location = "./model_coefficients/image_model.h5"
optimizer = tf.keras.optimizers.Adadelta()
lr_rate_metric = get_lr_metric(optimizer)
mapping = {"lr": lr_rate_metric}
image_model = tf.keras.models.load_model(image_model_location, compile=False, custom_objects=mapping)

# How many sub-frames to push through the image model at once. The convolutional
# activations scale with this, and TensorFlow keeps its allocation pool at the
# high-water mark, so this is the main knob for peak inference memory. 8 was the
# old hard-coded value; a smaller batch trades a little speed for a lot less RAM.
import os
IMG_BATCH = int(os.environ.get("PXNR_IMG_BATCH", "4"))


def _predict_image(frames):
    """Run the image model over `frames` (N, 270, 480, 3).

    Keras' predict() runs the forward pass inside an optimised tf.function that
    releases convolutional activations between batches far better than repeated
    eager model() calls do, so it both runs faster and peaks much lower. The
    only knob we expose is the batch size, which caps how much activation memory
    is live at once.
    """
    return image_model.predict(frames, batch_size=IMG_BATCH, verbose=0)


def yuv2RGB(yuv, depth, pixfmt):
    try:
        y, u, v = yuv
    except TypeError:
        # print('Type error too!')
        return False

    pixfmt = int(pixfmt)

    multdict = {411: (1, 4), 420: (2, 2), 410: (2, 4), 422: (1, 2), 444: (1, 1), 440: (2, 1)}

    # All the arithmetic below is done in float32 rather than the previous
    # int64/float64. The pixel values are small (0..255, or 0..1023 for 10-bit)
    # so float32 has ample precision, and it halves the working set for the
    # full-resolution colour planes, which dominate peak memory at 4K.
    y = np.asarray(y, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    if depth == 10:
        # scale to 8 bit
        y = (y + 2) / 4
        u = (u + 2) / 4
        v = (v + 2) / 4

    u = u.repeat(multdict[pixfmt][0], axis=0).repeat(multdict[pixfmt][1], axis=1)
    v = v.repeat(multdict[pixfmt][0], axis=0).repeat(multdict[pixfmt][1], axis=1)

    yt = y - 16.0
    ut = u - 128.0
    vt = v - 128.0

    B = 1.164 * yt + 2.018 * ut
    G = 1.164 * yt - 0.813 * vt - 0.391 * ut
    R = 1.164 * yt + 1.596 * vt

    # Write straight into a uint8 image (values are integers in 0..255 after the
    # clip, so this is lossless) instead of building an int32 dstack. That is a
    # 4x smaller output array and avoids three int32 temporaries.
    pix = np.empty(R.shape + (3,), dtype=np.uint8)
    pix[..., 0] = np.clip(R, 0.0, 255.0)
    pix[..., 1] = np.clip(G, 0.0, 255.0)
    pix[..., 2] = np.clip(B, 0.0, 255.0)

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

    frame_predictions = []
    beforetime = time.time()
    result = []
    while framesNeeded:
        yuv = video_object.loadFrame()

        if extraction_pattern[frameiter] == 0:
            # Going from yuv to rgb
            rgb = yuv2RGB(yuv, depth, pixfmt)
            yuv = []
            # Creating all subframes for this full-size-image
            currquad = matrix_utils.image_quadification(rgb, quad_depth, "rgb", image_dims)
            rgb = []
            reshaped_currquad = np.reshape(currquad, newshape=(num_slice_frames, 270, 480, 3))
            currquad = None
            predictions_for_currquad = _predict_image(reshaped_currquad)
            reshaped_currquad = None
            frame_predictions.extend(predictions_for_currquad)
            nr_added_frames += 1
            # print('per frame time:')
            # print(time.time()-beforetime)
            beforetime = time.time()

        frameiter += 1
        if nr_added_frames == nr_frames_to_extract:
            yuv = []
            framesNeeded = False
            break

    frame_pred_results = np.reshape(frame_predictions, (1, nr_frames_to_extract, 85, 1))

    return frame_pred_results
