import numpy as np
import tensorflow as tf
from helperfunctions import frame_extraction, get_lr_metric
from videosources import YUVSourceFFmpeg

__author__    = "Sebastian Emmot, Maciej Pedzisz, David Lindero, Samuel Chambers"
__contact__   = "david.lindero@ericsson.com"
__copyright__ = "Copyright (c) 2018 Ericsson"
__date__      = "2018-04-10"
__status__    = "ITU-Submission ready"


"""
Setting up constants
"""
video = './BigBuckBunny_8s_clip.mp4'
image_model_location = "./model_coefficients/img_model-ep01-vloss40.883.hdf5"
video_model_location = "./model_coefficients/noref_mostrained.hdf5"


"""
Sample variables, should be extracted from input video file. One way to get this
info is to run ffprobe on the video file first and parse the output manually.
"""
fps = 30  # frames per second
full_pix_fmt = 'yuv422p'  # pixel format
video_duration = 8.0  # in seconds

depth = 8  # bits per pixel/channel
if '10le' in full_pix_fmt:
    depth = 10
short_pixfmt = full_pix_fmt.split('p')[0][-3:]  # split 420 from yuv420p


"""
Running Model 1, image model
"""
optimizer = tf.keras.optimizers.Adadelta()
lr_rate_metric = get_lr_metric(optimizer)
mapping = {"lr": lr_rate_metric}

image_model = tf.keras.models.load_model(image_model_location, custom_objects=mapping)
nr_frames_to_extract = 96

predictions = []
vsource = YUVSourceFFmpeg(video, 3840, 2160, fps, depth, short_pixfmt, True)

nr_of_batches = 1  # this code was prepared to run for several batches as well, this is a remnant of that
for ii in range(0, nr_of_batches):
    frames = frame_extraction(vsource, fps, video_duration, depth, short_pixfmt, nr_frames_to_extract)
    currpred = image_model.predict(frames, batch_size=8, verbose=0)

    predictions.extend(currpred)

    frames = []
predictions = np.reshape(predictions, (nr_of_batches, nr_frames_to_extract, 85, 1))


"""
Running Model 2, aggregation model
"""
video_model = tf.keras.models.load_model(video_model_location)

all_video_predictions = []
for ii in range(0, nr_of_batches):  # once again, several batches.
    curr_pred = np.reshape(predictions[ii], (1, nr_frames_to_extract, 85, 1))
    all_video_predictions.append(video_model.predict(curr_pred, verbose=1))

final_pred = all_video_predictions[0]
print('Final prediction of {video} is \nMOS: {final_pred}'.format(**locals()))
