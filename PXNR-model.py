import numpy as np
import pickle
import tensorflow as tf
from helperfunctions import frame_extraction, get_lr_metric
from videosources import YUVSourceFFmpeg
import h5py

__author__    = "Sebastian Emmot, Maciej Pedzisz, David Lindero, Samuel Chambers, Rohit Rentala"
__contact__   = "david.lindero@ericsson.com"
__copyright__ = "Copyright (c) 2018 Ericsson"
__date__      = "2018-04-10"
__status__    = "New runtime-method, updated to Tensorflow 2.16"


"""
Setting up constants
"""
video = './BigBuckBunny_8s_clip.mp4'
# image_model_location = "./model_coefficients_up/img_model-ep01-vloss40.883.hdf5"
# video_model_location = "./model_coefficients_up/noref_mostrained.hdf5"
video_model_location = "./model_coefficients/video_model.h5"


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
nr_frames_to_extract = 96

predictions = []

vsource = YUVSourceFFmpeg(video, 3840, 2160, fps, depth, short_pixfmt, True)

nr_of_batches = 1  # this code was prepared to run for several batches as well, this is a remnant of that
frame_predictions = frame_extraction(vsource, fps, video_duration, depth, short_pixfmt, nr_frames_to_extract)

"""
Running Model 2, aggregation model
"""
video_model = tf.keras.models.load_model(video_model_location, compile=False)
final_pred = video_model.predict(frame_predictions, verbose=1)

print('Final ouput of {video} is \nMOS: {final_pred}'.format(**locals()))

