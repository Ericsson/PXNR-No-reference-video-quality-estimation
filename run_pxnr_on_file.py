import numpy as np
import pickle
import tensorflow as tf
from helperfunctions import frame_extraction, get_lr_metric
from videosources import YUVSourceFFmpeg
import h5py
import sys
import re
import os
import subprocess
import uuid
import time


__author__    = "Sebastian Emmot, Maciej Pedzisz, David Lindero, Samuel Chambers, Rohit Rentala"
__contact__   = "david.lindero@ericsson.com"
__copyright__ = "Copyright (c) 2018 Ericsson"
__date__      = "2018-04-10"
__status__    = "New runtime-method, updated to Tensorflow 2.16"


def extract_framerate(filename):
    match = re.search(r"_(\d+)hz", filename)
    if match:
        framerate = int(match.group(1))
    else:
        framerate = 60
    return framerate


def shell_call(cmd, raw=True):
    """
    Run a command and return output of (returncode, stdout, sterr)  as result.

    Arguments:
        - cmd: string or list of command parts in case of raw=False
        - raw: if true, interpret string as raw command
    """
    try:
        # print(cmd)
        x = subprocess.run(cmd, shell=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ret = (x.returncode, str(x.stdout, "utf-8"), str(x.stderr, "utf-8"))
        return ret
    except SubprocessError:
        print("System error running command: " + str(cmd))
        sys.exit(1)


def main():

    if len(sys.argv) > 1:
        pvs_path = sys.argv[1]
        print('evaluating PVS: {pvs_path}\n'.format(**locals()))
    else:
        print('Error: No input file')
        quit()


    """
    Setting up constants
    """
    # video = './BigBuckBunny_8s_clip.mp4'
    video = pvs_path
    video_model_location = "./model_coefficients/video_model.h5"
    output_path = './'

    """
    Sample variables, should be extracted from input video file. One way to get this
    info is to run ffprobe on the video file first and parse the output manually.
    """
    fps = extract_framerate(os.path.basename(video))

    full_pix_fmt = 'yuv420p10le'  # pixel format
    full_pix_fmt = 'yuv422p'
    video_duration = 5.0  # in seconds

    depth = 8  # bits per pixel/channel
    if '10le' in full_pix_fmt:
        depth = 10
    short_pixfmt = full_pix_fmt.split('p')[0][-3:]  # split 420 from yuv420p

    """
    Running Model 1, image model
    """
    nr_frames_to_extract = 96
    predictions = []

    uuid_value = str(uuid.uuid4()).split('-')[0]
    
    # tmp_pvs_name = f'/tmp/{pvs_base_name}_{uuid_value}.mp4'

    tempvideopath = f'/tmp/tempvideopath_{uuid_value}.mp4'
    # cmd_line = f'ffmpeg -s 3840x2160 -r {fps} -pix_fmt {full_pix_fmt} -i {video} -c:v libx265 -x265-params lossless=1 -r {fps} {tempvideopath}'
    cmd_line = f'ffmpeg -i {video} -c:v libx265 -x265-params lossless=1 -r {fps} {tempvideopath}'
    print(cmd_line)
    dec_ret, dec_out, dec_err = shell_call(cmd_line)
    print(dec_ret, dec_out, dec_err)

    starttime_model = time.time()
    vsource = YUVSourceFFmpeg(tempvideopath, 3840, 2160, fps, depth, short_pixfmt, True)

    nr_of_batches = 1  # this code was prepared to run for several batches as well, this is a remnant of that
    frame_predictions = frame_extraction(vsource, fps, video_duration, depth, short_pixfmt, nr_frames_to_extract)

    """
    Running Model 2, aggregation model
    """
    video_model = tf.keras.models.load_model(video_model_location, compile=False)
    final_pred = video_model.predict(frame_predictions, verbose=1)

    print('Final ouput of {video} is \nMOS: {final_pred}'.format(**locals()))
    output_file = output_path + os.path.basename(video) + '.txt'
    with open(output_file, "w+") as file:
        file.write(f'{video}: {final_pred}\n')

    endtime_model = time.time() - starttime_model
    print(f'Time: {endtime_model}s')


if __name__ == "__main__":
    main()

