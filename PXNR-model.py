#!/usr/bin/env python3
"""
PXNR no-reference video quality estimation.

Run the model on a single video file and print the predicted MOS as JSON on
standard output. All progress and diagnostic messages go to standard error, so
the standard output stream stays machine-readable (e.g. it can be piped into a
file or another tool).

Usage:
    python3 PXNR-model.py path/to/video.mp4
"""
import argparse
import json
import os
import re
import sys

__author__    = "Sebastian Emmot, Maciej Pedzisz, David Lindero, Samuel Chambers, Rohit Rentala"
__contact__   = "david.lindero@ericsson.com"
__copyright__ = "Copyright (c) 2018 Ericsson"
__date__      = "2018-04-10"
__status__    = "New runtime-method, updated to Tensorflow 2.18"


def eprint(*args, **kwargs):
    """Print to standard error, keeping standard output reserved for JSON."""
    print(*args, file=sys.stderr, **kwargs)


def extract_framerate(filename):
    """Guess the frame rate from a '..._NNhz...' filename, defaulting to 60."""
    match = re.search(r"_(\d+)hz", filename)
    if match:
        return int(match.group(1))
    return 60


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Estimate the no-reference video quality (MOS) of a video "
                    "file. The result is printed as JSON on standard output; "
                    "all progress and diagnostic messages go to standard error.")
    parser.add_argument(
        "input",
        help="path to the input video file to evaluate")
    parser.add_argument(
        "--fps", type=float, default=None,
        help="frame rate to sample the video at; by default this is taken from "
             "a '_NNhz' marker in the filename, falling back to 60")
    parser.add_argument(
        "--duration", type=float, default=8.0,
        help="duration in seconds to analyse (default: 8.0)")
    parser.add_argument(
        "--pix-fmt", default="yuv422p",
        help="pixel format used for decoding (default: yuv422p)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Resolve the input to an absolute path before anything else, then run from
    # the script's own directory so the relative model_coefficients/* paths
    # resolve no matter where the command was invoked from.
    video = os.path.abspath(args.input)
    if not os.path.isfile(video):
        eprint("Error: input file not found: {}".format(args.input))
        return 1
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Keep TensorFlow's own logging on stderr and a little quieter. Must be set
    # before TensorFlow is imported.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    # Heavy imports are deferred until after argument parsing and input
    # validation so that --help and a missing-file error stay fast and don't
    # load TensorFlow or the model coefficients.
    import tensorflow as tf
    from helperfunctions import frame_extraction
    from videosources import YUVSourceFFmpeg

    fps = args.fps if args.fps is not None else extract_framerate(os.path.basename(video))
    full_pix_fmt = args.pix_fmt
    video_duration = args.duration

    # 10-bit input (e.g. yuv420p10le) is scaled back to 8 bit during conversion.
    depth = 10 if "10le" in full_pix_fmt else 8
    short_pixfmt = full_pix_fmt.split("p")[0][-3:]  # e.g. 422 from yuv422p

    video_model_location = "./model_coefficients/video_model.h5"
    nr_frames_to_extract = 96

    eprint("evaluating {} at {} fps over {}s".format(video, fps, video_duration))

    # The input is decoded directly and scaled to 4K by YUVSourceFFmpeg, which
    # runs the image (per-frame) model. A previous version re-encoded the input
    # to lossless x265 first, but decoding that re-encode yields the exact same
    # frames, so it only cost time, disk and encoder memory.
    vsource = YUVSourceFFmpeg(video, 3840, 2160, fps, depth, short_pixfmt, True)
    frame_predictions = frame_extraction(
        vsource, fps, video_duration, depth, short_pixfmt, nr_frames_to_extract)

    # Running the aggregation (video) model on the per-frame predictions.
    video_model = tf.keras.models.load_model(video_model_location, compile=False)
    final_pred = video_model.predict(frame_predictions, verbose=0)
    mos = float(final_pred.reshape(-1)[0])

    eprint("MOS: {}".format(mos))

    result = {
        "input": args.input,
        "fps": fps,
        "duration": video_duration,
        "mos": mos,
    }
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
