import os
import sys
import signal
import subprocess
import numpy as np


__author__    = "Sebastian Emmot, Maciej Pedzisz, David Lindero, Samuel Chambers"
__contact__   = "david.lindero@ericsson.com"
__copyright__ = "Copyright (c) 2018 Ericsson"
__date__      = "2018-04-10"
__status__    = "ITU-Submission ready"


"""
There are four video I/O classes in this module:

1) Base video source class to be extended via inheritance by other concrete classes.

2) Class for reading raw data from YUV video files. It supports: uncompressed YUV
video files and random access to YCbCr data stored in a planar (unpacked) format,
three chroma sampling schemes and up to 16 bits integer pixels.

3) Class for reading YUV data from compressed video files. It uses 'ffmpeg' for
reading and decoding compressed videos and passes raw frame data via pipes. All
file formats and codecs supported by 'ffmpeg' can be read and decoded.

4) Class extending the functionality of the above classes to support multi-scale
video processing. It supports dyadic resolution pyramids only.
"""


class VideoSource:
    def __init__(self, filename, width, height, fps, depth, chroma):
        """Creates a base video source object that other classes should extend via inheritance."""

        if os.path.exists(filename):
            if not os.access(filename, os.R_OK):
                print("\nError:", filename, "can't be read.\n")
                sys.exit(1)
        else:
            print("\nError:", filename, "doesn't exist.\n")
            sys.exit(1)

        self.width  = width
        self.height = height
        self.fps    = fps
        self.depth  = depth
        self.chroma = chroma

        div_dict = {411: (1, 4), 420: (2, 2), 410: (2, 4), 422: (1, 2), 444: (1, 1), 440: (2, 1)}
        divisors = div_dict[int(self.chroma)]

        self._chromaWidth  = self.width // divisors[1]
        self._chromaHeight = self.height // divisors[0]

        self._lumaSize   = width * height
        self._chromaSize = self._chromaWidth * self._chromaHeight
        self._frameSize  = self._lumaSize + 2 * self._chromaSize

        if self.depth > 8:
            self._frameSizeInBytes = 2 * self._frameSize
        else:
            self._frameSizeInBytes = self._frameSize

        self.curFrame = 0

    def reshapeFrame(self, rawData):
        """Splits and reshapes frame data into individual colour channels."""

        y = rawData[0:self._lumaSize].reshape(self.height, self.width)

        offset = self._lumaSize
        u = rawData[offset:offset + self._chromaSize].reshape(self._chromaHeight, self._chromaWidth)

        offset += self._chromaSize
        v = rawData[offset:offset + self._chromaSize].reshape(self._chromaHeight, self._chromaWidth)

        return y, u, v

    def loadFields(self, frame=None):
        """Loads requested (or next) video fields as separate colour channels."""

        channels = self.loadFrame(frame)

        if channels is None:
            return None

        y = channels[0]
        u = channels[1]
        v = channels[2]

        topField    = y[0::2], u[0::2], v[0::2]
        bottomField = y[1::2], u[1::2], v[1::2]

        return topField, bottomField


class YUVSource(VideoSource):
    def __init__(self, filename, width, height, fps=25, depth=8, chroma=420):
        """Opens raw YUV video file specified by the input parameters and prepares it for reading."""

        super().__init__(filename, width, height, fps, depth, chroma)

        try:
            self._file = open(filename, 'rb')
            fileSize   = os.path.getsize(filename)
        except IOError as e:
            print("\nI/O error ({}): {}\n".format(e.errno, e.strerror))
            sys.exit(1)

        self.numFrames = fileSize // self._frameSizeInBytes

    def __del__(self):
        """Closes YUV video file if it was properly opened."""

        if hasattr(self, "_file"):
            if self._file:
                self._file.close()

    def loadFrame(self, frame=None):
        """Loads requested (or next) video frame as separate colour channels."""

        if frame is not None:
            if (frame < 1) or (frame > self.numFrames):
                print("\nError: frame number must be positive and not exceeding the number of frames in the input file.\n")
                sys.exit(1)

            self.curFrame = frame - 1
            self._file.seek(self.curFrame * self._frameSizeInBytes, os.SEEK_SET)

        if self.curFrame == self.numFrames:
            return None

        rawData = np.fromfile(self._file,
                              dtype=np.uint16 if self.depth > 8 else np.uint8,
                              count=self._frameSize,
                              sep="")

        self.curFrame += 1

        return self.reshapeFrame(rawData)


class YUVSourceFFmpeg(VideoSource):
    def __init__(self, filename, width, height, fps=25, depth=8, chroma=420, debug=False):
        """Opens compressed video file specified by the input parameters and prepares it for reading."""

        super().__init__(filename, width, height, fps, depth, chroma)

        pix_fmt = 'yuv{}p'.format(self.chroma)
        if self.depth > 8:
            pix_fmt += '{}le'.format(self.depth)

        self._debug = debug
        if self._debug:
            print("YUVSourceFFmpeg(): filename='{filename}', width={width}, height={height}, pix_fmt={pix_fmt}, depth={depth}, fps={fps}".format(
                **locals()))

        # rescale/convert frame rate using allowed FFmpeg parameters for pre-processing in pixel-based models
        vf_params = []
        vf_params.append('scale=w={0}:h={1}'.format(width, height))
        vf_params.append('fps={0}'.format(fps))

        command = ['ffmpeg']
        command.extend(['-i', filename])
        if vf_params:
            command.extend(['-filter:v', ','.join(vf_params)])
        command.extend(['-f', 'image2pipe'])
        command.extend(['-c:v', 'rawvideo'])
        command.extend(['-pix_fmt', pix_fmt])
        command.extend(['-'])

        if self._debug:
            print("YUVSourceFFmpeg(): " + ' '.join(command))

        try:
            on_posix = 'posix' in sys.builtin_module_names
            self._process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=(10**8), close_fds=on_posix)
        except OSError as e:
            if e.errno == os.errno.ENOENT:
                print("\nI/O error calling 'ffmpeg' to decode input video file/stream.\n")
                sys.exit(1)
            else:
                raise

        self.numFrames = 0

    def __del__(self):
        """Terminates 'ffmpeg' process if it was properly created."""

        if hasattr(self, "_process"):
            if self._process:
                self._process.stdout.close()
                self._process.send_signal(signal.SIGTERM)
                self._process.wait()

    def loadFrame(self):
        """Loads next video frame as separate colour channels."""

        rawByteString = self._process.stdout.read(self._frameSizeInBytes)
        self._process.stdout.flush()

        if len(rawByteString) < self._frameSizeInBytes:
            return None

        rawData = np.fromstring(rawByteString, dtype=np.uint16 if self.depth > 8 else np.uint8)
        self.curFrame += 1
        return self.reshapeFrame(rawData)
