# PXNR-No-reference-video-quality-estimation
The code herein is a sample implementation of the Ericsson submission to the ITU-T SG12 P.NATS phase 2 competition for a no-reference pixel based video quality estimation module.  

# License

See the ./licenses/ directory for individual license files for this software together with used and referenced libraries and modules. 

# Usage

First, install the prerequisites. Python3, FFmpeg and the necessary libraries and modules are listed below.  

Uncompress the 7zip-archive img_model-ep01-vloss40.883.hdf5.7z.001 and .002 of the model coefficients located in the model_coefficients-directory. The resulting file, img_model-ep01-vloss40.883.hdf5, should be located next to the 7z-archive for the code to work. 

After you have verified that everything has been installed properly, open a command line terminal on your platform and browse to the directory containing the
code. The model will run on the supplied .mp4-file and will produce an output that looks like this, together with some warnings depending on your platform etc:  


```
$ python3 PXNR-model.py

2019-11-19 11:26:21.772936: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
YUVSourceFFmpeg(): filename='./BigBuckBunny_8s_clip.mp4', width=3840, height=2160, pix_fmt=yuv422p, depth=8, fps=30
YUVSourceFFmpeg(): ffmpeg -i ./BigBuckBunny_8s_clip.mp4 -filter:v scale=w=3840:h=2160,fps=30 -f image2pipe -c:v rawvideo -pix_fmt yuv422p -
/lib/python3.5/site-packages/skimage/transform/_warps.py:110: UserWarning: Anti-aliasing will be enabled by default in skimage 0.15 to avoid aliasing artifacts when down-sampling images.
  warn("Anti-aliasing will be enabled by default in skimage 0.15 to "
1/1 [==============================] - 0s
Final prediction of ./BigBuckBunny_8s_clip.mp4 is  
MOS: [[2.515198]]
```

If you want to run it on another video file, add the path and proper sample variables in the `PXNR-model.py` file. 

# Prerequisites

This code has been run, as is, on Mac OS and Ubuntu, producing identical results. FFmpeg, Python and associated libraries have not been included here in this repository and must be installed separately. The model itself can work with other decoders as long as the pictures from the video can be read as arrays using yuv or rgb pixel formats. The user has to modify the code on their own if the use case is anything other than what has been shown in the included example. 

### Python

Python3 has been used when developing and running the models herein. The version supplied with the model in the competition was 3.5, but this has been tested with both 3.6 and 3.8 as well giving the same results. The updated tensorflow 2 version has only been tested with 3.8. 
  
These are the modules and versions that were used when running the code.  
  
* scikit-image==0.18.3
* tensorflow==2.3.0
* numpy==1.18.5
* h5py==2.8.0
  
This is also included in the requirements.txt file that can be used with pip or pipenv to create the necessary environment to run the code in. 

### FFmpeg 

This code has been tested with FFmpeg version 3.2.2, 4.1.1 and 4.4. The FFmpeg executeable must be included in the path variable for this code to run. Try running `ffmpeg -h` in a terminal, on your device, to see if you have the application in your path. 

### Big Buck Bunny clip

The included and heavily encoded video file is an 8s clip taken from the Big Buck Bunny movie, created by the Blender organization. The source material can be found on www.blender.org. 

# Upgrade

This repository was upgraded from tensorflow 1.4.1 to 2.3.0. To verify that this did not change the behavior of the model, a number of tests and comparisons was performed. No difference was found on the MOS level and only negligible differences where found on the interim predictions. See a more detailed report in the "Upgrade report"-directory. 

