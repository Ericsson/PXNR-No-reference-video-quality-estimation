# Upgrade report

Zero changes were needed in this code to go from Tensorflow 1.4.1 (v1) to 2.3.0 (v2). The code was tested on an Nvidia 2060 GTX card with CUDA 11.0 installed. Both the MOS output scores were compared between v1 and v2, but also the "predictions" array that is created on line 56 in the `PXNR-model.py` script. This is, for historical reasons, formatted as an 1x96x85x1 array that can be viewed as a 96x85 pixel black and white image. The "predictions" array was stored in a deepdish file and examples of these for v1 and v2 can be found in the dd_files directory. These were calculated from the BigBuckBunny-example included in the repository. The `.dd` files can be read by loading them using deepdish in python with something like:

```python
import deepdish as dd
predictions = dd.io.load('./dd_files/BigBuckBunny_8s_v1.dd')
```

Analysis in the form of SSIM and MSE was done on these "images". The differences between the two v1 and v2 are negligble and have no impact on the final MOS score. 

## Results

| Id | Meta | Execution time(v1) | Execution time (v2) | Mos score (v1) | Mos score (v2) | dSsim = 1-ssim  | MSE |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 1 (BBB) | 30 fps and 8 sec | 774.292 | 784.880 | 2.518 | 2.518 | 4.875e-12 | 1.495e-10 |
| 2 | 24 fps and 653 sec | 2550.662 | 2940.864 | 4.091 | 4.091 | 3.136e-12 | 1.121e-10 |
| 3 | 24 fps and 464 sec | 2035.643 | 2078.847 | 3.716 | 3.716 | 8.799e-13 | 5.117e-11 |
| 4 | 24 fps and 734 sec | 2711.327 | 2812.047 | 3.386 | 3.386 | 1.158e-12 | 6.511e-11 |
| 5 | 24 fps and 164 sec | 1147.050 | 1227.045 | 3.221 | 3.221 | 3.993e-12 | 2.093e-10 |
| 6 | 24 fps and 164 sec | 1292.695 | 1231.425 | 3.221 | 3.221 | 4.341e-12 | 2.179e-10 |
| 7 | 60 fps and 337 sec | 3087.322 | 3126.029 | 3.016 | 3.016 | 1.820e-12 | 1.127e-10 |
| 8 | 24 fps and 191 sec | 1305.844 | 1338.590 | 2.582 | 2.582 | 1.582e-12 | 8.291e-11 |

