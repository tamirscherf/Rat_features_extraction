**Please read first the “Using two nets for minimizing execution time” back in the main README.**

## Grooming behaviors images
**While training nets on 50X50 images I found that the nets perform badly on certain images. The behavior of the rat in those images are grooming behaviors.**
This is a video example of an extreme bad performance of one of the nets I first trained.

[![](http://img.youtube.com/vi/r0Umwu2CX-Y/0.jpg)](http://www.youtube.com/watch?v=r0Umwu2CX-Y)

**It is not so clear what are the exact angles, even to the human eye.** The yellow arrows in the next image indicate the wide range that can be relevant for the body angle.

<img src="https://github.com/tamirscherf/Rat_features_extraction/blob/master/visualization/Grooming_behavior.png">

**I implemented a function that uses two networks in order to overcome this problem.**

### two_nets_predict_direcs.m
This function receives the input images and returns the body angle for each image.
First it will predict all images using the MainNet, a net that works on 50X50. We will smooth this angle vector with a 7 sized window.
Afterwards it will calculate the variance over 100(VAR_CONST) elements of the first derivative of the angles vector. For every value of
this parameter that will be larger than the threshold(40), we will predict again those 100 frames with the HardNet, a net that works on
100X100 images and therefore is slower but more accurate. We will also smooth this 100 frame vector with an 11 sized window. We will
insert these 100 vectors instead of the prediction of the MainNet. In order to keep the continuity, when inserting this vector we will
use a ramp for the window- average between the first 10 (WINDOW_RAMP_SIZE) values of the HardNet and the MainNet.
