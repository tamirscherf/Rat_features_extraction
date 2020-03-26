Please read first the “Using two nets for minimizing prediction running time”
This function uses two nets in order to predict the body / head angle. 
While training nets on 50X50 images I found that the nets perform badly on certain images(An example is found in the “Bad_performence” video file).
The behavior of the rat in those images are grooming behaviors, as can be seen in the next image. 


It is not so clear what are the exact angles, even to the human eye. The yellow arrows indicate the wide range that can be relevant for the body angle.

### Inference_predict_body_direcs
The “inference_predict_body_direcs” function receives as input images and returns the body angle for each image.
First it will predict all images using the MainNet, a net that works on 50X50.
We will smooth this angle vector with a 7 sized window. Afterwards it will calculate the variance over 100 elements of the first
derivative of the angles vector. For every value of this parameter that will be larger than the threshold(40), we will predict again those
100 frames with the HardNet, a net that works on 100X100 images and therefore is slower but more accurate. We will also smooth this 100 
frame vector with an 11 sized window. We will insert these 100 vectors instead of the prediction of the MainNet. In order to keep the 
continuity, when inserting this vector we will average between the first 5 values of the HardNet and the MainNet.
