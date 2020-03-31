# Extracting the body & head angles of a rat out of a video frame using CNN

**As part of a research studying rat behavior in an arena with auditory stimuli I developed an autonomous module for feature extraction. The module automatically recognizes the head and body angles of the rat, out of a video frame, using CNN and Computer Vision tools. Those rat features are critical for understanding certain behaviors of the rat during the data analysis. 
The project was made at Prof. Eli Nelken's lab, the Hebrew University.**


### Predictions examples, head and body angles, presented as arrows, over an input image.
 <p align="center"><img src = "https://github.com/tamirscherf/Rat_features_extraction/blob/master/visualization/head_body_angles.png" width = "300" height = "300">

### Predicitons over continuous frames

<p align="center"><img src = "https://github.com/tamirscherf/Rat_features_extraction/blob/master/visualization/Results_video.gif" width = "300" height = "300"></p>

### Table of content
 - [Tagging](#Tagging) : Tagging the data was necessary, therefore I built a GUI for tagging data.
 - [Data Preparation](#Data-Preparation) : Augmentation was made for the frames that were tagged.
 - [Nets Architectures](#Nets-Architectures): Two architectures were tested:
   - [ResNet](#ResNet), custom written.
   - [Custom](#Custom) - a basic CNN model.
 - [Challenges](#Challenges):
   - [Cyclic regression layer](#Cyclic-regression-layer): Due to the need in a cyclic output (an angle between 0° to 359°), implementation of a new regression layer and a corresponding loss function was required.
   - [Using two nets for minimizing execution time](#Using-two-nets-for-minimizing-execution-time): Minimizing execution time (predicting time of the net) had a great importance, as this module is part of a larger data pipeline.
 - [Results and validation](#Results-and-Validation): In order to validate the nets performances properly, several validations were made.

**I will only include here the main code files in order to present the main ideas in the project. The project was written in Matlab due to the lab requirement.**

## Tagging
**I received frames out of a video of a rat in an arena. Tagging those frames was required. I implemented a GUI using matlab tools.**
From each frame the next information was extracted: 

- Location (X,Y) of the tail base, neck base and nose.

- Body Angle, calculated as the angle between the tale base and neck base with respect to the horizontal axis.

- Head Angle, calculated similarly with neck base and nose.

- Hard image / Normal image, a binary tag, was taken for future use.

<img src="https://github.com/tamirscherf/My_Code/blob/master/visualization/Tagger.png">

As can be seen in this example of the tagger, the frame is tagged by choosing the tail base, neck base and nose points.
Each frame that is being tagged is shown together with 5 frames before and 5 frames after in order to help the person who is tagging make the most accurate tag. For the same reason, the tagger supports changing the frame brightness.


## Data Preparation

**16 videos were sampled for the DB, 100 frames were tagged from each. Augmenting with a 64 factor gave a data set of 102400(16x100x64).**

Each image was augmented with the next methods:

Rotation by 90 degrees. X4

Vertical Flip. X2

Jitter image. X2

Gaussian noise. X2

Noise to tag: uniformly distributed noise to the image tag in order to make up on tagging inaccuracies. X2

## Nets Architectures

**Two architectures were tested, a ResNet model and a custom CNN model. The ResNet model was eventually chosen due to slightly better results. Both models were trained with ADAM optimizer.**

### ResNet
The main branch of the net contains 5 sections for a net trained for 50x50 pixel input image, and 6 sections for a net trained for 100x100 pixel images(information about the differenst input image sizes on the main README).
- The first section contains the image input layer and initial convolution layer.
- Afterwards there are 3 / 4 convolutional layers, with downsampling the spatial dimensions by a factor of 2.
- A final section with global average pooling, fully connected layer and my own implemented regression layer.
There are residual connections around the convolutional units and the activation in the residual connections change size with respect to when there is downsampling between the layers.

#### ResNet Architecture

<img src="https://github.com/tamirscherf/My_Code/blob/master/visualization/MainNet_Architecture.png" width="500" height="300">
**The net width is 24.**

### Custom
The custom architectures that were tested contained the same initial and last sections as the ResNet, and between 6 to 10 sections of convolutional units. Each unit contains a pooling layer, convolution layer, normalization layer and activation layer, using both max and average pooling.

## Results and Validation

**Each network that has been trained was validated by a few parameters.**

#### Validation error, less than 5° mean error.
A validation set sampled randomly from the DB(20%).

<p align="center"><img src="https://github.com/tamirscherf/My_Code/blob/master/visualization/Body_Angle_Linear_Loss_Validation_Graph.png" width ="600" height = "312">

#### Frames from outside of the database, 6.5° mean error.
A set of 100 tagged images that is not part of the DB that the training and validation sets are taken from.
Those 100 frames are taken from a different video in order to see how well the net performs on a video that it has never learned any
frame from. 

<p align="center"><img src="https://github.com/tamirscherf/My_Code/blob/master/visualization/Final_test_validation_graph.png" width ="600" height = "315">

#### Video of un-tagged data
**A prediction of the net over 42000 frames, from a 15 seconds long video. Those frames are combined again for a video,
together with an arrow visualizing the predictions of the net. The inference latency is 0.03 milliseconds per frame.**


<p align="center"><img src = "https://github.com/tamirscherf/Rat_features_extraction/blob/master/visualization/Results_video_1.gif" width = "300" height = "300"></p>

This video also gives a good validation about the net performance with frames from video that it did learn from before.  

## Challenges

### Cyclic regression layer 

**The need in cyclic output(an angle between 0° to 359°) required adjusting a regression layer. Due to the fact there were not any built in loss function for this output, I implemented a squared loss function(forwardLoss) and its derivative(backwardLoss) for the regression layer.**

#### forwardLoss function
Returns the squared loss between the predictions Y and the output targets T.
When considiring the squared distance between a prediction y and an output target t, in the cyclic range of 0 to 359, we should notice that:
If t = 45°, y = 315° we would like (t-y)^2 to be (90)^2 and not (-270)^2.
Implementing that attribute is made in this function.
           
#### backwardLoss function
Returns the derivative of the loss with respect to the predictions Y.
The implementation of (T-Y) was needed. The absolute value of this function was implemented with forward loss function logic. The sign of (T-Y) is also case dependent and was implemented according to the following tree:

<p align="center"><img src="https://github.com/tamirscherf/My_Code/blob/master/visualization/Cyclic_loss_derivative_cases.png" width = "400" height = "228">

### Using two nets for minimizing execution time 
**Most frames are predicted by a net based on 50X50 pixel input image in order to minimize execution time. This low resolution do not always give satisficing results due to certain behavior of the rat. A net based on 100X100 pixel input is used where the 50x50 fails. Recognizing those part required creative implementation**
The input images were downsampled from 200X200 pixels to 100X100 and 50X50. While the net performances on most of the 50X50 images were good, for a few images the 50X50 resolution was too low and caused bad results.  Those images were frames of a certain behavior of the rat.  Only a net trained over 100X100 input image, along with smoothing, gave satisfying results.
#### Bad performance of one of the first 50x50 nets VS good performance 100x100 net 

<p align="center"><img src = "https://github.com/tamirscherf/Rat_features_extraction/blob/master/visualization/Bad_performance.gif" width = "300" height = "300"></p>
<p align="center"><img src = "https://github.com/tamirscherf/Rat_features_extraction/blob/master/visualization/Good_performance.gif" width = "300" height = "300"></p>

 **The behavior of the rat in those images are grooming behaviors. It is not so clear what are the exact angles, even to the human eye.** The yellow arrows in the next image indicate the wide range that can be relevant for the body angle.

<p align="center"><img src="https://github.com/tamirscherf/Rat_features_extraction/blob/master/visualization/Grooming_behavior.png" width = "300" height = "300">

#### Identifying the difficultes frames
**Using a threshold over the variance of the first derivative of the 50x50 net predictions, I can identify the frames where the 50X50 net fails. I use the 100x100 net to predict those frames and combine the results into one vector of predictions.** 

![](visualization/Predicting_using_two_nets.png)

The graph above shows the predictions of the networks over continuous frames. In blue are the predictions of the MainNet(faster net with 50X50 input), in purple the predictions of the HardNet(slower net with 100x100 input) and in yellow their combination. The HardNet and the combination values were added 80° and 40° respectively for each prediction in order to separate between the graphs. The graph also includes the variance of the first derivative over 100 frames of the MainNet and a threshold for this parameter. This parameter indicates on frames where the MainNet performs badly on, parts where the predictions are not continuous, like in frames 3.15 to 3.25. One can notice that the HardNet performs well on those parts, and therefore the use of both give better results. Important to mention that during inference time HardNet is predicting only those hard parts and not all frames.
