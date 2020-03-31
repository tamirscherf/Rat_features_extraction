# Extracting the body & head angles of a rat out of a video frame using CNN

As part of a research studying rat behavior in an arena with auditory stimuli I developed an autonomous module for feature extraction. The module automatically recognizes the head and body angles of the rat, out of a video frame, using CNN and Computer Vision tools. Those rat features are critical for understanding certain behaviors of the rat during the data analysis. 
The project was made at Prof. Eli Nelken's lab, the Hebrew University.


### Predictions examples, head and body angles, presented as arrows, over an input image.
 <p align="center"><img src = "https://github.com/tamirscherf/Rat_features_extraction/blob/master/visualization/head_body_angles.png" width = "300" height = "300">

### Predicitons over continuous frames

<p align="center"><img src = "https://github.com/tamirscherf/Rat_features_extraction/blob/master/visualization/Results_video.gif" width = "300" height = "300"></p>

### Table of content
 - [Tagging](#Tagging) : Tagging the data was necessary, therefore I built a GUI for tagging data.
 - [Data Preparation](#Data Preparation) : Augmentation was made for the frames that were tagged.
 - Net architecture: Two architectures were tested:
   - ResNet, custom written.
   - Custom - a basic CNN model.
 - Challenges:
   - Implementing regrrsion layer: Due to the need in a cyclic output (an angle between 0° to 359°), implementation of a new regression layer and a corresponding loss function was required.
   - Minimizing execution time: Minimizing execution time (predicting time of the net) had a great importance, as this module is part of a larger data pipeline.
 - Results and validation: In order to validate the nets performnces proparly, sevral validations were made.

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
Each frame that is being tagged is shown together with 5 frames before and 5 frames after in order to help the person who is tagging make 
the most accurate tag. For the same reason, the tagger supports changing the frame brightness.

## Data Preparation

**16 videos were sampled for the DB, 100 frames were tagged from each. Augmenting with a 64 factor gave a data set of 102400(16x100x64).**

Each image was augmented with the next methods:

Rotation by 90 degrees. X4

Vertical Flip. X2

Jitter image. X2

Gaussian noise. X2

Noise to tag: uniformly distributed noise to the image tag in order to make up on tagging inaccuracies. X2



## Methods
### The folders contains the code files, methods details and further explanations.

- **Train_net:** Contains the nets architectures, validation methods, the regression layer and the training function.

- **Data_preparetion:** Contains the data augmentation function and information.

- **Tagger:** Contains the tagger code with an example and further explanaions.

- **Predict:** Contains information and code for predicting using two networks. More information below and within the folder.

## Challenges

### Using two nets for minimizing execution time 

Minimizing the running time of the prediction of the net had great importance to the project. Therefore the input images were downsampled from 200X200 pixels to 100X100 and 50X50. While the net performances on most of the 50X50 images were good, for a few images the 50X50 resolution was too low and caused bad results. Those images were frames of a certain behavior of the rat. Only a net trained over 100X100 input image gave satisfying results. In order to keep the lower execution time of the net working on 50X50 input images, for most of the frames, I implemented a function that can detect those parts where the 50X50 net fails, and used a 100X100 net only for predicting those “hard” parts. This function based on the variance of the first derivative of the 50x50 net predictions.

![](visualization/Predicting_using_two_nets.png)

The graph above shows the predictions of the networks over continuous frames. In blue are the predictions of the MainNet(faster net with 50X50 input), in purple the predictions of the HardNet(slower net with 100x100 input) and in yellow their combination. The HardNet and the combination values were added 80° and 40° respectively for each prediction in order to separate between the graphs. The graph also includes the variance of the first derivative over 100 frames of the MainNet and a threshold for this parameter. This parameter indicates on frames where the MainNet performs badly on, parts where the predictions are not continuous, like in frames 3.15 to 3.25. One can notice that the HardNet performs well on those parts, and therefore the use of both give better results. Important to mention that during inference time HardNet is predicting only those hard parts and not all frames.

**Further details in "Predict" folder.**

### Cyclic regression layer 

The need in cyclic output(an angle between 0° to 359°) required adjusting a regression layer. Due to the fact there were not any built in loss function for this output, I implemented a squared loss function and its derivative for the regression layer. Deriving the squared distance between target output (T) and prediction (Y) required the subtraction function
T - Y. Due to cyclicality this function is not trivial and the sign of it is case dependent.

![](visualization/Cyclic_loss_derivative_cases.png)

This tree shows the sepration into cases.

## Results

#### Validation of body angle network, less than 5° mean error.

![](visualization/Body_Angle_Linear_Loss_Validation_Graph.png)

#### Video of un-tagged data
A prediction of the body angle network over 42000 frames, from a 15 second long video. Those frames are combined again for a video, with an arrow visualizing the predictions of the net. The inference latency is 0.03 milliseconds per frame.

[![](http://img.youtube.com/vi/kqMZotVtYfY/0.jpg)](http://www.youtube.com/watch?v=kqMZotVtYfY)

**Further validation methods in "Train_net" folder.**

-------------------------------------------------------------------------------------------------------------


