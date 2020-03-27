## Nets Architectures

Two architectures were tested, a ResNet model and a custom CNN model. The ResNet model was eventually chosen due to slightly better results.

### ResNet
The main branch of the net contains 5 sections for a net works on 50x50 images, and 6 sections for 100x100 images. The first section contains the image input layer and initial convolution layer. After a 3 / 4 convolutional layer, with downsampling the spatial dimensions by a factor of 2. A final section with global average pooling, fully connected layer and my own implemented regression layer. There are residual connections around the convolutional units and the activation in the residual connections change size with respect to when there is downsampling between the layers.  

### Custom
Custom: The custom architectures that were tested contained the same initial and last sections as the ResNet, and between 6 to 10 sections of convolutional units. Each unit contains a pooling layer, convolution layer, normalization layer and activation layer, using both max and average pooling.

