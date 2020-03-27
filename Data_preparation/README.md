16 videos were sampled for the DB, 100 frames were tagged from each. Augmenting with a 64 factor gave a data set of 102400(16x100x64).

Each image was augmented with the next methods:

Rotation by 90 degrees. X4

Vertical Flip. X2

Jitter image. X2 

Gaussian noise. X2

Noise to tag: uniformly distributed noise to the image tag in order to make up on tagging inaccuracies. X2
