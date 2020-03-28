I received frames out of a video of a rat in an arena. Tagging those frames was required. I implemented a GUI using matlab tools. From each frame the next information was extracted: 

Location (X,Y) of the tail base, neck base and nose.

Body Angle, calculated as the angle between the tale base and neck base with respect to the horizontal axis.

Head Angle, calculated similarly with neck base and nose.

Hard image / Normal image, a binary tag, was taken for future use.

<img src="">

As can be seen in this example of the tagger, the frame is tagged by choosing the tail base, neck base and nose points.
Each frame that is being tagged is shown together with 5 frames before and 5 frames after in order to help the person who is tagging make 
the most accurate tag. For the same reason, the tagger supports changing the frame brightness.

16 videos were sampled for the DB, 100 frames were tagged from each. Augmenting with a 64 factor gave a data set of 102400(16x100x64).

