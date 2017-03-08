# Week 10
* Retrain temporal network to use 20 consecutive frames over 1 fold
* Test new temporal network with excitation backprop
* We could also look at another dataset (e.g. GTEA)

* CMU - similar to BEOID
* GTEA - preparing sandwiches (largest)
* BEOID - Bristol egocentric object interactions dataset
* UCF101 - 101 categories action recognition from YouTube

# Week 11 - Christmas
* Split original matlab code that takes in dense optical flow and
  produces the input to the network (as it does some sort of
  preprocessing prior to using the network)
* Use the above matlab code to produce network inputs from the optical
  flow we have, then rerun excitation backprop on these
* Which frame do we plot the attention map on? Investigate plotting on
  the first, middle and last to motivate decision.
* Write up background chapter
  * Write in a textbook style
  * Convolutional neural networks
  * Two stream networks
  * Methods of visualising networks:
    * Excitation backprop
    * Deconvolutional networks
    * Deep generative networks
  * Generate my own figures to demonstrate understanding
