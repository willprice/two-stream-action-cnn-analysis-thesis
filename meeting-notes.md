---
title: Meeting Notes
---

# 2017-01-22

## Actions

- [x] Calculate dimensions of network streams and compare
  * See `~/nets/vgg_16_ucf101/cuhk_action_{spatial,temporal}_vgg_16_split_1_layer_sizes.txt`
  * They are exactly the same apart from the input layer => all flows are merged during the first 3D convolution
- [x] Plot excitation map alone (to see whether there are bugs in overlay)
  * They look fine
- [x] Produce artificial image all black but one pixel and look at responses throughout the layers
  * Seems OK, although
* [x] Check how Hong Kong Caffe fork reads in flow images
  * As expected, u, then v, stacking.
  * See caffe/src/caffe/io.cpp:459-467 to see how they stack the images
* [x] Step through overlay code to see whether anything odd is happening (do for both spatial and temporal)
  * Seems good
- [x] Plot on desaturated images
  * This makes colormaps much easier to see
  * Currently we raise the attention map to the power of `alpha` to make lower values in the attention map more transparent
* [x] Investigate different colour maps
  * Chose to use *sequential colormaps* to highlight changes in a perceptually linear fashion instead of jet
* [x] Plot contrastive and non contrastive videos side by side to see how much benefit it is
  * I'm going to make a nice collage of source, flow u/v and both types of excitation
* [x] Calculate spatial excitation on UCF101
* [] Look into histogram comparison with gaze data


# 2016-02-03

## Pre notes

* Fixed UCF101 2014 trained network by mean subtraction and scaling flow
  images, initially when reading they were in the range `[0,1]`, by looking at
  the yjxiong's caffe fork, specifically his python examples, I found that the
  network expects inputs in the range `[0,255]` followed by mean subtraction
  `-128`. After correcting this in my scripts I get the expected output.
* Pathological example feature map check with Dima.
* Backpropped to data layer, seems like the same 5 attention maps are non zero,
  the rest (15) are zero (see Vgg16-Action-Recognition notebook) across
  videos/classes. The excitation becomes very very sparse, maybe we're suffering
  numerical instability losing really small numbers?
* Found out why the spatial backprop doesn't work: fc8 layer wasn't trained,
  all weights/biases are 0. This was due to me renaming a layer and the fact that
  when you remove a layer (i.e. by renaming it) it ignore the protobuf struct
  corresponding to that name.


## Actions

* Consider measures for computing similarity between excitation maps (of pool3) look into L2 and Chi^2, particularly pay attention to 2D measures
* Perhaps sum up and then divide by size (average smoothness)
* For each type of analysis (EBP contrastive/non contrastive for temporal/spatial) compute the standard deviation and mean smoothness over all sequences and produce box plot diagram (do this for UCF101)
* look into Non maximal supression: gets number of peaks of a distribution
* Slice surface of excitation map above threshold to get number of peaks


# 2016-02-15

## Pre notes

* Concerned about overflows when generating temporal excitation maps, previously hadn't noticed

```
INFO:two_stream_excitation_backprop:Generating spatial excitation map from '/home/will/thesis/generated/ucf101/test-1/frames/v_PlayingSitar_g05_c06/frame000422.jpg'
INFO:two_stream_excitation_backprop:Generating spatial excitation map from '/home/will/thesis/generated/ucf101/test-1/frames/v_PlayingSitar_g05_c06/frame000423.jpg'
INFO:two_stream_excitation_backprop:Generating spatial excitation map from '/home/will/thesis/generated/ucf101/test-1/frames/v_PlayingSitar_g05_c06/frame000424.jpg'
INFO:two_stream_excitation_backprop:Generating spatial excitation map from '/home/will/thesis/generated/ucf101/test-1/frames/v_PlayingSitar_g05_c06/frame000425.jpg'
INFO:two_stream_excitation_backprop:Generating spatial excitation map from '/home/will/thesis/generated/ucf101/test-1/frames/v_PlayingSitar_g05_c06/frame000426.jpg'
/home/will/thesis/lib/excitation_backprop.py:62: RuntimeWarning: overflow encountered in exp
  np.exp(net.blobs[self.top_blob_name].data[0][neuron_id].copy())
/home/will/thesis/lib/excitation_backprop.py:64: RuntimeWarning: invalid value encountered in float_scalars
  net.blobs[self.top_blob_name].diff[0][neuron_id].sum()
/usr/lib/python3/dist-packages/matplotlib/colors.py:581: RuntimeWarning: invalid value encountered in less
  cbook._putmask(xa, xa < 0.0, -1)
/usr/lib/python3/dist-packages/scipy/misc/pilutil.py:98: RuntimeWarning: invalid value encountered in greater
  bytedata[bytedata > high] = high
/usr/lib/python3/dist-packages/scipy/misc/pilutil.py:99: RuntimeWarning: invalid value encountered in less
  bytedata[bytedata < 0] = 0
INFO:two_stream_excitation_backprop:Generating spatial excitation map from '/home/will/thesis/generated/ucf101/test-1/frames/v_PlayingSitar_g05_c06/frame000427.jpg'
INFO:two_stream_excitation_backprop:Generating spatial excitation map from '/home/will/thesis/generated/ucf101/test-1/frames/v_PlayingSitar_g05_c06/frame000428.jpg'
INFO:two_stream_excitation_backprop:Generating spatial excitation map from '/home/will/thesis/generated/ucf101/test-1/frames/v_PlayingSitar_g05_c06/frame000429.jpg'
```

* Overflow not that common (worse for things like tabla)
* Dumped out excitation maps to binary blob to speed up analysis.
* Overflow affects both contrastive and non-contrastive, so 'jerkiness' present in both videos could potentially be attributed to the overflow.
* Could try adding a softmax layer after final layer and EBPing from there to try and avoid the overflow errors

# 2016-03-02

## Pre Notes

* Demo is done with contrastive EBP
* Results at bottom are shown with non-contrastive as they look better
* Should I be consistent?
* Not sure about text, remove and add more pictures?
* No BEOID examples, add some? Where?
*

## Post Notes

* [x] top two stream labels
* [x] remove equations
* [x] Specify that the example uses ReLUs
* [x] Note that there are separate rules for pooling/other layers etc
* [x] Conv layers aren't fully connected
* [x] Add title above ebp two stream visualisation to distinguish it from single stream
* [x] Add labels to two stream ebp diagram for spatial and temporal 
* [x] Add example of ebp for object detection networks
* [x] Change EBP FC example to 1 and 0 from 0.9 and 0.1
