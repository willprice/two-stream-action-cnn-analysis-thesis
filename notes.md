# Experimentation

## Visualising and Understanding convolutional neural network ##

Someone has tried to implement this as an [IPython
notebook](https://github.com/guruucsd/CNN_visualization). It was built
with Caffe.


## Deep visualisation toolbox ##

The deep visualisation toolbox is software to produce images that
maximally excite individual neurons. It also allows you to visualise
intermediate hidden layers to understand what features they have learnt.

The deep visualisation toolbox is available on github at
[yosinski/deep-visualization-toolbox](https://github.com/yosinski/deep-visualization-toolbox)
With a [homepage](http://yosinski.com/deepvis)


## Top-down neural attention by Excitation Backprop ##

This paper introduces a technique to produce heatmaps over instances to
indicate which parts of the image contribute to the classification.

Code avaiable on github at
[Caffe-ExcitationBP](https://github.com/jimmie33/Caffe-ExcitationBP).
It is built on top of Caffe.


### Experience ###

Master branch (dee0fcb) fails to build due to
`/include/caffe/util/device-alternative.hpp` not stubbing out gpu calls
for `CPU_ONLY` mode correctly. They've modified the base Layer class
adding some new methods:

* `_eb_gpu`
* `_dc_gpu`


## Deep Dream ##

[Visualising GoogLeNet blog post](http://www.auduno.com/2015/07/29/visualizing-googlenet-classes/)


## Synthesizing the preferred inputs for neurons in neural networks via deep generator networks ##

Generates images that maximally excite neurons using deep generator
networks.

Code is available at
[Evolving-AI-Lab/synthesizing](https://github.com/Evolving-AI-Lab/synthesizing)


# Thesis Notes

## Introduction

* No mention of filter analysis

## Background

## ANNs

* Very little about training

## CNNs

* Intro about receptive fields is a bit confused
* Lacking in computational details.
* In the two stream explanation I talk about momentum, SGD but don't explain
  them, perhaps add an explanation?
* Two stream explanation is a bit thin

## Datasets

* Add more details on BEOID, size of data set, number of operators, number of
  objects, number of actions, number of combinations of object/action
  interactions
* Add UCF101 details, size of data set, breadth of actions

## Visualising

* Add filter responses of spatial network

## EBP

* What is the selective tuning model of attention?
* The first page on EBP is a bit confused, the ordering of definitions could be
  improved
* Sort out eqref and switch to crossref
