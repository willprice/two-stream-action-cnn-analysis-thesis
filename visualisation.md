# Visualisation of CNNs

## Visualising and Understanding Convolutional Networks - Zeiler & Fergus 2014
### Aims
* Understand the function of intermediate layers in convnets.
* Use ablation and quantify the total activation of a layer to
  understand what the layer recognises in the image.

### Process

* Excite an individual feature map at any layer of a model.
* Project feature activations back to input pixel space
* Utilises deconvnet which maps features back to pixels
* Deconvnet works like a convnet in reverse
* Feature map is input to deconvnet
* Deconvnet unpools, rectifies and filters to reconstruct activity in
  previous layer

### Methods of reversing a convnet

**Unpooling**: pooling is a non invertible operation, however we can
calculate an approximation by recording the switch locations (the
location of the max input in an input region), the value we are trying
to unpool is then given to the neuron specified by the switch location,
i.e. the neuron that had the maximal value in the pooling region during
forward propagation.

The switch locations are **peculiar to a specific image**, this means
the **whole process is specific to an image**.

**Rectifying**: outputs of convolution are rectified using ReLU:
$\max(0, x)$ to ensure that we never have negative values throughout the
network, to invert this we rectify the input of the convolution in the
deconvnet

**Convolution**: The transpose of the filter in the convnet gives an
approximate inversion

### Feature to pixel space interpretation
The features mapped back down to the pixel space are specific to an
input image to the convnet (since we use the switch locations generated
by forward propping the image). The resulting image can be interpreted
as the parts of the input image that stimulate the neuron we're
inspecting.

### Occlusion
We can use occulsion of the input image to verify net is classifying
based on objects and not surrounding context. If we occlude the object
in the image we'd expect to see a decrease in feature activation.


## Top-down Neural Attention by Excitation Backprop - Zhang 2016
### Aims
* Similar to the ablation study in Zeiler's paper Zhang tries to find
  which regions of an image stimulate a neuron.
* Calculate attention maps over the input image of a specific neuron in
  the network using a probabilistic *winner takes all* method.
* Improve the attention maps by using *contrastive attention*.

### Previous work
* Previous methods have used deterministic WTA which generates binary
  attention maps, not as useful as probabilistic which generates a nice
  varied attention map--more information.

### Process
* Uses both bottom up and top down information
* Can be started from any layer
* Can be stopped at any layer
* We calculate the probability of a neuron being a winner top down from
  a neuron whose activation we're interested in.
* 

## Deep inside Convolutional Neural Networks: Visualising Image Classification  Models and Saliency Maps - Simonyan & Zisserman 2014
### Gradient descent on unit in network
* Does not highlight invariances of the neuron


## Computing Hessian of unit in network
* Shows invariance of unit
* Very difficult to understand at higher layers

## Synthesizing the preferred inputs for neurons in neural networks via deep generator networks - Nguyen 2016
### Aims
* AM (activation maximization), synthesize an input to maximally excite
  a neuron
* Introduces the use of deep generator networks (DGN) for AM (activation
  maximization)

### Activation maximisation
* Randomly generate an image
* Using backprop, calculate how each pixel colour should change to
  increase the activation of a specific neuron in the network
* Without some sort of bias to natural images this process often
  generates uninterpretable pathological images
* An image prior is often utilised to guide the synthesis to natural
  interpretable images: gaussian blur, alpha-norm, total variation,
  jitter, center-bis regularisation, initialising from mean image.
* Nguyen proposes we learn a prior instead of hand crafting one
* Clip generated layer codes such that each neuron activation is between
  0 and $3\sigma$.
