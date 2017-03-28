---
title: Feature analysis of dual stream convolutional neural networks for egocentric action recognition
author: Will Price
abstract: |
  Abstract contents
bibliography: ~/references.bib
keywords:
  - cnn
  - ann
  - convnet
  - excitation backprop
  - visualisation
  - feature analysis
  - two stream cnn
  - action recognition
colorlinks: true
toc: yes
lot: yes # list of tables
lof: yes # list of figures
link-citations: true
papersize: A4
classoption: draft
geometry: margin=1in
documentclass: scrreprt
link-citations: true
header-includes:
  - \usepackage{bm} # For \bm instead of \mathbf
  - \usepackage[mathscr]{eucal} # For \mathscr
  - \usepackage{caption} # For captions
  - \usepackage{subfig} # For side by side figures
  - \usepackage{xcolor} # For colouring text like the macro `note`
  - \usepackage{algorithm2e} # For typesetting algorithms
  # Watermark
  - \hypersetup{bookmarks}
  - \usepackage{draftwatermark} # For watermark
  - \setkeys{Gin}{draft=false} # Enable pictures in draft mode
  - |
    \SetWatermarkAngle{90}
    \SetWatermarkHorCenter{.06\paperwidth}
  # End watermark
  - \graphicspath{{../}{./}}
  - \newcommand{\etal}{\textit{et al}.}
  - \DeclareMathOperator*{\argmin}{arg\,min}
  - \DeclareMathOperator*{\argmax}{arg\,max}
  - \newcommand{\learningrate}{\eta}
  - \newcommand{\neuron}[2]{a_{#2}^{(#1)}}
    # 1: layer index
    # 2: neuron index
  - \newcommand{\neuronforward}[2]{\hat{a}_{#2}^{(#1)}}
    # 1: layer index
    # 2: neuron index
  - \newcommand{\ebpscalar}[2]{Z_{#2}^{(#1)}}
    # 1: layer index
    # 2: neuron index
  - \newcommand{\weight}[3]{w_{#2,#3}^{(#1)}}
    # 1: layer index
    # 2: from neuron index
    # 3: to neuron index
  - \newcommand{\children}[2]{\mathscr{C}^{(#1)}_{#2}}
    # 1: layer index
    # 2: neuron index
  - \newcommand{\parents}[2]{\mathscr{P}^{(#1)}_{#2}}
    # 1: layer index
    # 2: neuron index
  - \newcommand{\cwp}[4]{P\left(\neuron{#1}{#2} | \neuron{#3}{#4}\right)}
    # 1: child layer index
    # 2: child neuron index
    # 3: parent layer index
    # 3: parent neuron index
  - \newcommand{\mwp}[2]{P\left(\neuron{#1}{#2}\right)}
    # 1: layer index
    # 2: neuron index
  - \newcommand{\neuroninput}[2]{\tilde{a}^{(#1)}_{#2}}
    # 1: layer index
    # 2: neuron index
  - \newcommand{\neuronoutput}[2]{\hat{a}^{(#1)}_{#2}}
    # 1: layer index
    # 2: neuron index
---

<!--- Draft options -->
\SetWatermarkScale{0.3}
\SetWatermarkText{\textbf{Draft: \today}}

<##define note|\textcolor{red}{NOTE: #1}\newline>
<##define check|\textcolor{blue}{CHECK: #1}\newline>
<##define todo|\textcolor{green}{TODO: #1}\newline>
# Introduction {#sec:introduction}

Action Recognition in Computer Vision refers approaches that aim to infer the
action or actions of an actor or actors using visual observations, either images
or videos. We further constrain the definition to infer actions
from video sequences (sequences of images captured by video cameras at regular
intervals). Action recognition from video has many
critical applications[@ranasinghe2016_reviewapplicationsactivity] such as detecting
suspicious behaviours of travellers in airports from CCTV footage, recognising
the fall of an elderly person who lives alone, and ensuring the safety of the
operator of a machine by automatically stopping the machine in case of an
accident.

Convolutional Neural Networks (CNNs), a form of supervised deep learning model,
have recently been used to obtain state of the art results in object detection
in images[@krizhevsky2012_Imagenetclassificationdeep]. Naturally researchers
have questioned whether these performance increases translate to action
recognition. The results of extending CNN architectures to cope with video
sequences have yielded similar increases in performance
[@feichtenhofer2016_ConvolutionalTwoStreamNetwork;@wang2016_TemporalSegmentNetworks]

One downside of CNNs compared to other models such as decision trees is their
lack of transparency in predictions: why does the model classify this particular
video sequence of a person performing pull ups as skipping? With a decision tree
we would be able to provide an explanation based on the features from the root
of the tree to the branch, but there's no obvious analogous technique to
understand a CNN's prediction.

There are efforts made to develop techniques to help understand the
features learnt by CNNs to aid debugging, but most of these were developed
and applied to object detection networks, there is little research to see
whether the techniques generalise to networks trained for other tasks such as
action recognition.

This thesis investigates the applicability of visualisation techniques for two
stream CNNs[@simonyan2014_TwoStreamConvolutionalNetworks] trained for action
recognition. There are other architectures for action recognition, but they are
out of the scope of this investigation. A method for determining the importance
of regions in an input frame to a network called Excitation Backpropagation
(EBP)[@zhang2016_TopdownNeuralAttention] is utilised and extended to produce
excitation maps (heat maps indicating the regions of importance to exciting a
neuron in the network) across sequences of frames from a video sequence.

# Background {#sec:background}

We introduce the basic concepts of artificial neural networks and convolutional
neural networks, then we go on to look at techniques developed to understand the
features learnt by CNNs with a particular focus on excitation back propagation
which is extended in [@sec:ebp-for-temporal-networks] for use on temporal
networks.

## Artificial neural networks (ANNs) {#sec:background:ann}

Biology is a rich source of inspiration for techniques in computer
science. Artificial neural networks (ANNs) form a strand of biologically inspired
computational models based on the learning and communication processes in the
brain. To understand neural networks, we will take each concept from the bottom
up step by step until we arrive at the modern model of an artificial neural network.
First we shall examine *artificial neurons*, of which there are several models,
the earliest being the McCulloch-Pitts
neuron[@mcculloch1943_logicalcalculusideas], followed by the
perceptron[@rosenblatt1957_Perceptronperceivingrecognising]. We will then see
how one can form a network made up of artificial neurons using perceptrons, then
briefly discuss the challenges scaling these networks up to process images or
videos leading into the introduction to convolutional neural networks, a
constrained ANN architecture encoding certain assumptions about the input to
make training these models on modern computers a viable proposition.

### The McCulloch-Pitt's neuron

The McCulloch-Pitt's neuron is mostly a historical curiosity, and if the
evolution of artificial neural networks doesn't interest you skip ahead to the
perceptron.

Warren McCulloch and Walter Pitts were arguably the first to provide a
mathematical model of neuron inspired by biology, they developed a logical
calculus describing neuron behaviour[@mcculloch1943_logicalcalculusideas]. Their
model neuron, known as McCulloch-Pitt's neuron (MCP), was shown to be
computationally universal. Every network of MCP neurons encoded an equivalent
logical proposition.

MCP neurons have a set of inputs, the sum of which is compared to a threshold
which determines whether the neuron fires or not. Both excitatory and inhibitory
signals were modelled, if an incoming inhibitory connection is firing, then the
output is completely inhibited regardless of the firing of the other incoming signals.

Networks of MCP neurons were investigated to implement more complicated
propositions.

### The Perceptron

The next major contribution to the realisation of artificial neural networks
following McCulloch and Pitt's work was the
Perceptron[@rosenblatt1957_Perceptronperceivingrecognising], initially conceived
as a physical device for learning to recognise patterns in images or sounds
(each of these using the same principles, but with different inputs) by Frank
Rosenblatt, it was later formulated as an algorithm.

The modern perceptron is a supervised learning algorithm that produces a binary
classifier with a linear decision boundary. First we'll step through each term
in this definition before presenting the perceptron:

* *learning* is concerned with the problem of constructing a function $f : X
  \rightarrow Y$, where $X$ denotes the feature space, $Y$ denotes the label space.
* *feature space*, the vector space of data we want to predict a class from
  (domain of $f$). The data available to be used as input to the model
  influences the choice of domain, e.g. an image of size $W \times H$ could be
  analysed for edges and those used as the feature vector, or every pixel could
  be used in $W \times H$ long vector as input to the predictive function,
  amongst many other possibilities.
* *label space*, the vector space in which the desired result resides (co-domain
  of $f$).
* *supervised learning* uses a labelled training set $X_{\text{train}} =
  \{ (\bm{x}_0, y_0), \ldots, (\bm{x}_n, y_n) \}$ to learn $f$ where each instance
  in $X_{\text{train}}$ is used.
* *classification* further refines the function $f$ to be learnt, classification
  is about learning a function that predicts one of a finite number of labels
  hence the label space will be a finite set of labels/classes.
* *binary classification* specifies that $f$ is to predict 2 labels, usually
  referred to as the *positive* and *negative* classes.
* a *decision boundary* is a term used in classification problems in reference
  to the surfaces separating the different areas of uniform class in the feature
  space.

The perceptron learns a linear classifier which takes the form $\bm{w} \cdot
\bm{x} > 0$, where $\bm{w}$ is the learnt weight vector, and $\bm{x}$ a the
feature vector. If the dot product of the weight vector with the feature vector
is greater than zero the test instance is labelled with the positive class,
otherwise the negative class. A graphical representation of this is given in
[@fig:perceptron], each element $x_i$ of the feature vector forms an input node
on a graph, elements of the weight vector $w_i$ form edges from the
corresponding input ($x_i$) to the perceptron body. As inputs flow along the
edges they are multiplied by the weight on the edge and then summed in the
perceptron.

![Graphical representation of the perceptron](media/images/perceptron.pdf){#fig:perceptron}

The perceptron learning algorithm constructs a weight vector $\bm{w}$ from a set
of labelled training examples $\mathscr{X} = \{ (\bm{x}_0, y_0), \ldots,
(\bm{x}_n, y_n) \}$ where $\bm{x}_i$ is the feature representation of the $i$-th
training example, and $y_i$ is the true label of the example.

The following algorithm (Algorithm \ref{alg:perceptron-training}) is used to
learn a weight vector $\bm{w}$ that correctly classifies all examples on the
training set (if possible, otherwise the algorithm fails to terminate).

\begin{algorithm}[H]
\label{alg:perceptron-training}
\caption{Perceptron training}
\SetKwData{Converged}{converged}
\KwData{Training set: $\mathscr{X} = \{(\bm{x_0}, y_0), \cdots, (\bm{x_n}, y_n)\}$}
\KwResult{Binary classifier: $\bm{w} \cdot \bm{x} > 0$}
$\bm{w} \leftarrow \bm{0}$\;
\Converged $\leftarrow$ false\;
\While{\Converged = false}{
  \Converged $\leftarrow$ true\;
  \For{$i = 1$ to $|\mathscr{X}|$}{
    \If{$y_i \bm{w} \cdot \bm{x_i} \leq 0$}{
    $\bm{w} \leftarrow \bm{w} + \learningrate y_i \bm{x_i}$\;
    \Converged $\leftarrow$ false\;
    }
  }
}
\end{algorithm}

The idea is to iteratively build up a weight vector $\bm{w}$ that correctly
classifies all training data. Initially starting with the zero vector will
result in the misclassification of all training examples as they will all lie on
the decision boundary, $\bm{w} \cdot x = 0$. The core of the algorithm depends
on interpreting the dot product as a measure of similarity. By adding weighted
training feature vectors and factoring in the correct sign of $y_i$, $\eta
\bm{x}_i y_i$ to the weight vector $\bm{w}$, we increase the similarity of the
updated weight vector with the training example resulting in a more positive dot
product between $\bm{w}$ and $\bm{x}_i y_i$ which is more likely to pass the
decision threshold.

The perceptron learns a linear classifier, which is only of use if the data is
linearly separable, if it isn't then we have to consider alternatives. Minsky
and Papert used the following example ([@fig:xor]) in
Perceptrons[@minsky1969_Perceptrons] to demonstrate the limitations of a single
perceptron, the figure shows the function XOR where the red
points cannot be separated from the green points with a linear boundary, so a
boundary like that shown in [@fig:xor-non-linear-decision-boundary] is needed to
separate the data.

![XOR](media/images/xor.pdf){#fig:xor}

![XOR with non linear decision boundary](media/images/xor-boundaries.pdf){#fig:xor-non-linear-decision-boundary}

Perceptrons can be used to learn a non linear decision boundary in two ways. The
first technique is to replace the dot product with a *kernel*, a function with
properties similar to that of the dot product. Use of a kernel as a replacement
for the dot product can be thought of as a transformation of the instances into
a new space in which a linear decision boundary is learnt. A kernel is chosen
such that it is likely that the data will be linearly separable in this new space.
Alternatively, the other technique is to stack perceptrons so that the output of
one is used as (one of) the input(s) to another, the network formed is called a
*multilayer perceptron (MLP)*.

When using MLPs we have to adapt the perceptron's output to be followed by a
non-linear transformation $\phi$; the reason for this is that if we otherwise
stack perceptrons without modification the network would compute nested linear
transformations which can be represented as a single linear transformation, i.e.
MLPs without non linearity applied to the output of each unit are no more
expressive than a single perceptron; the complexity of the decision boundaries
learnt by MLPs is due to the successive application of linear transformations
and non linearities.

A small multi layer perceptron network is given in [@fig:ann-example]. Each
circle represents the body of a perceptron in which the weighted sum of its
inputs are calculated and then passed through an activation function. Each edge
between perceptrons indicates the connectivity and weight between them.
For example, $\neuron{1}{0}$ has two incoming connections from $\neuron{0}{0}$
with a weight $+1$ and $\neuron{0}{1}$ with a weight $0$, it will output the
value ${\phi\left(1 \cdot \neuron{0}{0} + 0 \cdot \neuron{0}{1})\right)}$

![Example Multilayer Perceptron with a single hidden layer](media/images/ann-example.pdf){#fig:ann-example}

When producing any predictive model it is important to be able to evaluate it to
see whether it performs sufficiently well for its use case. There are many
measures to evaluate models including (but not limited to): accuracy, precision,
recall, and the $F_1$ score; picking a measure depends on the class ratio of the
data you expect to run your model on and the cost ratio that defines how costly
it is to mistake one class for another. Let's assume we've chosen accuracy to
evaluate a perceptron we've just trained, to evaluate it we could see how it
performs on the training data however since we know that the perceptron
perfectly splits the training data into two classes otherwise the algorithm
doesn't terminate the training accuracy will always be 100%, which makes this a
useless test, instead we need a new dataset (perhaps some data kept back from
the training set) on which the model hasn't been trained, referred to as the
*cross validation dataset*, on which we evaluate the performance.

<##check Do you want me to add more about cross-fold validation etc?>

<##todo Talk about how two-stream NNs are tested using forward  propagation and
how the decisions are fused before training and tuning>

A forward pass of the network in [@fig:ann-example] is computed using the
activation function ${\phi(x) = \max(x, 0)}$ in [@fig:ann-forward]. We traverse
the graph from left to right, computing the values of every perceptron in each
layer before moving to the next layer. The edges are relabelled with the product
of the weight and input to the edge, the diamonds below each perceptron show the
sum of the weighted inputs (the sum of the values labelled on the edges) and the
diamonds above show the output value of the perceptron after passing the
weighted sum of inputs through the activation function $\phi$.

![Forward propagation of the example ANN](media/images/ann-forward.pdf){#fig:ann-forward}

To solve the XOR problem we can construct individual perceptrons that simulate
Boolean functions and then use the XOR propositional decomposition ($p \oplus q
= (p \lor q) \land \lnot (p \land q)$) to construct a network that implements
XOR, but this solution negates the main benefit of using a learning algorithm in
the first place: we want the machine to form a solution.

Combining multiple perceptrons into a network forming a multi layer perceptron
brings us closer to the modern artificial neural network, however we now have a
new problem: learning the weights of all the perceptrons. Since the weight
vectors of the perceptrons in the network are not independent, changing one will
effect inputs deeper in the network causing a change in the final output meaning
we cannot use Algorithm \ref{alg:perceptron-training}. An exhaustive search over
the weights of the perceptrons would be able to find an optimal weight
configuration, but would be computationally intractable due to the combinatorial
nature of the search.

Training multilayer perceptrons was the main impediment to their use until the
process of *error back propagation* (first developed by
Kelley[@kelley1960_GradientTheoryOptimal] and
Bryson[@dreyfus1990_Artificialneuralnetworks][@schmidhuber2015_DeepLearningNeural])
was applied to the problem by Paul Werbos[@schmidhuber2015_DeepLearningNeural].
Back propagation gives us a tool to understand how modifying each component of
the weight vector of each perceptron changes the output of the network. A loss
function is defined that specifies how the output of the network differs from
the expected output, the partial derivatives of the loss function with respect
to each weight component $\frac{\partial a^{(n)}_i}{\partial w^{(l)}_j}$ are
calculated. Having obtained the partial derivatives we can perform gradient
descent to tweak the weights in such a way that the output of the network
becomes closer to the desired output for each training example, thereby
minimising the loss function.

The MLP is the foundation of modern neural networks, although
in modern parlance it is known as a *fully connected feedforward network*. The
network is *fully connected* since each neuron in layer $l + 1$ is connected to
every neuron in layer $l$. A *feedforward* network is one which neurons are
connected in such a way that no cycles are present (networks with cycles are
known as *recurrent networks*).


## Convolutional neural networks (CNNs) {#sec:background:cnns}

CNNs, a specialised form of ANNs, were first proposed in
[@fukushima1980_Neocognitronselforganizingneural] as a network architecture
called the *neocognitron* inspired by the research of Hubel and Wiesel on the
visual cortex[@hubel1959_Receptivefieldssingle]. Hubel and Wiesel found that the
neurons in the visual cortex of a cat's brain responded to patterns in regions
in the cats field of view, they termed the region causing an excitation of a
neuron as the *receptive field* of that neuron. Furthermore, they discovered that
neurons were arranged in such a way that neurons that had similar receptive
fields were also physically co-located in the cortex. Fukushima
\etal{} designed the connectivity of the neurons in the neocognitron to model
the connectivity of the neurons in the visual cortex such that each neuron was
connected to neurons in the previous layer to form a receptive field. This
architecture is very similar to those currently in use.

Building on the work of the neocognitron, modern CNN models introduce one
substantial improvement: rather than learning parameters for each individual
neuron in a layer, we instead assume that there exist neurons that fire in
response to a class of patterns in the input for all receptive fields, this
allows us to learn only a single set of weights to detect such patterns and
effectively construct the neurons for each receptive field at runtime. To
implement this, convolutional filters are learnt instead of individual neuronal
weights which are then convolved with the input tensor to produce the output.

The restricted architectures of CNNs facilitates a new view of these networks
compared to ANNs; the overarching theme is to raise the level of abstraction
from neurons to layers, and individual inputs to input volumes. ANNs
have no fixed/predetermined function, different groups of neurons in a layer can serve
different purposes, however this is not the case in CNNs, layers are homogeneous
in their function, e.g. a convolutional layer only computes convolution of its
input. CNN architectures are described by their constituent layers and the
hyperparameters that configure those layers, different layer types have
different hyperparameters.

Layers are constructed using this conceptual model and can be mapped down to the
traditional ANN model of neurons and weights.

Inputs and outputs from layers are thought of as tensors rather than sets of
disparate features encoded in a vector, this view is enabled the homogeneity of
the input. Typically for image and video processing networks, the input is a 3D
block, where width, $W$, and height, $H$, correspond to the width and height of
the input image, and the depth, $D$, of the block corresponds to the number of
channels in the image (e.g. 3 for RGB images, 1 for grayscale).

### Layers

Layers can be thought of volume transformation functions, a volume of dimensions
$W \times H \times D$ used as input by a layer is transformed to an output
volume of dimensions $W' \times H' \times D'$ where the new dimensions are a
function of the input volume's dimensions and layer's parameters.

There are many types of layers, but they mainly fit into four broad categories:
*fully connected*, *pooling*, *convolutional*, *activation*.

#### Fully connected

Fully connected layers are like those in a MLP where each neuron
is connected to every neuron in the previous layer, see [@fig:layers:fully-connected]. These layers are very large in
parameters so are usually used further in the network when the input volume size
is considerably reduced. In CNNs, fully connected layers draw together
high level features from regions that are spatially distant from each other,
consider the task of detecting a bike in an image, if you have filters that fire
based on wheels, there will be neurons that activate when wheels are present in
different locations in the image, the fully connected layer will be able to draw
together the wheel-neuron activations that are spatially separate and help
discriminate images of bikes from images with wheels that don't share the same
spatial relationship that wheels on bikes do.

<##check Dima, are the bike wheels a sensible example?>

![Fully Connected layer](media/images/layer-fully-connected.pdf){#fig:layers:fully-connected}


#### Pooling

Pooling layers exist to increase the receptive field of deeper layers enabling
them to learn features than span larger spatial extents, this is accomplished by
reducing the size of the input volume by computing some function over a region
of the input yielding a single value, this operation is computed by a *pooling
filter*, see [@fig:layers:pooling]. Max pooling is a common pooling filter where the maximum value in an
input region is selected to be propagated forward discarding the rest of the
values in the region.

Pooling layers typically have *size*, *pad* and *stride* parameters. The *size*
determines the region over which pooling takes place, *padding* specifies the
whether to zero pad the input along it's the borders of each axis and if so, how
wide/deep the padding is, *stride* specifies how many elements to slide the
filter along the input between each application of the pooling filter. For
example, a 2D ^[3D pooling is possible and used in some action recognition
architectures]
max pooling layer with size $2 \times 2$, padding $1 \times 1$
and stride $2 \times 2$ will first pad it's input with a border of zeroes 1
element deep, then apply a $2 \times 2$ max pooling filter propagating the max
value over that region to its output, it will shift by 2 elements right along
the row and repeatedly be applied until reaching the end of the row then will be
shifted 2 down, this process repeats until the filter reaches the bottom right
most position. For an input tensor of size $224 \times 224 \times 10$ the padded
input will be of size $226 \times 226 \times 10$, since the pooling region is $2
\times 2$ there are $(226 - 2) \times (226 - 2)$ locations it could be applied
at, but our stride isn't $1 \times 1$, but $2 \times 2$ hence the output tensor
will have dimensions ${(226 - 2) / 2 \times (226 - 2) / 2 = 112 \times 112}$


<##check Dima, could you double check I haven't messed this up!>

![Pooling layer](media/images/layer-pooling.pdf){#fig:layers:pooling}


#### Convolutional

Convolutional layers consist of one or more filters that are slid along an input
volume and convolved at each location producing a single value which are
aggregated in an output volume, see [@fig:layers:convolution]. The filter parameters are learnt and are
constant across different locations in the input volume, this massively reduces
the number of parameters of the model compared to a fully connected layer
handling similar volumes sizes making them much more space efficient than fully
connected networks.

The number of parameters in a convolutional layer is *only* dependent upon the
filters. The filter is parameterised by its size, stride and zero padding. The
*size* determines the volume of the filter; *stride*, how the filter is moved
through the input volume; *zero padding*, whether or not the volume is padded with
zeros when convolved with the filter.

For a layer with 4 filters with the parameters:

* Size: $W_f \times H_f \times D_f$ = $4 \times 4 \times 3$
* Padding: $W_p \times H_p \times D_p$ = $0 \times 0 \times 0$
* Stride: $S_w \times S_h \times S_d$ = $1 \times 1 \times 0$

The layer has a total of $4 \cdot 4 \cdot 3 \cdot 1 \cdot 1 = 38$ parameters.


![Convolutional layer](media/images/layer-convolution.pdf){#fig:layers:convolution}

<##todo Go through the computation of a single or few cells in the above example>
#### Activation

Activation layers are much the same as in traditional ANNs, an activation
function is chosen and applied element wise to the input tensor to produce an
output tensor of the same dimensions, see [@fig:layers:activation]. Activation functions take the form
$\phi(x)$, common functions used include the
rectified linear unit (ReLU), $\phi(x) = \max(x, 0)$, sigmoid
$\phi(x) = \frac{1}{1 + e^{-x}}$ and hyperbolic tangent ${\phi(x) =
\frac{e^{2x} - 1}{e^{2x} + 1}}$

![Activation layer](media/images/layer-activation.pdf){#fig:layers:activation}

### Architectures

The architecture of a CNN refers to the choice of layers, and parameters and
connectivity of those layers. Architectures are designed to solve a particular
problem where the number of layers limits the complexity of the features learnt
by the network. Networks are designed such that they have sufficient
representational power (i.e. number of layers, and number of filters in those
layers) necessary to learn the desired mapping from input to output, but are as
small as possible as each new layer adds additional hyperparameters (number of
filters, filter size, stride size) to the network which further increases the
already considerable time spent searching for optimal hyperparameters by
training a network per hyperparameter configuration.

First we look at architectures for object detection as this task has been the
focus of most research inspiring architectures for action recognition which we
discuss afterwards.

#### Object detection {#sec:background:object-detection}

Historically CNNs were extensively applied to the object detection problem
popularised by the ImageNet challenge [@russakovsky2014_ImageNetLargeScale]. The
challenge consists of two main problems *object detection* and *object
localisation*, participants produce a model capable of predicting the likelihood
of object classes presence in a test image. Models are evaluated based on their
top-1 and top-5 error rate where the top-$N$ error rate is defined as the
proportion of test images whose prediction is considered an error if the ground
truth label does not appear in the top-$N$ predictions.

##### AlexNet

AlexNet[@krizhevsky2012_Imagenetclassificationdeep] was the first CNN submission
to ImageNet 2012 challenge achieving a large error-rate reduction over previous
state of the art methods scoring a top-1 error rate of 36.7% and top-5 error
rate of 15.3%.

![AlexNet Architecture](media/images/alexnet.pdf){#fig:architecture:alexnet height=2cm}

##### VGG16

The VGG16 architecture [@simonyan2014_VeryDeepConvolutional] was developed as an
enhancement over the original
AlexNet[@krizhevsky2012_Imagenetclassificationdeep] architecture investigating
the effects of using a 'very deep' architecture with many stacked convolutional
layers. 6 similar network architectures with increasing depth were trained and
their classification performance tested against the ImageNet dataset, the
network configurations with more convolutional layers performed better than
those with fewer resulting in two configuration VGG16 and VGG19 with 16 and 19
convolutional layers respectively. The VGG architectures (used in an ensemble)
won first place in the object classification stream of the ImageNet 2014
challenge scoring a top-1 error rate of 24.4% and top-5 error rate of 7.0%, a
considerable improvement over AlexNet. Another deep network architecture by
Google[@szegedy2014_GoingDeeperConvolutions] achieved similar error rates
(second place) with 22 layers suggesting that more layers isn't necessarily
better, but that the architectures in 2012 were too shallow.

![VGG16[@simonyan2014_VeryDeepConvolutional] Architecture](media/images/vgg16-rotated90.pdf){#fig:architecture:vgg16 height=100% width=2cm}

#### Action recognition

The challenge of recognising actions from video sequences has recently seen the
application of CNNs inspired from their performance on object detection. A
variety of architectures for tackling the problem have emerged which we shall
explore in chronological order to see how architectures have evolved over time
concluding with the architecture used in our experiments.

![Which way is the tap turning? (BEOID)](media/images/beoid-turn-tap.jpg){#fig:tap-turn height=2in}

Action recognition can be performed from single images or video sequences, both
approaches have been investigated and we will outline prior work in this
section. Some actions (or combinations of actions) cannot be reliable predicted
by single images alone, this is because the motion in the action is the only
distinguishing factor from other actions. For example, take the example in
[@fig:tap-turn] of someone turning a water tap in a kitchen sink on and off. We
cannot reliably determine whether the tap is being turned on or off from a
single image alone, instead it is necessary to examine multiple video frames
over time to determine the direction of turning (and flow of water) and hence
distinguish the two actions: turning the tap on and turning the tap off.

![Contextual clues from the surrounding environment can aid action recognition
from single frames](media/images/action-recognition-contextual-clues.pdf){#fig:contextual-clues height=2.5in}

In contrast to the observation above, single images alone can perform well based
on *cues* in the image. Consider the two images in [@fig:contextual-clues], the
actions being performed can be determined from appearance alone, we can infer
that the man is writing on the board because he is facing the board, the board
has writing on it, and he is holding a pen close to where there is writing.
Similarly, the head massage image
Gkioxari \etal have investigated the use of these contextual clues in action
recognition[@gkioxari2015_ContextualActionRecognition]

The first investigation of CNNs for action recognition operating on raw frame
data (i.e. without explicit feature extraction) was conducted in
[@baccouche2011_SequentialDeepLearning]. They introduced an architecture with an
input of a stack of video frames which were then processed by multiple
convolutional layers to learn spatio-temporal features on the KTH human actions
dataset[@schuldt2004_Recognizinghumanactions] the output of which was then used
by a recurrent ANN called a long short-term memory (LSTM) to obtain a prediction
over the whole video sequence, although they also compared classification with
the LSTM layer to a linear classifier and only found modest performance benefits
(on the order of a few percentage points) indicating that short clips from a
video may be sufficient for action recognition.

A similar architecture with a larger input volume was investigated in
[@ji2013_3DConvolutionalNeural], instead of training the whole network, the
first layers were hand initialised to obtain the following transformations:
grayscale, spatial gradients in both x and y directions, and optical flow in
both x and y directions. A video sequence processed by the first layer results
in a stack of grayscale video frames, spatial gradient frames and optical flow
frames. The network was evaluated on both the KTH dataset with competitive
performance to other methods developed at the time and TRECVID[@_TRECVIDData]
dataset improving over the state of the art.

##### Two stream

A biologically inspired architecture based on the two-stream visual processing
hypothesis is introduced in [@simonyan2014_TwoStreamConvolutionalNetworks]. The
two stream hypothesis states that two processing streams are used in the brain
for processing visual input: the dorsal stream for motion, good at detecting and
recognising movements; and the ventral stream recognising form, good at
detecting objects. An architecture with two streams based on the biological
hypothesis is given, it has two streams, the spatial for handling the
appearance(analog of the ventral stream) and the temporal for handling the
motion(analog of the dorsal stream). A video sequence is processed to obtain the
optical flow frames using the TVL1[@zach2007_DualityBasedApproach] algorithm
which are used as input to the temporal stream, and a single frame is used as
input to the spatial stream. The two streams process the inputs in parallel each
of which produces an action prediction, the results are then combined using a
linear classifier, see [@fig:architecture:two-stream].

The spatial stream takes in a single input frame $\tau$, size $W \times H \times
3$ (using RGB images). The corresponding input to the temporal stream is $W
\times H \times 2L$ where $L$ is the temporal duration, a hyperparameter of the
network determining over how a long period to compute optical flow from,
In [@simonyan2014_TwoStreamConvolutionalNetworks] use $L = 10$ and both our
networks are also trained with $L = 10$. The temporal input is computed from the
frames $\tau - L/2$ to $\tau + L/2$ yielding $L$ frames from which $2L$ optical
flow frames are obtained, twice as many as the input due to computing both
flow in $u$ and $v$ directions. The flow frames are then combined by
alternating $u$ and $v$ frames, all even frames in the optical flow stack are in
$u$ direction and odd, the $v$ direction.

Raw optical flow frames stored as floats can take up a large amount of space so
instead they are converted to greyscale images in the range $[0, 255]$ and
compressed using JPEG to reduce storage requirements. On input to the network
the frames are mean-centred around $(127, 127, 127)$.

The networks are trained concurrently using mini-batch stochastic gradient
descent. 256 video sequences are selected from the training dataset uniformly
across the classes from which a single frame $\tau$ is sampled from each of
these videos and the corresponding optical flow stack from $\tau - L/2$ to
$\tau + L/2$ is computed forming the input of the spatial and temporal network
respectively. Stochastic gradient descent with momentum is used to train the
networks against the ground truth actions of the sample video. A common strategy
in training CNNs is to initialise the weights of the network to those of the
same network architecture trained on ImageNet, this helps avoid overfitting on
small datasets. The weights of the UCF101 trained network was initialised with
ImageNet trained weights, and the BEOID trained network was initialised on the
weights of the UCF101 trained network.

The

<##check Dima, this is not entirely true, the UCF101 trained network was from a
separate source, not the one that Mike/Davide trained, but I'm hesitant to state
this as it muddies the explanation>

Classification of a video using the two stream network is accomplished by
sampling a fixed number of frames with equal temporal distance between each
pair of consecutively sampled frames. For each sampled frame $F$, a new set of
frames are computed by flipping and cropping $F$. The corresponding input to the
temporal network is computed from the frames post-transformation. The overall
class scores for the whole video is computed as the average of the class scores
for each individual sample. For example sampling 20 frames from a 60 second long
clip (at 24 FPS) will yield frames with indices $k \cdot 24 \cdot \frac{60}{20}$
for $k \in [0 .. 19]$, each frame $F_k$ at index $k$ in the video will then be
cropped and flipped to produce a set of derived frames $\mathscr{F_k}$, each
frame in the set will have its class score computed by a forward pass through
the spatial network, a corresponding input for the temporal network is also
computed, the scores are then combined (*fused*) by a linear combination.

| Dataset | Stream                  | Accuracy |
|---------|-------------------------|----------|
| BEOID   | Spatial                 |    83.9% |
|         | Temporal                |    92.9% |
|         | Post convolution fusion |    94.8% |
| UCF101  | Spatial                 |    78.4% |
|         | Temporal                |    87.0% |
|         | Late fusion             |    91.4% |

<##todo Add SGD, mini batch, momentum explanation>

![Two stream CNN
architecture[@simonyan2014_TwoStreamConvolutionalNetworks]](media/images/two-stream-cnn.pdf){#fig:architecture:two-stream}

## Video Datasets {#sec:background:datasets}

In [@sec:background:understanding] the surveyed papers frequently make use
of datasets, rather than explaining them as they are referenced they are instead
described and consolidated in this section.

### KTH - Human actions

The KTH human action[@schuldt2004_Recognizinghumanactions] dataset is composed
of 6 action classes: walking, jogging, running, boxing, hand waving and hand
clapping performed by 25 subjects in 4 scenarios: outdoors, outdoors with scale
variation, outdoors with different clothes, and indoors. Each action class has
100 example clips

![KTH Human action[@schuldt2004_Recognizinghumanactions] samples](media/images/kth-sample.png)

### TRECVID - London Gatwick Airport Surveillance video

TRECVID[@_TRECVIDData] is a competition held each year by The National Institute
of Standards and Technology. In 2008 one of the challenges held asked
participants to detect 10 different events in 100 hours of CCTV camera footage
shot inside London Gatwick Airport[@rose2009_TRECVid2008Event]. The events to
detect were: person puts mobile phone to ear; elevator doors opening with a
person waiting in front of them, but the person doesn't get in before the doors
close; someone drops or puts down an object; someone moves through a controlled
access door opposite to the normal flow of traffic; one or more people walk up
to one or more other people, stop, and some communication occurs; when one or
more people separate themselves from a group of two or more people, who are
either standing, sitting , or moving together communicating, and then leaves the
frame; person running; person pointing; person taking a picture (descriptions
taken from [@rose2009_TRECVid2008Event]).

### Sports-1M - YouTube sport actions

Sports-1M[@karpathy2014_LargeScaleVideoClassification] is a weakly annotated
action dataset obtained from YouTube consisting of 1 million videos over 487
sport classes. The videos are obtained by searching for the sport class and then
collecting videos from the search results hence the labels in the dataset are
noisy^[There exists incorrect labelled examples in the dataset].

### HMDB51 - Human motion database

HMDB51[@kuehne2011_HMDBlargevideo] is a human activity dataset containing 6849
video clips over 51 action classes each containing a minimum of 101 clips each
fitting one of 5 broad categories: general facial actions, facial actions with
object manipulation, general body movements, body movements with object
interaction, body movements for human interaction. Examples are given
in [@fig:dataset:hmdb51:samples]^[HMDB51 Samples image obtained from
http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/]

![HMDB51[@kuehne2011_HMDBlargevideo] sample actions](media/images/hmdb51-sample.png){ #fig:dataset:hmdb51:samples }


### UCF101 - Action recognition

UCF101[@soomro2012_UCF101Dataset101] is an action dataset composed of 101
different actions across 5 broad categories: human-object interaction,
body-motion only, human-human interaction, playing musical instruments, and
sports. Each action class has a minimum of 100 example clips associated with it.
The dataset has a diverse range of camera viewpoints, camera motion, object
appearance and pose, illumination conditions making it quite challenging
compared to some of the earlier datasets used for action recognition like KTH.

![UCF101[@soomro2012_UCF101Dataset101] sample actions](media/images/ucf101-sample.pdf)

### BEOID - Bristol Egocentric Object Interaction Dataset

BEOID[@_BristolEgocentricObject,@damen2014_DiscoveringTaskRelevant] is an
human-object interaction dataset composed of videos shot from a head mounted
(egocentric) camera where the operator performs actions in one of 6 different
locations: kitchen, workspace, printer, corridoor with locked door, cardiac
gymn, and weight-lifting machine.

![BEOID[@_BristolEgocentricObject] sample object interactions]](media/images/beoid-sample.png)

## Understanding CNNs {#sec:background:understanding}

It is typical for CNNs to have on the order of $10^7$--$10^9$ parameters, with
this complexity comes a difficulty in understanding how the network works. There
is a need to understand why a network correctly classifies some examples but not
others to aid the researcher in determining higher performing architectures, and
problems in the dataset or training process.

There is a substantial body of work CNN visualisation techniques and at first
glance it can seem there are many different methods, however most methods fit
into one of the following four main categories of visualisation:

* **Attention mapping**, generating a heatmap over the input indicating which
  regions contribute to activation of a certain neuron or set of neurons.
* **Layer code analysis**, given a layer code, the output values computed at
  specific layer, what is an input to the network that results in this layer code.
* **Activation optimisation**, given a neuron, or set of neurons, determine the
  input to the network that maximally or minimally excites the neurons.
* **Filter analysis**, visualisation techniques based at the filter level: e.g.
  filter response and filter visualisation.

<##todo Read over whole section and think about whether using input or image
makes more sense for each case and try and be consistent throughout>

### Filter analysis

<##todo Think of a better name than Filter analysis>

**Filter visualisation** is the process of taking a filter and visualising it as
an image. Typically this is most useful at the first layer in the network where
the input to the filters are the raw input to the CNN (e.g. images), this allows
us to draw direct comparisons with other filters used on similar inputs like
edge detection filters for images thus giving us some insights to what the first
layer is doing. [@zeiler2013_VisualizingUnderstandingConvolutional] visualises
the first layer filters of AlexNet[@krizhevsky2012_Imagenetclassificationdeep]
demonstrating that there are a number of 'dead' filters, uniform filters that
don't compute any useful transform. Zeiler \etal{} empirically establish a new
architecture which learns fewer dead filters in the first layer using first
layer filter visualisation to check this.

**Filter response** involves visualising the response of a filter after
application to a specific input, similar to *filter visualisation* this gives us
insight as to what transformation the filters are computing: edge detection,
contrast encoding etc. This tends to be most useful at lower layers in the
network where the outputs from each layer are still recognisable as some
transformation of the input image. Yosinski \etal{} introduce a tool called Deep
Visualisation Toolbox ^[https://github.com/yosinski/deep-visualization-toolbox]
in [@yosinski2015_UnderstandingNeuralNetworks] capable of showing filter
responses for arbitrary networks and input images in realtime (futhermore the
tool can also compute the deconvolution and gradient attention maps). Yosinski
\etal{} emphasise the importance of analysing all filters simultaneously as
individual filters shouldn't be considered on their own but in the context of
all the filters of the layer as this is the way the next layer uses them.

### Activation optimisation

Activation optimisation covers a broad range of visualisation techniques used to
optimise the activation of a neuron, usually we want to maximally activate a
neuron to determine what features it recognises. Formally, to find an image that
maximally activates a single neuron. Let $h_j^{(i)}(\bm{x})$ define the
activation of neuron $j$ in layer $i$ when the network forward propagates the
input $\bm{x}$. Now we want to find $\bm{x}^{*} = \argmax_x h_j^{(i)}(\bm{x})$,
this is not necessarily unique as many different inputs can cause maximal
activation of a neuron. Whilst this may seem like a fairly simple optimisation
problem the main challenge is to produce a 'natural' looking image, without any
form of regularisation this optimisation produces extremely noisy unrecognisable
images. Most of the prior work on this technique involves the proposal of
different priors used to constrain the optimisation to produce interpretable images.

Erhan \etal{} were the first to introduce *activation maximisation* for
generating an artificial image to maximise the activation of a chosen neuron by
performing gradient ascent on an input
image[@erhan2009_VisualizingHigherLayerFeatures]. The authors only constrict the
optimisation problem by ensuring that $||\bm{x}|| = \rho$, where $\rho$ is used
to bound the magnitude of the generated image. To evaluate the method, they
apply the technique to a deep belief network[@hinton2009_Deepbeliefnetworks]
(DBN) and a stacked denoising
autoencoder[@vincent2010_StackedDenoisingAutoencoders] (SDAE) trained on the
MNIST dataset of 60,000 hand written digits. Images generated for neurons in
lower levels showed blobs and simple patterns whereas neurons deeper in the
network produced recognisable digit images indicating that the neurons have
learnt higher level features from the combinations of lower ones.

Simonyan
\etal{} investigated activation maximisation using L2 regularisation for object
detection CNNs[@simonyan2013_DeepConvolutionalNetworks] producing images with
interpretable outlines but inaccurate colouring.

Mahendran \etal{} investigate
the use of priors to restrict the generated image to look 'natural' (i.e. not
computer generated) for layer code inversion (see [@sec:vis:code-inversion]).
They use the L_n norm and total variation of the generated
image[@mahendran2014_UnderstandingDeepImagea] producing images with more
accurate colouring than Simonyan \etal{}'s method. They later expand on their
previous work in [@mahendran2016_VisualizingDeepConvolutional] recognising the
general applicability of natural image priors to layer code inversion,
activation maximisation and caricaturization. They note that pixels in a
generated image should be constrained within a certain range, so propose a
*bounded range* regularisation term to ensure the pixels are limited to be no
greater than a chosen threshold. The also use jittering as proposed in
[@2015_InceptionismGoingDeeper] which shifts the generated image between steps
of the gradient ascent optimisation based on the assumption that the image
should still produce a strong activation of the neuron if the edges are
occluded.

Nguyen \etal{} propose an innovative method to encode the prior that the image
should be 'natural' by use of a deep generative network (DGN) trained to invert
a deep feature map back to image space[@nguyen2016_Synthesizingpreferredinputs].
A DGN $G$ for a network $N$ is trained to produce an input $\bm{x}$ to $N$ from
a feature map $\bm{m}$ from layer $l$ in $N$ such that forward propagating
$\bm{x}$ in network $N$ will produce $\bm{m}$ at layer $l$, i.e. the goal of a
DGN is to invert a feature map back to input space for a given network. Instead
of performing gradient ascent in the input space to generate an input $\bm{x}^*$
to maximally activate $h_j^{(j)}(\bm{x})$ we instead perform it in the layer
code space and connect the output of the $G$ to $N$ forming a chain of these two
networks. The use of the DGN acts as a strong prior for producing natural
inputs since the DGN has in effect learnt what makes an image 'natural' or not.

Nguyen \etal{} also make an interesting discovery of multi-faceted
neurons[@nguyen2016_MultifacetedFeatureVisualization], neurons which are
activated from multiple features and introduce a method to search for the
different features that activate the neuron.

An interesting take on the idea of activation maximisation is to try and fool
the network by generating inputs that are classified incorrectly with almost
perfect confidence that to the human eye are completely incorrect, In
[@nguyen2014_DeepNeuralNetworksa] Nguyen \etal{} generate a series of images of
abstract patterns that are classified with almost perfect confidence on ImageNet
trained object detection networks.

<##todo Add visual comparison of different activation maximisation techniques>

A simple technique for gauging what features a neuron might have learnt is to
generate a set of inputs by searching through the examples used to train the
network and collecting the top-$n$ example that minimally and maximally excite a
chosen neuron. The variety of images in the top-$n$ excitation example set give
clues to the invariants of the neuron.

### Feature map inversion {#sec:vis:feature-map-inversion}

Feature maps (a.k.a CNN codes) are the outputs produced by a layer for use by
the next layer. In feature map inversion we try to determine an input $\bm{x}$
to the network to produce a given feature map $\bm{m}$ at layer $l$. This can be
seen as a generalisation of activation maximisation where activation
maximisation is feature map inversion for a one hot feature maps.

Mahendran \etal{} were the first to investigate the inversion of feature maps in
object detection CNNs[@mahendran2014_UnderstandingDeepImagea] using gradient
descent to solve the following minimisation problem: given an input $\bm{x}$,
which forward propagated to through the network to layer $l$ produces a feature
map $\bm{m}$, find an input $\bm{x}^*$ from inverting $\bm{m}$ minimising the
loss $\ell(\bm{x}, \bm{x}^*)$. Similar to activation maximisation they also make
use of priors to ensure the inversion looks like a 'natural' image. The authors
elaborate on their approach in [@mahendran2016_VisualizingDeepConvolutional]
with the same approach but with new priors (the same as those discussed in the
activation optimisation section) to produce more natural looking images.

Dosovitskiy \etal{} use up-convolutional networks to invert feature maps
[@dosovitskiy2015_InvertingVisualRepresentationsa]. The authors train a decoder
network on ImageNet images and feature maps from
AlexNet[@krizhevsky2012_Imagenetclassificationdeep] producing an up
convolutional network. In [@nguyen2016_Synthesizingpreferredinputs] Nguyen
\etal{} propose a DGN for use as a natural image prior in activation
maximisation, however there is no reason that this couldn't also be used in the
same manner as the up-convolutional decoder network for feature map inversion.

### Attention mapping

An *attention map* for a given input $\bm{x}$ and trained CNN is a heatmap
indicating the regions in the input that contribute to its classification.

An occlusion study[@zeiler2013_VisualizingUnderstandingConvolutional] is a
simple but computationally expensive method for determining the regions of an
input that contribute to its classification. For a specific input $\bm{w}$ a
region is occluded by a mask smaller than the input size, a forward pass is
computed, and the confidence in the ground truth class is recorded, the
occluding region is then slid along the input into a new position and the
process repeated over the whole input producing a tensor recording the class
confidences at each location which can then be used as a heatmap.

Simonyan \etal{}[@simonyan2013_DeepConvolutionalNetworks] observe that the
weights of a linear classifier can be interpreted as the relative importance of
the components feature vector. Since CNNs compute a non-linear transform of
their input so the same technique cannot be used, however a linear approximation
$\hat{\bm{w}} \cdot \bm{x}$ to the network about a specific input $\bm{x}$ can
be computed using a first order Taylor expansion whose weights $\hat{\bm{w}}$
can be interpreted as the importance of the corresponding input elements.

In [@zeiler2013_VisualizingUnderstandingConvolutional], Zeiler & Fergus
introduce deconvolutional visualisation in which an input is propagated through
the network, the neuron for visualisation is chosen and a deconvolutional
network[@zeiler2010_Deconvolutionalnetworks] constructed from the
network-under-analysis' weights is attached to the layer in which the neuron of
interest resides. All other neurons in the layer of the chosen neuron are set to
zero to produce a one-hot CNN code which is used as input to the deconvolutional
network that progressively inverts the operation of the original network until
the CNN code is fully inverted back into an image. The resulting image retains
aspects of the original image in areas that contribute to the activation of the
chosen neuron. To invert a convolutional layer $l_c$, a corresponding
convolutional layer $l_c'$ is constructed in the deconvolutional network where
the filters from $l$ are transposed in $l_c'$ and the input to $l_c'$ is the
output of $l_c$. Rectified linear unit (ReLU) layers are inverted by also
applying ReLU, the idea being that a ReLU layer ensures that the output of the
layer is non negative, to preserve this property that the output of a layer is
non negative in the deconvolutional network we too have to add a ReLU layer.
Pooling layers are inverted by recording the location in the filter from which
the max activation originated from, consider the following example: in a pooling
layer with $2 \times 2$ filters, index each location in the filter by $i$, let
$i_{\text{max}}$ by the index of the location from which the maximum value
originates. When inverting the network, the value to by distributed back to the
$2 \times 2$ grid is entirely given to location $i_{\text{max}}$. Yu \etal{}
make a qualitative comparison in [@yu2014_VisualizingComparingConvolutional]
between AlexNet[@krizhevsky2012_Imagenetclassificationdeep] and
VGG16[@simonyan2014_VeryDeepConvolutional] using Deconvolutional visualisations
of neurons in different layers showing that the deeper layers in VGG16 learn
more discriminate features than those in AlexNet.


<##todo Reword
explanation of deconv with diagrams>

<##todo Add figures explaining deconvolution>

Excitation backprop[@zhang2016_TopdownNeuralAttention] is addressed in the
following section ([@sec:ebp]).

<!--
Samek 2015: Evaluating the visualisation of what a deep network learned

* "Layer-wise relevance propagation provides a better explanation of what made a
  DNN arrive at a particular classification decision than the sensitivity based
  approach or the deconvolution method"
* Looks at these visualisation techniques as a way to determine why the DNN
  predicted the class it did and not others
* Criticises deconvolution for *only* using image specific information when
  going backwards through ReLU layer, I think they want it to use more
  information so the results are more heavily tied to that image
* Crticises taylor series expansion of partial derivatives to produce image
  indicating importance of pixels for classification (simonyan et al) as it
  "tells us what change would make the image more or less belong to the category
  car" and not "what speaks for the presence of a car in the image"
* Uses LVP
-->
Layerwise relevance propagation[@bach2015_PixelWiseExplanationsNonLinear;@samek2015_Evaluatingvisualizationwhat;@lapuschkin2016_LRPToolboxArtificial]


# EBP {#sec:ebp}

First a forward pass of the network is computed, this produces the intermediate
neuron values which are used as *bottom up* salience factors, then a probability
distribution over the output layer is used to specify *top down* salience, then a
excitation backprop pass uses the probability distribution, intermediate neuron
values and weights to determine the probability of each intermediate neuron
being a winner at an arbitrary depth of the network.

Contrastive top down attention uses the insight that we're not only interested
in class we're localising, but also the absences of the other classes (as
classes may be correlated), we EBP the class of interest one layer, then invert
the output probability distribution, EBP one layer and compute the difference
between the two MWPs of the second last layer, then EBP from there to the input.

<!--
![Excitation Backprop winning probability aggregated from multiple parents to single child](media/images/excitation-bp-parent-sharing.png)

![Excitation Backprop winning probability shared between multiple children](media/images/excitation-bp-child-sharing.png))
-->


## Example EBP calculation

We demonstrate EBP with a simple network composed of 5 neurons over 3 layers all
using ReLU activations.

Notation:

* $\neuron{i}{j}$ denotes the neuron with index $j$ (0 indexed) in layer $i$.
* $\weight{i}{j}{k}$ denotes the weight of the edge from neuron $j$ in layer $i$
  to neuron $k$ in layer $i + 1$.
* $\neuroninput{i}{j}$ denotes the weighted sum of inputs to neuron $j$ in layer $i$
* $\neuronoutput{i}{j}$ denotes the output of neuron $j$ in layer $i$

\begin{equation}
\label{eq:neuron-input}
\neuroninput{i + 1}{j} = \sum_{a_{k}^{(i)} \in \children{i + 1}{j}} \weight{i}{k}{j} \neuronoutput{i}{k}
\end{equation}

\begin{equation}
\label{eq:neuron-output}
\neuronoutput{i}{j} = \phi(\neuroninput{i}{j})
\end{equation}

Where $\phi$ is an activation, if not explicitly stated it is assumed $\phi(x) =
\max(0, x)$ (ReLU activation).

At a high level, EBP consists of the following steps:

* Compute a forward pass of the network to determine the outputs of each neuron $\neuronforward{i}{j}$
* Compute the scaling factors $\ebpscalar{i}{j}$ of each neuron used in
  calculating the conditional winning probabilities of the children of that neuron.
* Compute the conditional winning probabilities $\cwp{i}{j}{i + 1}{k}$ of each neuron in the network.
* Compute the winning probabilities of each neuron $\mwp{i}{j}$ by
  computing the probability of each neuron and it's parents being winning
  neurons, then marginalising over the parent neurons.


\begin{equation}
\label{eq:ebp-cwp}
\cwp{i}{k}{i + 1}{j} = \begin{cases}
    \ebpscalar{i + 1}{j} \neuronforward{i}{k} \weight{i}{k}{j} & \weight{i}{k}{j} \geq 0 \\
    0 & \text{otherwise}
  \end{cases}
\end{equation}

\begin{equation}
\label{eq:ebp-cwp-scalar}
\ebpscalar{i + 1}{j} = 1 / \sum_{k:\weight{i}{k}{j} \geq 0} \neuronforward{i}{k} \weight{i}{k}{j}
\end{equation}

\begin{equation}
\label{eq:ebp-mwp}
\mwp{i}{k} = \sum_{\neuron{i+1}{j} \in \parents{i}{k}} \cwp{i}{k}{i + 1}{j} \mwp{i + 1}{j}
\end{equation}

Performing excitation backprop on the example network in [@fig:ann-example]. The
forward pass is detailed in [@fig:ann-forward].

First we define the input of the network (these could be any arbitrary input):

\begin{align*}
\neuronforward{0}{0} &= 2\\
\neuronforward{0}{1} &= 1\\
\end{align*}

Now we compute the forward pass using the forward propagation rule

<##todo move this to ANN section>
$$\neuronforward{i}{k} = \phi(\sum_j \neuronforward{i - 1}{j} \cdot \weight{i - 1}{j}{k})$$

\begin{align*}
\neuronforward{1}{0} &= \max(0, \neuronforward{0}{0} \cdot \weight{0}{0}{0} +
    \neuronforward{0}{1} \cdot \weight{0}{1}{0})
  = max(0, (2 \cdot 1) + (1 \cdot 0)) = 2 \\
\neuronforward{1}{1} &= \max(0, (2 \cdot -1) + (1 \cdot 1)) = \max(0, -1) = 0\\
\neuronforward{1}{2} &= \max(0, (2 \cdot 1) + (1 \cdot 1)) = 3\\
\\
\neuronforward{2}{0} &= \max(0, \neuronforward{1}{0} \cdot \weight{1}{0}{0} +
    \neuronforward{1}{1} \cdot \weight{1}{1}{0} +
    \neuronforward{1}{2} \cdot \weight{1}{2}{0}) = 4 \\
\neuronforward{2}{1} &= \max(0, (2 \cdot 1) + (0 \cdot 2) + (3 \cdot -1)) = 0\\
\end{align*}

The next step is to compute the conditional winning probabilities of each neuron
given each parent neuron wins using [@eq:ebp-cwp], to compute this we need the
scaling factors $\ebpscalar{i}{j}$ which we will compute first using
[@eq:ebp-cwp-scalar] (in a computational implementation these would be computed
on a per layer basis and thrown away once the layer values are
calculated).

![Flow of probabilities in EBP](media/images/ebp-example-mwp.pdf)

![EBP CWP on the example network](media/images/ebp-example-mwp-concrete.pdf)

<##todo change legend to have $P(a_k^{(i)})$ in diamond>

\begin{align*}
\ebpscalar{2}{0} &=
  \frac{1}{\left(\weight{1}{1}{0}\neuronforward{1}{1}\right) +
  \left(\weight{1}{2}{0} \neuronforward{1}{2}\right)}
  = \frac{1}{(1 \cdot 0) + (2 \cdot 3)}
  = \frac{1}{6}
  \\
\ebpscalar{2}{1} &= \frac{1}{(1\cdot 2) + (2 \cdot 0)} = \frac{1}{2}\\
\ebpscalar{1}{0} &= \frac{1}{
  \left(\weight{0}{0}{0} \neuronforward{0}{0}\right) +
  \left(\weight{0}{1}{0} \neuronforward{0}{0}\right)}
  = \frac{1}{(1 \cdot 2) + (0 \cdot 1)}
  = \frac{1}{2}
  \\
\ebpscalar{1}{1} &= \frac{1}{(1 \cdot 1)} = 1\\
\ebpscalar{1}{2} &= \frac{1}{(1 \cdot 2) + (1 \cdot 1)} = \frac{1}{3}\\
\end{align*}

Now for the conditional winning probabilities between layers 2 and 1:

\begin{align*}
\cwp{1}{0}{2}{0} &= 0 \\
\cwp{1}{0}{2}{1} &= \ebpscalar{2}{1} \neuronforward{1}{0} \weight{1}{0}{1} =
  \frac{1}{2} \cdot 2 \cdot 1 = 1
  \\
\cwp{1}{1}{2}{0} &= \ebpscalar{2}{0} \neuronforward{1}{1} \weight{1}{1}{0} =
  \frac{1}{6} \cdot 0 \cdot 1 = 0
  \\
\cwp{1}{1}{2}{1} &= \ebpscalar{2}{1} \neuronforward{1}{1} \weight{1}{1}{1} =
  \frac{1}{2} \cdot 0 \cdot 2 = 0
  \\
\cwp{1}{2}{2}{0} &= \ebpscalar{2}{0} \neuronforward{1}{2} \weight{1}{2}{0} =
  \frac{1}{6} \cdot 3 \cdot 2 = 1
  \\
\cwp{1}{2}{2}{1} &= 0\\
\end{align*}

Now layers 1 and 0:

\begin{align*}
\cwp{0}{0}{1}{0} &= \ebpscalar{1}{0} \neuronforward{0}{0} \weight{0}{0}{0} =
  \frac{1}{2} \cdot 2 \cdot 1 = 1\\
\cwp{0}{0}{1}{1} &= 0 \\
\cwp{0}{0}{1}{2} &= \frac{1}{3} \cdot 2 \cdot 1 = \frac{2}{3} \\
\cwp{0}{1}{1}{0} &= 0 \\
\cwp{0}{1}{1}{1} &= 1 \cdot 1 \cdot 1 = 1 \\
\cwp{0}{1}{1}{2} &= \frac{1}{3} \cdot 1 \cdot 1 = \frac{1}{3} \\
\end{align*}

We can now marginalise over the parent neurons in the conditional winning
probabilities if a prior distribution over the output neurons is given to obtain
the marginal winning probabilities of each neuron using [@eq:ebp-mwp].

Let's choose $\mwp{2}{0} = 0.9$ and $\mwp{2}{1} = 0.1$ for the prior
distribution. If we were investigating the saliency of a single neuron we'd
instead set the MWP of that neuron to 1 and the MWP of all other neurons
would be 0.

Marginalising over the parents of the hidden layer:

\begin{align*}
\mwp{1}{0} &= \sum_{\neuron{2}{j} \in \parents{1}{0}} \cwp{1}{0}{2}{j} \mwp{2}{j} \\
           &= \cwp{1}{0}{2}{0} \mwp{2}{0} + \cwp{1}{0}{2}{1} \mwp{2}{1} \\
           &= 0 \cdot 0.9 + 1 \cdot 0.1 = 0.1\\
\mwp{1}{1} &= 0 \cdot 0.9 + 0 \cdot 0.1 = 0\\
\mwp{1}{2} &= 1 \cdot 0.9 + 0 \cdot 0.1 = 0.9\\
\end{align*}

Finally to calculate the MWP of the input neurons to obtain the posterior
distribution:

\begin{align*}
\mwp{0}{0} &= 1 \cdot 0.1 + 0 \cdot 0 + \frac{2}{3} \cdot 0.9 = 0.7\\
\mwp{0}{1} &= 0 \cdot 0.1 + 1 \cdot 0 + \frac{1}{3} \cdot 0.9 = 0.3\\
\end{align*}


# Excitation backprop for temporal networks {#sec:ebp-for-temporal-networks}

<##todo Results from EBP on TSCNN trained on UCF101 and BEOID>

# Future work {#sec:future-work}

* Applying activation maximisation on temporal networks with a prior encoding similarity
  over the multiple frames to generate videos.
* Train DGN to invert temporal network to generate videos (like in [@nguyen2016_Synthesizingpreferredinputs])
* Use deepdraw a la TSN paper to visualise actions of my networks

<!--
Karpathy 2014 notes

* Introduction of Sports-1M collected from YouTube with 487 classes
* Comparison of single frame to multiple frame networks
* Only slightly better performance for multiframe models
* Low res context stream with high res fovea stream inspired by the eye to
  improve training speeds
* Transfer learning from networks trained on Sports-1M to UCF101
* 178x178 input, down sampled to 89x89 for context stream and center cropped to
  89x89 for fovea stream obtaining similar accuracy to full 178x178 stream but
  with quicker training times
* 4 architectures: single frame, late fusion, early fusion, slow fusion
* motion aware networks underperform when there is camera motion
* Slow fusion network performs best
-->

![CNN Architectures evaluated in [@karpathy2014_LargeScaleVideoClassification],
layer colours indicate function: red--convolutional, green--normalization,
blue--pooling. The bottom white boxes indicate a series of frames that are used
as input to the CNN](media/images/karpathy2014-fusion-architectures.png){#fig:karpathy2014-fusion-architectures}

Investigations of different architectures for video classification were
performed in [@karpathy2014_LargeScaleVideoClassification]. Four different
styles of architecture were investigated to determine optimal stages of fusing
temporal and spatial information. Each architecture had a different connectivity
to the video sequence stack, from using a single frame as input to a dense
sub-sequence of frames (see [@fig:karpath2014-fusion-architectures] for
architectures and video sequence connectivity). Slow fusion, an architecture
that progressively enlarges the temporal and spatial field of view as the input
propagates deeper into the network performed best.

<##todo expand on this>


<!--
Tran 2014

* 3D CNN better than 2D CNN
* 3x3x3 kernels in all layers obtains best results in all architectures tested
  in paper
* Says Karpathy 2014 used 2D Convolutions in all architectures but slow fusion
  which they hypothesis is why it performed best
* Investigate varying depth of 3D convolution whilst holding other
  hyperparameters constant
* *visualisation* deconvolution of network
* Call their architecture C3D
-->

<!--
Feichtenhofer 2016
* Extension of Simonyan 2014 architecture
* Investigation of fusion of networks
* Fusing at last convolutional layer gives best performance
* Improve performance further by keeping both streams and fusing at the end too
* State that  Wang 2015 and Tran 2014 have current SoA methods for UCF101 and HMDB51
* Identifies deficiency of basic two stream network in that it can't see what is
  happening where (caused by the late fusion)
* By fusing earlier, the network can correlate movement with appearance and thus
  rectify the deficiency of not being able to see what is moving where in the
  basic two stream network
* New architecture beats SoA (i.e. better than Wang 2015 and Tran 2014)
* Still see an improvement when combining the new network with IDT features
-->

In [@feichtenhofer2016_ConvolutionalTwoStreamNetwork], the authors extend the
architecture presented in [@simonyan2014_TwoStreamConvolutionalNetworks] by
observing that the previous architecture is incapable of matching appearance in
one sub-region to motion in another sub-region since each stream is separate, to
remedy this, the introduce a modified architecture in which the two streams are
combined after the last convolutional layers resulting in a single
spatio-temporal stream from the fully connected layers onwards. The authors find
that keeping the spatial stream in addition to the spatio-temporal stream and
combining their respective predictions further increases performance over
predictions from the spatio-temporal stream alone.
<##todo Report performance improvements over fusing into single tower>

# Glossary

ANN
: Artificial Neural Network

CNN
: Convolutional Neural Network

DNN
: Deep artificial neural network (one with multiple hidden layers)

EBP
: Excitation Backpropagation

Top down attention
: Attention driven by top down factors like task information

Bottom up attention
: Attention based on the salience of regions of the input image.

Excitation Map
: A heat map over an image denoting the regions contributing contributing to its
classification

# Notation

A full listing of all notation used.

$\learningrate$

: Learning rate (e.g. see algorithm \ref{alg:perceptron-training})

$\neuron{l}{j}$
: Neuron in layer $l$ (0 indexed) at position $j$ (0 indexed)

$\neuronforward{l}{j}$
: The result of forward propagation at the neuron in layer $l$ at position $j$

$\weight{l}{j}{k}$
: The weight connecting neuron $\neuron{l}{j}$ to neuron $\neuron{l + 1}{k}$

$\ebpscalar{l}{j}$
: The scaling factor used in calculating the conditional probabilities in EBP
ensuring that the probabilities sum to one.

$\children{l}{j}$
: The child neurons (those in layer $l - 1$) of the neuron in layer $l$ with index $j$

$\parents{l}{j}$
: The parent neurons (those in layer $l + 1$) of the neuron in layer $l$ with index $j$

$\cwp{l}{j}{l + 1}{k}$
: The *conditional winning probability* of $\neuron{l}{j}$ given that $\neuron{l
+ 1}{k}$ is winning neuron (see EBP).

$\mwp{l}{j}$
: The *marginal winning probability* of $\neuron{l}{j}$ (see EBP)

$\neuroninput{l}{j}$
: The input to neuron $\neuron{l}{j}$.

$\neuronoutput{l}{j}$
: The output of neuron $\neuron{l}{j}$

# References
