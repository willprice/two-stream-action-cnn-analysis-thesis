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

## Video analysis
### UCF101

* v_BalanceBeam_g07_c04 shows a good difference between temporal contrastive vs
  noncontrastive where contrastive seems to be superior with less attention
  distributed to edges
* v_Basketball_g05_c04 spatial highlights basketball hoop and ball, but
  contrastive doesn't
* v_Biking_g02_c03, large fluctuations in spatial contrastive attention map,
  much more localised in spatial EBP (tracks same object throughout the video)
* v_Billiards_g06_c05 similar to above, temporal non-contrastive highlights more
  of the balls than contrastive. Spatial EBP highlights balls and border of
  table
* v_Bowling_c01_g01 spatial non-contrastive highlights the skittles, contrastive
  seems to highlight similar things but jumps about from frame to frame

* v_CliffDiving_g01_c05 remarkably good performance from both spatial and
  temporal, spatial contrastive not so good

* v_CuttingInKitchen_g01_c01, temporal does a good good localising the whole
  action, spatial seems to love the bottom right corner of the onion (onion
  white contrasts with shadowed background)

* v_Diving_g05_c02, v_Diving_g05_c05 good demonstration of diving board
  preference, diving boards aren't a necessity to the 'diving' action, but
  examples include them and are highlighted prominently in the attention maps

* v_Fencing_g07_c04 good example of poor spatial contrastive action
  localisation

* v_GolfSwing_g03_c06, spatial learnt environment more than action

* v_Hammering_g07_c05 good temporal localisation

* v_HammerThrow_g02_c07 spatial poor action localisation, temporal good
  indicates spatial has learnt to recognise environment and temporal the action itself

* v_HighJump_g01_c02 similar to above, temporal does good action localisation
  and spatial recognises environment

* v_HorseRiding_g04_c04 attention maps very similar across all

* v_JumpRope_g03_c03 good localisation but for spatial noncontrastive

* v_JumpRope_g04_c07 guy skipping in front of mirror, temporal contrastive primarily
  localises the guy, not the reflection, non-contrastive highlights both



* It seems spatial contrastive highlights a subset of what is present in the
  contrastive frame, but that between frames the highlighted objects are
  different whereas in non-contrastive the salient objects are much more constant

## Jitter

Approach:

* Consider the top 2 of each spatial/temporal contrastive/non constrative max/min
  jitter
* watch the video 4 times
* categorise whether the attention maps are good for SC, SNC, TC, TNC
* decide whether the attention maps demonstrate interesting behaviour that the
  previous attention maps don't
  * If they don't go to the next video
  * If they do, produce a graphic with interesting frames ideally with a
    comparison that demonstrates good behaviour
  * Discuss the interesting frames
    * Discuss what we would expect to be highlighted for correct classification
    * Contrast with what we see
    * Give a possible explanation for the behaviour.

### UCF101

| Video                       | Jitter     | SC | SNC | TC  | TNC | Interesting | Frames of interest | Behaviour                                                                               |
|-----------------------------|------------|----|-----|-----|-----|-------------|--------------------|-----------------------------------------------------------------------------------------|
| v_PlayingGuitar_g05_c01	  | SC low     | G  | G   | VB  | G   | 2/5         | All                | SC very stable, TC variable                                                             |
| v_PlayingFlute_g03_c05      | SC low     | IG | G   | G   | G   | 4/5         |                    | SC flips between face & flute                                                           |
| v_PlayingTabla_g07_c01      | SNC low    | B  | G   | G   | G   | 3/5         |                    | SC flips between tabla and face, SNC most stable, TC & TNC good                         |
| v_WritingOnBoard_g04_c03    | SNC low    | IG | G   | G   | G   | 5/5         |                    | SNC highlight writing not action, SC flips between patches of writing                   |
| v_RopeClimbing_g05_c07      | TC low     | G  | G   | G   | G   | 5/5         |                    | TC very good, doesn't highlight person unlike TNC                                       |
| v_Bowling_g01_c01           | TC low     | VB | G   | G   | G   | 2/5         |                    | SC all over the place, SNC recognises alley, T good, mix of skittles OF failure and arm |
| v_Hammering_g07_c05	      | TNC low    | B  | O   | G   | G   | 3/5         |                    | TNC very well localised to hammering action, TC good too                                |
| v_SoccerPenalty_g01_c06     | TNC low    | B  | G   | IB  | G   | 4/5         |                    | TNC excellent, TC not bad either, SC all over the place                                 |
|                             |            |    |     |     |     |             |                    |                                                                                         |
| v_Swing_g06_c07             | S(N)C high | B  | O   | O   | B   | 2/5         |                    | Horrible, very noisy                                                                    |
| v_TennisSwing_g07_c02       | SC high    | VB | O   | B   | O   | 3/5         |                    | TC crap, TNC OK                                                                         |
| v_Mixing_g05_c04	          | SNC high   | VB | VB  | O   | B   | 3/5         |                    | TC OK, TNC bad, good example                                                            |
| v_Haircut_g06_c01           | TC high    | VB | G   | O   | O   | 3/5         |                    | Neither TC/TNC are good, hard to interpret what is recognised, falling hair?            |
| v_Mixing_g01_c04            | TC high    | O  | VG  | IVB | G   | 3/5         |                    | SC noisy, TC looks at reflection and not real action!                                   |
| v_Knitting_g07_c05	      | TNC high   | B  | G   | IG  | G   | 4/5         |                    | TC jumpy, but still good                                                                |
| v_CuttingInKitchen_g01_c01  | TNC high   | B  | G   | B   | O   | 2/5         |                    |                                                                                         |

| SC | SNC | Video                                   |
|----|-----|-----------------------------------------|
| G  | G   | v_PlayingFlute_g03_c05                  |
| G  | B   |                                         |
| B  | G   | v_SoccerPenalty_g01_c06                 |
| B  | B   | v_TennisSwing_g07_c02, v_Mixing_g01_c04 |

Spatial Interesting:
* v_WritingOnBoard_g04_c03
* v_RopeClimbing_g05_c07


| TC | TNC | Video                                       |
|----|-----|---------------------------------------------|
| G  | G   | v_PlayingFlute_g03_c05, v_Hammering_g07_c05 |
| G  | B   | v_RopeClimbing_g05_c07                      |
| B  | G   | v_PlayingGuitar_g05_c01	                 |
| B  | B   | v_Swing_g06_c07                             |
|    |     |                                             |


To show
| Video                       | SC | SNC | TC | TNC | Frames (start,stop,step) |
|-----------------------------|----|-----|----|-----|--------------------------|
| v_PlayingFlute_g03_c05      | x  | x   | x  | x   | 6,16,2                   |
| v_Hammering_g07_c05         |    |     | x  | x   | 6,26,4                   |
| v_RopeClimbing_g05_c07      | x  | x   | x  | x   | 144,154,2            |
| v_PlayingGuitar_g05_c01	 |    |     | x  | x   | 133,143,2            |
| v_Swing_g06_c07             |    |     | x  | x   | 35,45,2              |
| v_SoccerPenalty_g01_c06     | x  | x   | x  | x   | 47,57,2                  |
| v_TennisSwing_g07_c02       | x  | x   |    |     | 203,213,2                |
| v_Mixing_g01_c04            | x  | x   |    |     | 61,65,1                |
| v_WritingOnBoard_g04_c03    | x  | x   |    |     |  50,54,1                 |


### BEOID

| Video                                     | Jitter      | SC | SNC | TC  | TNC | Interesting                                       | Behaviour |
|-------------------------------------------|-------------|----|-----|-----|-----|---------------------------------------------------|-----------|
| 06_Treadmill1_press_button_4469-4493	  | L SC        | B  | VB  | B   | B   | Highlights foot rather than button                |           |
| 07_Treadmill1_press_button_193-305	    | L SC        | VB | G   | B   | G   | Contrastive ignores finger press                  |           |
| 03_Sink2_stir_spoon_1793-1887	         | L SNC       | VB | G   | G   | G   | TC localises end of spoon movement                |           |
| 07_Row2_push_rowing-machine_844-904	   | L SNC       | O  | O   | O   | G   | TC skippy, nob recognised                         |           |
| 04_Sink1_press_button_800-835	         | L TC, L TNC | B  | B   | IVB | O   | Reflection, nob, TNC localises reflection + press |           |
| 04_Door2_open_door_284-333	            | L TC        | G  | G   | B   | O   | TNC localises door disparity                      |           |
| 01_Sink2_press_button_527-561	         | L TNC       | VB | IVB | IG  | B   | TC much better than TNC                           |           |
| 05_Row2_pull_rowing-machine_2030-2060	 | H SC        | VB | G   | O   | O   |                                                   |           |
| 02_Sink2_pick-up_jar_1003-1027	        | H SNC       | B  | IB  | O   | G   | SNC localises rowing machine handle               |           |
| 01_Desk2_pick-up_tape_957-998	         | H SNC       | IB | IO  | O   | IG  | Large camera movements, spatial recognise env     |           |
| 02_Sink1_scoop_spoon_1294-1332	        | H TC        | O  | O   | IO  | IO  | Temporal, noisy OF microwave                      |           |
| 00_Sink1_turn_tap_694-717	             | H TC        | IG | IO  | B   | IG  | Localise hand turn quite well                     |           |
| 01_Sink1_turn_tap_406-441                 | H TNC       | B  | O   | B   | O   |                                                   |           |
| 06_Row2_pull_rowing-machine_613-641	   | H TNC       | B  | G   | B   | VG  |                                                   |           |

| SC | SNC | Video                              |
|----|-----|------------------------------------|
| G  | G   | 04_Door2_open_door_284-333         |
| G  | B   | 00_Sink1_turn_tap_694-717          |
| B  | G   | 07_Treadmill1_press_button_193-305 |
| B  | B   | 01_Sink2_press_button_527-561      |

| TC | TNC | Video                                |
|----|-----|--------------------------------------|
| G  | G   | 03_Sink2_stir_spoon_1793-1887        |
| G  | B   | 01_Sink2_press_button_527-561        |
| B  | G   | 00_Sink1_turn_tap_694-717            |
| B  | B   | 06_Treadmill1_press_button_4469-4493 |

In addition:
05_Row2_pull_rowing-machine_2030-2060

To show
| Video                                | Spatial | Temporal | Frames (start,stop,step) |
|--------------------------------------|---------|----------|--------------------------|
| 04_Door2_open_door_284-333           | x       |          | 10,30,4                  |
| 07_Treadmill1_press_button_193-305   | x       |          | 42,62,4          |
| 01_Sink2_press_button_527-561        | x       | x        | 9,29,4                  |
| 00_Sink1_turn_tap_694-717            | x       | x        | 9,14,1                   |
| 03_Sink2_stir_spoon_1793-1887        |         | x        | 7,17,2                   |
| 06_Treadmill1_press_button_4469-4493 |         | x        | 1,11,2                   |


Interesting:

* 06_Treadmill1_hold-down_button_331-528 - temporal watches feet as there is no
  motion in the action, SNC covers lots of things, interestingly SC covers a lot
  less, dosen't localise the action, but gets rid of background
*


# ToDo

* [x] Redo graph titles for smoothness -> jitter
* [x] Bound jitter graphs at 0
* [x] Regenerate UCF101 examples with correct frame underlays
* [x] Include UCf101 examples
* [x] Write about UCF101 examples
* [x] Perform BEOID smoothness example selection
* [x] Select frames from BEOID examples
* [x] Generate PDFs of BEOID examples
* [x] Include BEOID examples
* [x] Discuss BEOID examples
* [ ] Make a comparison of underlay, we chose the last frame, trails behind the
  action, otherwise plotting ahead is confusing producing a cognitive
  dissonance, very few things in life have a leading 'trail'
* [x] Add billiards example for underlay choice
* [x] Move filters to results section
* [x] Add filter diff
* [x] Make observations on the filter visualisations
* [x] Make frame underlay choice examples bigger (that big A4) figure
* [x] Explain how the distance between the attention map peak and gaze location
  is calculated
* [x] Add figure demonstrating the above showing what the distance actually is.
* [x] Move jitter analysis future work back of jitter analysis section
* [x] Make sure reference to the concept of jitter is distinct from the
  L2-jitter measure
* [x] Complete quantitative jitter analysis
* [x] Complete quantitative gaze analysis
* [x] Add references to abstract
* [x] Discuss hypothesis as to why contrastive doesn't work well, class overlap,
  common winner neurons
* [x] Revisit definition of jitter in light of insight of regions flickering in and out of existence in the contrastive attention map sequences
* [x]Compare the 'press-button' action across the different locations in BEOID qualitative analysis
* [x] Complete the explanation of jitter caused by contrastive attention
* [x] Complete visualisation hierarchy figure
* [x] Add t-SNE feature map visualisation section to background
* [x] Add caricaturing section to background

## Gaze
* [x] Select interesting examples from gaze analysis, do similar thing to what
  we did for jitter
* [x] Add plots of gaze distance distribution
* [x] Produce a table listing the proportion of frames with gaze-attention map
  peak distance under certain thresholds, e.g. 10/25/50/75/90/100% limits and
  the corresponding bounding distance for which all those frames are under

## Future work
* [x] Add underlay choice to future work, saying we don't know what is optimal.
* [x] Compare attention maps from temporal to spatial?

## Tidy
* [x] Reduce number of frames in figures and make them bigger
* [x] Reduce size of text in figures and make frames bigger
* [x] Move filter analysis to results section
* [ ] Add intro and conclusion in each section. Introduction should outline the
  motivation or the section and what is included, conclusion should summarise
  what was discussed and significant insights.

## Final checklist

* [x] Add short captions to all figures
* [ ] Make titles more descriptive, you should be able to read them and
 understand what is included
* [ ] Final read through note down new concepts and notation add to glossary and
  notation sections
* [ ] Check each section has an introduction
* [ ] Check each section has a conclusion
* [ ] Abstract
* [ ] Conclusion
* [ ] Acknowledgements
* [ ] Spell check
* [ ] Weasel word check
* [ ] Add cool picture to cover page
* [ ] Figure positioning (VERY LAST!)

 Format for future work:
 Title, what is it, how is it currently used, critique on this, propose
 something better


## Final read through notes

* [ ] If I add the filter section then update the abstract.
* [ ]
