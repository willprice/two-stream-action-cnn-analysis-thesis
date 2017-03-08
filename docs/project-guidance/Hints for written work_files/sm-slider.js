// Site manager slider emebd v.0.4
$(document).ready(function(){var _pause='<a class="slider-pause" title="Pause" href="#">Pause</a>';var _pager='<div class="cycle-pager"></div>';var _caption='<div class="cycle-overlay slide-caption"></div>';var _navLinks='<div class="cycle-prev"></div><div class="cycle-next"></div>';var sliderSettings={log:false,timeout:7000,fx:'fade',slides:'> div.slide'};var imagesSettings={log:false,timeout:7000,fx:'scrollHorz'};$('.slider-base').each(function(){var thisSlider=$(this);var numSlides=$('> .slide, > img',thisSlider).length;if(numSlides<2){return};var isImagesSlider=thisSlider.hasClass("slider-images");var hasSlideDescriptions=$('> [data-cycle-desc]',thisSlider).length>0;var pauseBtn=$(_pause);var caption=$(_caption);pauseBtn.click(function(event){event.preventDefault();if(!thisSlider.is('.cycle-paused')){thisSlider.cycle('pause');pauseBtn.attr('title','Play')}
else{thisSlider.cycle('resume');pauseBtn.attr('title','Pause');}});thisSlider.append(pauseBtn);if(hasSlideDescriptions){thisSlider.append(caption);}
if(isImagesSlider){if(!thisSlider.hasClass("_fx-fade")){thisSlider.append(_navLinks);}
else{imagesSettings.fx="fade";thisSlider.append(_pager);}
thisSlider.cycle(imagesSettings);}
else{thisSlider.append(_pager);thisSlider.cycle(sliderSettings);}
if(hasSlideDescriptions){thisSlider.on('cycle-update-view',function(event,optionHash,outgoingSlideEl,incomingSlideEl,forwardFlag){if(!$(incomingSlideEl).attr('data-cycle-desc')){caption.addClass("hideme");}
else{caption.removeClass("hideme");}});}});});