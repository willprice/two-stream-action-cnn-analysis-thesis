/* v.0.9.4a */
var UoBGlobal={};$(document).ready(function(){var self=this;var anchorLink=String(window.location).split("#")[1];$('body').addClass("jsActive");var contentTabs=$('#contentTabs');if(contentTabs.length>0){var ctAnchorLink=anchorLink;self.switchTabs=function(thisLink){var activeTab=$('.active',contentTabs);var activeTabContent=$(activeTab.attr('href'));activeTabContent.animate({opacity:0},150,'linear',function(){activeTabContent.addClass('hiddenContentTab');var newActiveTab=$(thisLink.attr('href'));newActiveTab.removeClass('hiddenContentTab');newActiveTab.css('opacity',0);$('.active',contentTabs).removeClass('active');thisLink.addClass('active');newActiveTab.animate({opacity:1},250,'linear',function(){$(this).css('filter','none');});});thisLink.focus();}
contentTabs.addClass('contentTabs');var ua=$.browser;var safeUA=true;if(ua.msie&&ua.version<=7){safeUA=false;}
if($.address&&safeUA){$.address.strict(false);var canUseAddress=true;}
var tabsList=$('li',contentTabs);if(tabsList.length>0){var hashIsRecognised=false;if(ctAnchorLink){if($('> a[href="#'+ctAnchorLink+'"]',tabsList).length>0){ctAnchorLink="#"+ctAnchorLink;hashIsRecognised=true;}
else{ctAnchorLink=null;}}
$('> a',tabsList).each(function(i){var thisLink=$(this);var tabID=thisLink.attr('href');if(tabID.charAt(0)=="#"){var thisTab=$(tabID);thisTab.addClass('contentTab');if(ctAnchorLink===tabID){thisLink.addClass('active');}
else if(i===0&&!ctAnchorLink){thisLink.addClass('active');thisTab.removeClass('hiddenContentTab').css('opacity',1);thisTab.css('filter','none');}
else{thisTab.addClass('hiddenContentTab').css('opacity',0);}
thisLink.click(function(event){if(!thisLink.hasClass('active')){if(!canUseAddress){self.switchTabs(thisLink);event.preventDefault();}}
else{event.preventDefault();}});}
else{thisLink.addClass("external");}});}
if(canUseAddress){$.address.change(function(event){var val=$.address.value();var update=true;var thisLink=$('a',tabsList).eq(0);if(!val){}
else if($('a[href="#'+val+'"]',tabsList).length>0){var thisLink=$('> a[href=#'+val+']',tabsList);}
else if($('#'+val).length>0){update=false;}
if(!thisLink.hasClass('active')){if(update){self.switchTabs(thisLink);}}});}
if(hashIsRecognised){$(window).scrollTop(0);$('a').eq(0).focus();}}
$('.expandableLinksList').each(function(){var thisLinkList=$(this);$('.sublist').hide();var links=$('.list-heading > a',this);links.click(function(event){var clickedLink=$(this);$('ul:visible',thisLinkList).slideUp(600);if(clickedLink.next('ul:visible',thisLinkList).length<=0){clickedLink.next('ul').slideDown(600);}
event.preventDefault();});});$('.rssFeed').each(function(i){var feedLink=$(this);var numOverride=String(feedLink.attr('class')).match(/_n(?=([0-9]*))/);var showHeader=String(feedLink.attr('class')).match("_noHeader")?false:true;var showDesc=String(feedLink.attr('class')).match("_noDesc")?false:true;var showSnippet=String(feedLink.attr('class')).match("_fullText")?false:true;if(showHeader){var newHeader=feedLink.attr('title')?feedLink.attr('title'):false;}
var numStories=numOverride?numOverride[1]:5;var URL=String(feedLink.attr('href'));var feedDiv=$('<div id="rssFeed'+i+'"></div>');feedLink.replaceWith(feedDiv);feedDiv.rssfeed(URL,{limit:numStories,header:showHeader,content:showDesc,snippet:showSnippet},function(){if(showHeader&&newHeader){$('.rssHeader > a',feedDiv).html(newHeader);}
if(!showSnippet){feedDiv.removeClass('rssFeed').addClass('rssFullText');}});});var dropDownMenus=$('.drop-down-menu');if(dropDownMenus.length>0){dropDownMenus.each(function(i){var thisDropDown=$(this);if(!thisDropDown.attr('id')){thisDropDown.attr('id','drop-down-'+(i+1));}
var headings=thisDropDown.children('h2, h3, h4, h5, h6').not('.dropdown-ignore');headings.each(function(j){var thisHeading=$(this);thisHeading.addClass('drop-down-header');if(!thisHeading.attr('id')){thisHeading.attr('id',"dropdown-heading"+i+"-"+j);}
thisHeading.wrapInner('<a href="#'+thisHeading.attr('id')+'"></a>');var dropDownContentBlock=thisHeading.next('div');dropDownContentBlock.addClass('drop-down-content content-padding');if(anchorLink!==thisHeading.attr('id')){dropDownContentBlock.hide();}
else{thisHeading.addClass('active');if(contentTabs){var parentTabId='#'+thisDropDown.parents('.contentTab').attr('id');var contentTabLink=$('a[href="'+parentTabId+'"]','#contentTabs');if(contentTabLink.length>0){self.switchTabs(contentTabLink);}}}
var depth=thisHeading.parents('.drop-down-menu').length-1;if(depth>0){thisDropDown.addClass('drop-down-child drop-down-level'+depth);if(anchorLink===thisHeading.attr('id')){thisDropDown.parents('.drop-down-content:hidden').show(500).prev('.drop-down-header').addClass('active');}}});var collapseContentBlock=function(dropDownContentBlock){dropDownContentBlock.find('.drop-down-content:visible').slideUp('slow').removeClass('active');dropDownContentBlock.find('.drop-down-content:hidden').hide();dropDownContentBlock.find('.drop-down-header, .active').removeClass('active');}
thisDropDown.click(function(e){var clickedHeading=$(e.target).hasClass('drop-down-header')?$(e.target):$(e.target).parentsUntil('.drop-down-menu','.drop-down-header');if(clickedHeading.length>0){var dropDownContentBlock=clickedHeading.next('div');if(dropDownContentBlock.is(':hidden')){dropDownContentBlock.slideDown('slow');clickedHeading.addClass('active');if(!thisDropDown.hasClass('ignore-siblings')){dropDownContentBlock.siblings('.drop-down-content:visible').slideUp('slow');headings.not(clickedHeading).removeClass('active');if(thisDropDown.hasClass('collapse-children')){collapseContentBlock(dropDownContentBlock.siblings('.drop-down-content').children());}}}
else{dropDownContentBlock.slideUp("slow");clickedHeading.removeClass("active");if(thisDropDown.hasClass('collapse-children')){collapseContentBlock(dropDownContentBlock);}}
e.stopPropagation();e.preventDefault();}});$('.btop','.drop-down-content').parent('.screen').each(function(){$(this).css('display','none');});});dropDownMenus.each(function(i){var thisDropDown=$(this);if(thisDropDown.hasClass('expand-first')&&thisDropDown.find('.drop-down-header.active').length==0){var firstDropDown=thisDropDown.children('h2, h3, h4, h5, h6').first()
firstDropDown.addClass('active');firstDropDown.next('div').show(500);}});if(anchorLink){var offset=$('#'+anchorLink).offset();$(window).scrollTop(offset.top);}}});

/**
 * fastLiveFilter jQuery plugin 1.0.3
 * 
 * Copyright (c) 2011, Anthony Bush
 * License: <http://www.opensource.org/licenses/bsd-license.php>
 * Project Website: http://anthonybush.com/projects/jquery_fast_live_filter/
 **/
jQuery.fn.fastLiveFilter=function(list,options){options=options||{};list=jQuery(list);var input=this;var timeout=options.timeout||0;var callback=options.callback||function(){};var keyTimeout;var lis=list.children();var len=lis.length;var oldDisplay=len>0?lis[0].style.display:"block";callback(len);input.change(function(){var filter=input.val().toLowerCase();var li;var numShown=0;for(var i=0;i<len;i++){li=lis[i];if((li.textContent||li.innerText||"").toLowerCase().indexOf(filter)>=0){if(li.style.display=="none"){li.style.display=oldDisplay;}
numShown++;}else{if(li.style.display!="none"){li.style.display="none";}}}
callback(numShown);return false;}).keydown(function(){clearTimeout(keyTimeout);keyTimeout=setTimeout(function(){input.change();},timeout);});return this;}