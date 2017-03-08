$(document).ready(function() {
                   
					var screenContainer = $('.uob-header-container:first');
					var screen = $('<div class="screen-overlay hidden"></div>').appendTo(screenContainer);
					
					var pageHeight = $(document).height();
					screen.css('height', pageHeight - screenContainer.offset().top);

					

                  screen.click(function() {
					  var displayedDropDown = $('.uob-mainnav > li div.display');
                      $('a[tabindex=0]', displayedDropDown).attr("tabindex", "-1");
					  displayedDropDown.removeClass('display');
					  screen.addClass('hidden');
                      $('.u-m-link-active').removeClass('u-m-link-active');
				  });



                  $('#uob-mainnav-study, #uob-mainnav-year, #uob-mainnav-university').click(function(event) {
                        event.preventDefault();

                        var jqSelf = $(this);

                        var dropDown = jqSelf.next('.dropdown');

                        var displayedDropDown = $('.uob-mainnav > li div.display');

                        displayedDropDown.removeClass('display');
                        $('a[tabindex=0]', displayedDropDown).attr("tabindex", "-1");
                        $('.u-m-link-active').removeClass('u-m-link-active');

                        jqSelf.addClass('u-m-link-active');

                        if(!dropDown.is(displayedDropDown)) {
                            dropDown.addClass('display');
                            $('a[tabindex=-1]', dropDown).attr("tabindex", "0");
                            screen.removeClass('hidden');
                           // $('a:first', dropDown).focus();
                        }

                        else {
                            dropDown.removeClass('display');
                          //  jqSelf.blur(); // has unfortunate effect in all browsers apart from Firefox
                            screen.addClass('hidden');
                            $('.u-m-link-active').removeClass('u-m-link-active');

                        }
                  });
});