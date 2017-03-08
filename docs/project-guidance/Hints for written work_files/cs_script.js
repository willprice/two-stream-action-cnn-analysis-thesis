
if (window.addEventListener) { window.addEventListener(
 "load",
 function load () {
  if (document.forms[0].words.value==='') document.forms[0].words.style.backgroundImage='url(/images/google_custom_search_watermark.gif)';

  function l (f) {
   return function () {
    var e=document.getElementById(f) || document.anchors[f];
    var t=e.getAttribute("tabindex");
    if (!t) e.setAttribute("tabindex",-1);
    e.focus();
    if (!t) e.removeAttribute("tabindex");
   };
  }
  var a=document.links;
  for (var i=0; i<a.length; i++) {
   var h=a[i].getAttribute("href");
   if (h && h.charAt(0)==="#" && h.length>1) a[i].addEventListener("click",l(h.slice(1)),false);
  }
  window.removeEventListener("load",load,false);
 },
 false);
}

          top.visible_id = 'csnav'; 

          function toggle_visibility() { 

             var right_e = document.getElementById('uobnav'); 
             var left_e = document.getElementById('csnav'); 
            if (top.visible_id == 'uobnav') {
              right_e.style.display = 'none'; 
              left_e.style.display = 'block'; 
              top.visible_id = 'csnav';
            } else {
              right_e.style.display = 'block'; 
              left_e.style.display = 'none'; 
              top.visible_id = 'uobnav';
            }
            
            return false;
          }   
