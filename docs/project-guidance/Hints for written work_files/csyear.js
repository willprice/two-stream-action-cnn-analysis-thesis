/* Add an onload init function in a way which doesn't affect other scripts */

if (window.addEventListener)
{ 
   window.addEventListener("load", initCS, false);
}
else if (window.attachEvent)
{ 
   window.attachEvent("onload", initCS);
}

/* Read the cookies which the server maintains to manage academic years,
* and use it to display the year and change the background colour.
*/

function initCS()
{
   var year = readCookie("year");
   var age = readCookie("age");
   if (year == null || age == null)
   { 
      /* setText("year2", ""); */
      return;
   }
   document.getElementsByTagName("html")[0].className = age;
   setText("year1", year);
   //if (age == "now") year="";
   //setClass("year1", age+'yeartext');
}

/* Change the class of an element */

function setClass(id, cls)
{
   var element = document.getElementById(id);
   element.className = cls;
}

/* Change the text of an element */

function setText(id, text)
{
   var element = document.getElementById(id);
   element.innerHTML = text;
}

/* Read a cookie */

function readCookie(name)
{
   var nameEQ = name + "=";
   var ca = document.cookie.split(';');
   for (var i=0; i < ca.length; i++)
   {
      var c = ca[i];
      while (c.charAt(0)==' ') c = c.substring (1, c.length);
      if (c.indexOf(nameEQ) == 0) return c.substring (nameEQ.length, c.length);
   }
   return null;
}
