
  var config = {
    path:        'tiles',
    fileExt:     '{imgformat}',
    tileSize:     384,
    defaultZoom:  1,
    maxZoom:      {maxzoom},
    cacheMinutes: 0, // Change this to have browsers automatically request new images every x minutes
    debug:        false
  };


/* signGroups -- A list of signpost groups.  A signpost can fall into zero, one, or more than one
 * group.  See below for some examples.
 *
 * Required: 
 *     label : string.  Displayed in the drop down menu control.
 *     match : function.  Applied to each marker (from markers.js).  It is returns true if the marker
 *                        Should be part of the group.
 *
 * Optional:
 *     checked : boolean.  Set to true to have the group visible by default
 */
var signGroups = [
//    {label: "'To'", checked: false, match: function(s) {return s.msg.match(/to/)}},
//    {label: "Storage", match: function(s) {return s.msg.match(/storage/i) || s.msg.match(/dirt/i) || s.msg.match(/sand/)}},
//    {label: "Below Sealevel", match: function(s) { return s.y<64;}},   
    {label: "All", match: function(s) {return true}}
];

// Please leave the following variables here:
var markerCollection = {}; // holds groups of markers

var map;

var markersInit = false;
var regionsInit = false;
