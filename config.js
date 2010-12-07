
  var config = {
    path:        'tiles',
    fileExt:     '{imgformat}',
    tileSize:     384,
    defaultZoom:  1,
    maxZoom:      {maxzoom},
    cacheMinutes: 0, // Change this to have browsers automatically request new images every x minutes
    debug:        false
  };


// define a list of pattern-label pairs.  Each label will appear
// in the 'Signposts' control, allowing your users to quickly enable
// or disable certain labels.  See below for some examples:
var signGroups = {
//    "Directions": /^#Direction/i,
//    "Big Dig": /big\s*dig/i,
//    "Warnings": /warning/i,
};

// Please leave the following variables here:
var markerCollection = {}; // holds groups of markers

var map;

var markersInit = false;
var regionsInit = false;
