
  var config = {
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
 *     icon : string. Used to specify an icon url.
 */
var signGroups = [
//    {label: "'To'", checked: false, match: function(s) {return s.msg.match(/to/)}},
//    {label: "Storage", match: function(s) {return s.msg.match(/storage/i) || s.msg.match(/dirt/i) || s.msg.match(/sand/)}},
//    {label: "Below Sealevel", match: function(s) { return s.y<64;}},   
//    {label: "Info", match: function(s) { return s.msg.match("\\[info\\]");}, icon:"http://google-maps-icons.googlecode.com/files/info.png"},   
    {label: "All", match: function(s) {return true}},
];

/* mapTypeData -- a list of alternate map renderings available. At least one rendering must be
 * listed.  When more than one are provided, controls to switch between them are provided, with
 * the first one being the default.
 *
 * Required:
 *     label : string. Displayed on the control.
 *     path  : string. Location of the rendered tiles.
 */
var mapTypeData=[
  {'label': 'Unlit', 'path': 'tiles'},
//  {'label': 'Day',   'path': 'lighting/tiles'},
//  {'label': 'Night', 'path': 'night/tiles'},
//  {'label': 'Spawn', 'path': 'spawn/tiles'}
];

// Please leave the following variables here:
var markerCollection = {}; // holds groups of markers

var map;

var markersInit = false;
var regionsInit = false;
