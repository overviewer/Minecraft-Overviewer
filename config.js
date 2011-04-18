
  var config = {
    tileSize:     384,
    defaultZoom:  2,
    maxZoom:      {maxzoom},
    // center on this point, in world coordinates, ex:
    //center:       [0,0,0],
    center:       {spawn_coords},
    cacheMinutes: 0, // Change this to have browsers automatically request new images every x minutes
    bg_color:     '{bg_color}',  // You can set this in settings.py
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

/* regionGroups -- A list of region groups.  A region can fall into zero, one, or more than one
 * group.  See below for some examples.
 * regions have been designed to work with the 
 * WorldGuard Overviewer Region importer at https://github.com/pironic/WG2OvR But your host must support php in order
 * to run WG2OvR. You can also continue to use any other region format.
 *
 * Required: 
 *     label : string.  Displayed in the drop down menu control.
 *     clickable : boolean. Will determine if we should generate an experimental info window
 *                          that shows details about the clicked region. 
 * NOTE: if a region (as definned in region.js) does not have a label, this will default to false.
 *     match : function.  Applied to each region (from region.js).  It returns true if the region
 *                        Should be part of the group.
 *
 * Optional:
 *     checked : boolean.  Set to true to have the group visible by default
 */
var regionGroups = [
    //{label: "All", clickable: false, checked: false, match: function(s) {return true}},
];

/* mapTypeData -- a list of alternate map renderings available. At least one rendering must be
 * listed.  When more than one are provided, controls to switch between them are provided, with
 * the first one being the default.
 *
 * Required:
 *     label    : string. Displayed on the control.
 *     path     : string. Location of the rendered tiles.
 * Optional:
 *     base     : string. Base of the url path for tile locations, useful for serving tiles from a different server than the js/html server.
 *    imgformat : string. File extension used for these tiles. Defaults to png.
 *    overlay   : bool. If true, this tile set will be treated like an overlay

var mapTypeData=[
  {'label': 'Unlit', 'path': 'tiles'},
//  {'label': 'Day',   'path': 'lighting/tiles'},
//  {'label': 'Night', 'path': 'night/tiles', 'imgformat': 'jpg'},
//  {'label': 'Spawn', 'path': 'spawn/tiles', 'base': 'http://example.cdn.amazon.com/'},
//  {'label': 'Overlay', 'path': 'overlay/tiles', 'overlay': true}
];
 */

var mapTypeData = {maptypedata};

