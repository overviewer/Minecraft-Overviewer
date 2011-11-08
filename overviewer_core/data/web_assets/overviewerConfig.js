var overviewerConfig = {
    /**
     * These are things that will probably not need to be changed, but are there
     * because otherwise changing them is a giant PITA. If you, the user, sees
     * that its crucial for you to change these settings then the document
     * repository might be able to assist you.
     * http://docs.overviewer.org/en/latest/options/#customizing-web-assets
     */
    'CONST': {
        /**
         * Height and width of the tiles in pixels.
         */
        'tileSize':             384,
        /**
         * Various images used for markers and stuff.
         */
        'image': {
            'defaultMarker':    'signpost.png',
            'signMarker':       'signpost_icon.png',
            'compass':          'compass_{north_direction}.png',
            'spawnMarker':      'http://google-maps-icons.googlecode.com/files/home.png',
            'queryMarker':      'http://google-maps-icons.googlecode.com/files/regroup.png',
			'cowMarker':		'cow.png',
			'sheepMarker':		'sheep.png',
			'pigMarker':		'pig.png',
			'chickenMarker':	'chicken.png',
			'squidMarker':		'squid.png',
			'arkMarker':		'ark.png'
        },
        'mapDivId':             'mcmap',
        'regionStrokeWeight':   2
    },
    /**
     * General map settings.
     */
    'map': {
        /**
         * Control the visibility of various controls.
         */
        'controls': {
            /**
             * Pan control is the hand with the arrows around it in the upper left.
             */
            'pan':      true,
            /**
             * Zoom control is the zoom slider bar in the upper left.
             */
            'zoom':     true,
            /**
             * Spawn control is the "Spawn" button that centers the map on spawn.
             */
            'spawn':    true,
            /**
             * The compass in the upper right.
             */
            'compass':  true,
            /**
             * The mapType control is the slider for selecting different map types.
             */
            'mapType':  true,
            /**
             * The coordsBox control is the box showing the XYZ coordinates
             * under the cursor.
             */
            'coordsBox': true,
            /**
             * The overlays control is the drop-down box for selecting overlays.
             */
            'overlays': true,
            /**
             * The searchBox control is the search box for markers.
             */
            'searchBox': true
        },
        /**
         * The zoom level when the page is loaded without a specific zoom setting
         */
        'defaultZoom':  0,
        /**
         * This controls how far you can zoom out.
         */
        'minZoom':      {minzoom},
        /**
         * This controls how close you can zoom in.
         */
        'maxZoom':      {maxzoom},
        /**
         * This tells us how many total zoom levels Overviewer rendered.
         * DO NOT change this, even if you change minZoom and maxZoom, because
         * it's used for marker position calculations and map resizing.
         */
        'zoomLevels':   {zoomlevels},
        /**
         * Center on this point, in world coordinates. Should be an array, ex:
         * [0,0,0]
         */
        'center':       {spawn_coords},
        /**
         * Set this to tell browsers how long they should cache tiles in minutes.
         * Essentially if set to 0, the url for tiles will end in .png
         * if not set to 0 it will amend a number derived from the current time 
         * to the end of the url, like .png?c=123456. This is a great method for 
         * preventing browsers from caching the images. 0 saves bandwidth, not 0
         * prevents caching.
         */
        'cacheMinutes': 0,
        /**
         * Set to true to turn on debug mode, which adds a grid to the map along
         * with co-ordinates and a bunch of console output.
         */
        'debug':        false,
        /**
         * Set which way north points.
         */
        'north_direction': '{north_direction}'
    },
    /**
     * Group definitions for objects that are partially selectable (signs and
     * regions).
     */
    'objectGroups': {
        /* signs -- A list of signpost groups.  A signpost can fall into zero,
         * one, or more than one group.  See below for some examples.
         *
         * Required: 
         *     label : string.  Displayed in the drop down menu control.
         *     match : function.  Applied to each marker (from markers.js). It
         *                        is returns true if the marker should be part
         *                        of the group.
         *
         * Optional:
         *     checked : boolean.  Set to true to have the group visible by default
         *     icon : string. Used to specify an icon url.
         */
        'signs': [
            //{label: "'To'", checked: false, match: function(s) {return s.msg.match(/to/)}},
            //{label: "Storage", match: function(s) {return s.msg.match(/storage/i) || s.msg.match(/dirt/i) || s.msg.match(/sand/)}},
            //{label: "Below Sealevel", match: function(s) { return s.y<64;}},   
            //{label: "Info", match: function(s) { return s.msg.match("\\[info\\]");}, icon:"http://google-maps-icons.googlecode.com/files/info.png"},   
            {'label':'All', 'match':function(sign){return true;}}
        ],
		/* animals -- A list of the passive animals.
 		 */
		'animals': [
			//
			{'label':'ark', 'icon': 'ark.png', 'match':function(animal){
				return (animal.type == "cow" ||
				   animal.type == "pig" ||
				   animal.type == "sheep" ||
				   animal.type == "chicken" ||
				   animal.type == "squid");
			}},
			{'label':'cow', 'icon': 'cow.png', 'match':function(animal){ return (animal.type == "cow") }},
			{'label':'pig', 'icon': 'pig.png', 'match':function(animal){ return (animal.type == "pig") }},
			{'label':'sheep', 'icon': 'sheep.png', 'match':function(animal){ return (animal.type == "sheep") }},
			{'label':'chicken', 'icon': 'chicken.png', 'match':function(animal){ return (animal.type == "chicken") }},
			{'label':'squid', 'icon': 'squid.png', 'match':function(animal){ return (animal.type == "squid") }}
		],

        /* regions -- A list of region groups.  A region can fall into zero,
         * one, or more than one group.  See below for some examples.
         * Regions have been designed to work with the WorldGuard Overviewer
         * Region importer at @link http://goo.gl/dc0tV but your
         * host must support php in order to run WG2OvR. You can also continue
         * to use any other region format.
         *
         * Required: 
         *     label : string.  Displayed in the drop down menu control.
         *     clickable : boolean. Will determine if we should generate an
         *                          experimental info window that shows details
         *                          about the clicked region. 
         *                          NOTE: if a region (as defined in region.js)
         *                          does not have a label, this will default to
         *                          false.
         *     match : function.  Applied to each region (from region.js). It
         *                        returns true if the region should be part of
         *                        the group.
         *
         * Optional:
         *     checked : boolean.  Set to true to have the group visible by default
         */
        'regions': [
            //{'label':'All','clickable':true,'match':function(region){return true;}}
        ]
    },
    /* mapTypes -- a list of alternate map renderings available. At least one
     * rendering must be listed.  When more than one are provided, controls to
     * switch between them are provided, with the first one being the default.
     *
     * Required:
     *     label    : string. Displayed on the control.
     *     path     : string. Location of the rendered tiles.
     * Optional:
     *     base     : string. Base of the url path for tile locations, useful
     *                        for serving tiles from a different server than
     *                        the js/html server.
     *    imgformat : string. File extension used for these tiles. Defaults to png.
     *    overlay   : bool. If true, this tile set will be treated like an overlay
     *    bg_color  : string. A #RRGGBB format background color.
     *    shortname : string. Used in the dynamic anchor link mechanism. If not
     *                        present, label is used instead.
     *
     * Example:
     *  'mapTypes': [
     *      {'label': 'Day',   'path': 'lighting/tiles'},
     *      {'label': 'Night', 'path': 'night/tiles', 'imgformat': 'jpg'},
     *      {'label': 'Spawn', 'path': 'spawn/tiles', 'base': 'http://example.cdn.amazon.com/'},
     *      {'label': 'Overlay', 'path': 'overlay/tiles', 'overlay': true}
     *  ]
     */
    'mapTypes':         {maptypedata}
};
