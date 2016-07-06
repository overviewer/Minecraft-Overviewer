/* Overviewer.js
 *
 * Must be the first file included from index.html
 */


var overviewer = {};


/**
 * This holds the map, probably the most important var in this file
 */
overviewer.map = null;
overviewer.worldCtrl = null;
overviewer.layerCtrl = null;
overviewer.current_world = null;

/// Records the current layer by name (if any) of each world
overviewer.current_layer = {};
    

overviewer.collections = {
        /**
         * MapTypes that aren't overlays will end up in here.
         */
        'mapTypes':     {},
        /**
         * The mapType names are in here.
         */
        'mapTypeIds':   [],
        /**
         * This is the current infoWindow object, we keep track of it so that
         * there is only one open at a time.
         */
        'infoWindow':   null,

        /**
         * When switching regionsets, where should we zoom to?
         * Defaults to spawn.  Stored as map of world names to [latlng, zoom]
         */
        'centers': {},

        'overlays': {},

        'worldViews': [],

        'haveSigns': false,

        /**
         * Hold the raw marker data for each tilest
         */
        'markerInfo': {},

        /**
         * holds a reference to the current spawn marker.
         */
        'spawnMarker': null,
        /**
         * contains the spawn marker for each world
         */
        'spawnMarkers': {},
	
	/**
	 * if a user visits a specific URL, this marker will point to the coordinates in the hash
	 */
        'locationMarker': null
    };

overviewer.classes = {
        /**
         * Our custom projection maps Latitude to Y, and Longitude to X as
         * normal, but it maps the range [0.0, 1.0] to [0, tileSize] in both
         * directions so it is easier to position markers, etc. based on their
         * position (find their position in the lowest-zoom image, and divide
         * by tileSize)
         */
        'MapProjection' : function() {
            this.inverseTileSize = 1.0 / overviewerConfig.CONST.tileSize;
        },
        /**
         * This is a mapType used only for debugging, to draw a grid on the screen
         * showing the tile co-ordinates and tile path. Currently the tile path
         * part does not work.
         * 
         * @param google.maps.Size tileSize
         */
        'CoordMapType': function(tileSize) {
            this.tileSize = tileSize;
        }

};


overviewer.gmap = {

        /**
         * Generate a function to get the path to a tile at a particular location
         * and zoom level.
         * 
         * @param string path
         * @param string pathBase
         * @param string pathExt
         */
        'getTileUrlGenerator': function(path, pathBase, pathExt) {
            return function(o) {
                var url = path;
                var zoom = o.z;
                var urlBase = ( pathBase ? pathBase : '' );
                if(o.x < 0 || o.x >= Math.pow(2, zoom) ||
                   o.y < 0 || o.y >= Math.pow(2, zoom)) {
                    url += '/blank';
                } else if(zoom === 0) {
                    url += '/base';
                } else {
                    for(var z = zoom - 1; z >= 0; --z) {
                        var x = Math.floor(o.x / Math.pow(2, z)) % 2;
                        var y = Math.floor(o.y / Math.pow(2, z)) % 2;
                        url += '/' + (x + 2 * y);
                    }
                }
                url = url + '.' + pathExt;
                if(typeof overviewerConfig.map.cacheTag !== 'undefined') {
                    url += '?c=' + overviewerConfig.map.cacheTag;
                }
                return(urlBase + url);
            };
        }
};
