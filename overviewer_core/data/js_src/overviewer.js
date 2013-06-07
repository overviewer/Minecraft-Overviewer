/* Overviewer.js
 *
 * Must be the first file included from index.html
 */


var overviewer = {};


/**
 * This holds the map, probably the most important var in this file
 */
overviewer.map = null;
overviewer.mapView = null;


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

        'worldViews': [],

        'haveSigns': false,

        /**
         * Hold the raw marker data for each tilest
         */
        'markerInfo': {},

        /**
         * holds a reference to the spawn marker. 
         */
        'spawnMarker': null,
	
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
            return function(tile, zoom) {
                var url = path;
                var urlBase = ( pathBase ? pathBase : '' );
                if(tile.x < 0 || tile.x >= Math.pow(2, zoom) ||
                   tile.y < 0 || tile.y >= Math.pow(2, zoom)) {
                    url += '/blank';
                } else if(zoom === 0) {
                    url += '/base';
                } else {
                    for(var z = zoom - 1; z >= 0; --z) {
                        var x = Math.floor(tile.x / Math.pow(2, z)) % 2;
                        var y = Math.floor(tile.y / Math.pow(2, z)) % 2;
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
