overviewer.util = {
    
    // vars for callback
    readyQueue: [],
    isReady: false,

    lastHash: null,

    /* fuzz tester!
     */
    'testMaths': function(t) {
        var initx = Math.floor(Math.random() * 400) - 200;
        var inity = 64;
        var initz = Math.floor(Math.random() * 400) - 200;
        console.log("Initial point: %r,%r,%r", initx, inity, initz);

        var latlng = overviewer.util.fromWorldToLatLng(initx, inity, initz, t);
        console.log("LatLng: %r,%r", latlng.lat(), latlng.lng());

        var p = overviewer.util.fromLatLngToWorld(latlng.lat(), latlng.lng(), t);
        console.log("Result: %r,%r,%r", p.x, p.y, p.z);
        if (p.x == initx && p.y == inity && p.z == initz) {
            console.log("Pass");
        }


    },

    /**
     * General initialization function, called when the page is loaded.
     * Probably shouldn't need changing unless some very different kind of new
     * feature gets added.
     */
    'initialize': function() {
        overviewer.util.initializeClassPrototypes();

        overviewer.collections.worlds = new overviewer.models.WorldCollection();

        $.each(overviewerConfig.worlds, function(index, el) {
                var n = new overviewer.models.WorldModel({name: el, id:el});
                overviewer.collections.worlds.add(n);
                });

        $.each(overviewerConfig.tilesets, function(index, el) {
                var newTset = new overviewer.models.TileSetModel(el);
                overviewer.collections.worlds.get(el.world).get("tileSets").add(newTset);
                });

        overviewer.collections.worlds.each(function(world, index, list) {
                var nv = new overviewer.views.WorldView({model: world});
                overviewer.collections.worldViews.push(nv);
                });

        overviewer.mapModel = new overviewer.models.GoogleMapModel({});
        overviewer.mapView = new overviewer.views.GoogleMapView({el: document.getElementById(overviewerConfig.CONST.mapDivId), model:overviewer.mapModel});

        // any controls must be created after the GoogleMapView is created
        // controls should be added in the order they should appear on screen, 
        // with controls on the outside of the page being added first

        var compass = new overviewer.views.CompassView({tagName: 'DIV', model:overviewer.mapModel});
        // no need to render the compass now.  it's render event will get fired by
        // the maptypeid_chagned event

        var coordsdiv = new overviewer.views.CoordboxView({tagName: 'DIV'});
        coordsdiv.render();

        var progressdiv = new overviewer.views.ProgressView({tagName: 'DIV'});
        progressdiv.render();
        progressdiv.updateProgress();

        if (overviewer.collections.haveSigns) {
            var signs = new overviewer.views.SignControlView();
            signs.registerEvents(signs);
        }

        var overlayControl = new overviewer.views.OverlayControlView();

        var spawnmarker = new overviewer.views.SpawnIconView();

        // Update coords on mousemove
        google.maps.event.addListener(overviewer.map, 'mousemove', function (event) {
            coordsdiv.updateCoords(event.latLng);    
        });
        google.maps.event.addListener(overviewer.map, 'idle', function (event) {
            overviewer.util.updateHash();
        });

        google.maps.event.addListener(overviewer.map, 'maptypeid_changed', function(event) {
            // it's handy to keep track of the currently visible tileset.  we let
            // the GoogleMapView manage this
            overviewer.mapView.updateCurrentTileset();

            compass.render();
            spawnmarker.render();
            if (overviewer.collections.locationMarker) {
                overviewer.collections.locationMarker.setMap(null);
                overviewer.collections.locationMarker = null;
            }

            // update list of spawn overlays
            overlayControl.render();

            // re-center on the last viewport
            var currentWorldView = overviewer.mapModel.get("currentWorldView");
            if (currentWorldView.options.lastViewport) {
                var x = currentWorldView.options.lastViewport[0];
                var y = currentWorldView.options.lastViewport[1];
                var z = currentWorldView.options.lastViewport[2];
                var zoom = currentWorldView.options.lastViewport[3];

                var latlngcoords = overviewer.util.fromWorldToLatLng(x, y, z,
                    overviewer.mapView.options.currentTileSet);
                overviewer.map.setCenter(latlngcoords);

                if (zoom == 'max') {
                    zoom = overviewer.mapView.options.currentTileSet.get('maxZoom');
                } else if (zoom == 'min') {
                    zoom = overviewer.mapView.options.currentTileSet.get('minZoom');
                } else {
                    zoom = parseInt(zoom);
                    if (zoom < 0) {
                        // if zoom is negative, treat it as a "zoom out from max"
                        zoom += overviewer.mapView.options.currentTileSet.get('maxZoom');
                    } else {
                        // fall back to default zoom
                        zoom = overviewer.mapView.options.currentTileSet.get('defaultZoom');
                    }
                }
                
                // clip zoom
                if (zoom > overviewer.mapView.options.currentTileSet.get('maxZoom'))
                    zoom = overviewer.mapView.options.currentTileSet.get('maxZoom');
                if (zoom < overviewer.mapView.options.currentTileSet.get('minZoom'))
                    zoom = overviewer.mapView.options.currentTileSet.get('minZoom');
                
                overviewer.map.setZoom(zoom);
            }


        });


        // hook up some events

        overviewer.mapModel.bind("change:currentWorldView", overviewer.mapView.render, overviewer.mapView);

        overviewer.mapView.render();
         
        // Jump to the hash if given (and do so for any further hash changes)
        overviewer.util.initHash();
        $(window).on('hashchange', function() { overviewer.util.initHash(); });

        // create this control after initHash so it can correctly select the current world
        var worldSelector = new overviewer.views.WorldSelectorView({tagName:'DIV'});
        overviewer.collections.worlds.bind("add", worldSelector.render, worldSelector);

        
        overviewer.util.initializeMarkers();

        /*
           overviewer.util.initializeMapTypes();
           overviewer.util.initializeMap();
           overviewer.util.initializeRegions();
           overviewer.util.createMapControls();
           */
           
        // run ready callbacks now
        google.maps.event.addListenerOnce(overviewer.map, 'idle', function(){
            // ok now..
            overviewer.util.runReadyQueue();
            overviewer.util.isReady = true;
        });
    },

    'injectMarkerScript': function(url) {
        var m = document.createElement('script'); m.type = 'text/javascript'; m.async = false;
        m.src = url;
        var s = document.getElementsByTagName('script')[0]; s.parentNode.appendChild(m);
    },

    'initializeMarkers': function() {
        return;

    },


    /**
     * This adds some methods to these classes because Javascript is stupid
     * and this seems like the best way to avoid re-creating the same methods
     * on each object at object creation time.
     */
    'initializeClassPrototypes': function() {
        overviewer.classes.MapProjection.prototype.fromLatLngToPoint = function(latLng) {
            var x = latLng.lng() * overviewerConfig.CONST.tileSize;
            var y = latLng.lat() * overviewerConfig.CONST.tileSize;
            return new google.maps.Point(x, y);
        };

        overviewer.classes.MapProjection.prototype.fromPointToLatLng = function(point) {
            var lng = point.x * this.inverseTileSize;
            var lat = point.y * this.inverseTileSize;
            return new google.maps.LatLng(lat, lng);
        };

        overviewer.classes.CoordMapType.prototype.getTile = function(coord, zoom, ownerDocument) {
            var div = ownerDocument.createElement('DIV');
            div.innerHTML = '(' + coord.x + ', ' + coord.y + ', ' + zoom +
                ')' + '<br />';
            //TODO: figure out how to get the current mapType, I think this
            //will add the maptile url to the grid thing once it works

            //div.innerHTML += overviewer.collections.mapTypes[0].getTileUrl(coord, zoom);

            //this should probably just have a css class
            div.style.width = this.tileSize.width + 'px';
            div.style.height = this.tileSize.height + 'px';
            div.style.fontSize = '10px';
            div.style.borderStyle = 'solid';
            div.style.borderWidth = '1px';
            div.style.borderColor = '#AAAAAA';
            return div;
        };
    },
    /**
     * onready function for other scripts that rely on overviewer
     * usage: overviewer.util.ready(function(){ // do stuff });
     *
     *
     */
    'ready': function(callback){
        if (!callback || !_.isFunction(callback)) return;
        if (overviewer.util.isReady){ // run instantly if overviewer already is ready
            overviewer.util.readyQueue.push(callback);
            overviewer.util.runReadyQueue();
        } else {
            overviewer.util.readyQueue.push(callback); // wait until initialize is finished
        }
    },       
    'runReadyQueue': function(){
        _.each(overviewer.util.readyQueue, function(callback){
            callback();
        });
        overviewer.util.readyQueue.length = 0;
    },
    /**
     * Quote an arbitrary string for use in a regex matcher.
     * WTB parametized regexes, JavaScript...
     *
     *   From http://kevin.vanzonneveld.net
     *   original by: booeyOH
     *   improved by: Ates Goral (http://magnetiq.com)
     *   improved by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
     *   bugfixed by: Onno Marsman
     *     example 1: preg_quote("$40");
     *     returns 1: '\$40'
     *     example 2: preg_quote("*RRRING* Hello?");
     *     returns 2: '\*RRRING\* Hello\?'
     *     example 3: preg_quote("\\.+*?[^]$(){}=!<>|:");
     *     returns 3: '\\\.\+\*\?\[\^\]\$\(\)\{\}\=\!\<\>\|\:'
     */
    "pregQuote": function(str) {
        return (str+'').replace(/([\\\.\+\*\?\[\^\]\$\(\)\{\}\=\!\<\>\|\:])/g, "\\$1");
    },
    /**
     * Change the map's div's background color according to the mapType's bg_color setting
     *
     * @param string mapTypeId
     * @return string
     */
    'getMapTypeBackgroundColor': function(id) {
        return overviewerConfig.tilesets[id].bgcolor;
    },
    /**
     * Gee, I wonder what this does.
     * 
     * @param string msg
     */
    'debug': function(msg) {
        if (overviewerConfig.map.debug) {
            console.log(msg);
        }
    },
    /**
     * Simple helper function to split the query string into key/value
     * pairs. Doesn't do any type conversion but both are lowercase'd.
     * 
     * @return Object
     */
    'parseQueryString': function() {
        var results = {};
        var queryString = location.search.substring(1);
        var pairs = queryString.split('&');
        for (i in pairs) {
            var pos = pairs[i].indexOf('=');
            var key = pairs[i].substring(0,pos).toLowerCase();
            var value = pairs[i].substring(pos+1).toLowerCase();
            overviewer.util.debug( 'Found GET paramter: ' + key + ' = ' + value);
            results[key] = value;
        }
        return results;
    },
    'getDefaultMapTypeId': function() {
        return overviewer.collections.mapTypeIds[0];
    },
    /**
     * helper to get map LatLng from world coordinates takes arguments in
     * X, Y, Z order (arguments are *out of order*, because within the
     * function we use the axes like the rest of Minecraft Overviewer --
     * with the Z and Y flipped from normal minecraft usage.)
     * 
     * @param int x
     * @param int z
     * @param int y
     * @param TileSetModel model
     * 
     * @return google.maps.LatLng
     */
    'fromWorldToLatLng': function(x, y, z, model) {

        var zoomLevels = model.get("zoomLevels");
        var north_direction = model.get('north_direction');

        // the width and height of all the highest-zoom tiles combined,
        // inverted
        var perPixel = 1.0 / (overviewerConfig.CONST.tileSize *
                Math.pow(2, zoomLevels));

        if (north_direction == overviewerConfig.CONST.UPPERRIGHT){
            temp = x;
            x = -z+15;
            z = temp;
        } else if(north_direction == overviewerConfig.CONST.LOWERRIGHT){
            x = -x+15;
            z = -z+15;
        } else if(north_direction == overviewerConfig.CONST.LOWERLEFT){
            temp = x;
            x = z;
            z = -temp+15;
        }

        // This information about where the center column is may change with
        // a different drawing implementation -- check it again after any
        // drawing overhauls!

        // point (0, 0, 127) is at (0.5, 0.0) of tile (tiles/2 - 1, tiles/2)
        // so the Y coordinate is at 0.5, and the X is at 0.5 -
        // ((tileSize / 2) / (tileSize * 2^zoomLevels))
        // or equivalently, 0.5 - (1 / 2^(zoomLevels + 1))
        var lng = 0.5 - (1.0 / Math.pow(2, zoomLevels + 1));
        var lat = 0.5;

        // the following metrics mimic those in
        // chunk_render in src/iterate.c

        // each block on X axis adds 12px to x and subtracts 6px from y
        lng += 12 * x * perPixel;
        lat -= 6 * x * perPixel;

        // each block on Y axis adds 12px to x and adds 6px to y
        lng += 12 * z * perPixel;
        lat += 6 * z * perPixel;

        // each block down along Z adds 12px to y
        lat += 12 * (256 - y) * perPixel;

        // add on 12 px to the X coordinate to center our point
        lng += 12 * perPixel;

        return new google.maps.LatLng(lat, lng);
    },
    /**
     * The opposite of fromWorldToLatLng
     * NOTE: X, Y and Z in this function are Minecraft world definitions
     * (that is, X is horizontal, Y is altitude and Z is vertical).
     * 
     * @param float lat
     * @param float lng
     * 
     * @return Array
     */
    'fromLatLngToWorld': function(lat, lng, model) {
        var zoomLevels = model.get("zoomLevels");
        var north_direction = model.get("north_direction");

        // Initialize world x/y/z object to be returned
        var point = Array();
        point.x = 0;
        point.y = 64;
        point.z = 0;

        // the width and height of all the highest-zoom tiles combined,
        // inverted
        var perPixel = 1.0 / (overviewerConfig.CONST.tileSize *
                Math.pow(2, zoomLevels));

        // Revert base positioning
        // See equivalent code in fromWorldToLatLng()
        lng -= 0.5 - (1.0 / Math.pow(2, zoomLevels + 1));
        lat -= 0.5;

        // I'll admit, I plugged this into Wolfram Alpha:
        //   a = (x * 12 * r) + (z * 12 * r), b = (z * 6 * r) - (x * 6 * r)
        // And I don't know the math behind solving for for X and Z given
        // A (lng) and B (lat).  But Wolfram Alpha did. :)  I'd welcome
        // suggestions for splitting this up into long form and documenting
        // it. -RF
        point.x = Math.floor((lng - 2 * lat) / (24 * perPixel));
        point.z = Math.floor((lng + 2 * lat) / (24 * perPixel));

        // Adjust for the fact that we we can't figure out what Y is given
        // only latitude and longitude, so assume Y=64. Since this is lowering
        // down from the height of a chunk, it depends on the chunk height as
        // so:
        point.x += 256-64;
        point.z -= 256-64;

        if(north_direction == overviewerConfig.CONST.UPPERRIGHT){
            temp = point.z;
            point.z = -point.x+15;
            point.x = temp;
        } else if(north_direction == overviewerConfig.CONST.LOWERRIGHT){
            point.x = -point.x+15;
            point.z = -point.z+15;
        } else if(north_direction == overviewerConfig.CONST.LOWERLEFT){
            temp = point.z;
            point.z = point.x;
            point.x = -temp+15;
        }

        return point;
    },
    /**
     * Create the pop-up infobox for when you click on a region, this can't
     * be done in-line because of stupid Javascript scoping problems with
     * closures or something.
     * 
     * @param google.maps.Polygon|google.maps.Polyline shape
     */
    'createRegionInfoWindow': function(shape) {
        var infowindow = new google.maps.InfoWindow();
        google.maps.event.addListener(shape, 'click', function(event, i) {
                if (overviewer.collections.infoWindow) {
                overviewer.collections.infoWindow.close();
                }
                // Replace our Info Window's content and position
                var point = overviewer.util.fromLatLngToWorld(event.latLng.lat(),event.latLng.lng());
                var contentString = '<b>Region: ' + shape.name + '</b><br />' +
                'Clicked Location: <br />' + Math.round(point.x,1) + ', ' + point.y
                + ', ' + Math.round(point.z,1)
                + '<br />';
                infowindow.setContent(contentString);
                infowindow.setPosition(event.latLng);
                infowindow.open(overviewer.map);
                overviewer.collections.infoWindow = infowindow;
                });
    },
    /**
     * Same as createRegionInfoWindow()
     * 
     * @param google.maps.Marker marker
     */
    'createMarkerInfoWindow': function(marker) {
        var windowContent = '<div class="infoWindow"><p><img src="' + marker.icon +
            '"/><br />' + marker.content.replace(/\n/g,'<br/>') + '</p></div>';
        var infowindow = new google.maps.InfoWindow({
            'content': windowContent
        });
        google.maps.event.addListener(marker, 'click', function() {
            if (overviewer.collections.infoWindow) {
                overviewer.collections.infoWindow.close();
            }
            infowindow.open(overviewer.map, marker);
            overviewer.collections.infoWindow = infowindow;
        });
    },
    'initHash': function() {
        var newHash = window.location.hash;
        if (overviewer.util.lastHash !== newHash) {
            overviewer.util.lastHash = newHash;
            if(newHash.split("/").length > 1) {
                overviewer.util.goToHash();
                // Clean up the hash.
                overviewer.util.updateHash();
            }
        }
    },
    'setHash': function(x, y, z, zoom, w, maptype)    {
        // save this info is a nice easy to parse format
        var currentWorldView = overviewer.mapModel.get("currentWorldView");
        currentWorldView.options.lastViewport = [x,y,z,zoom];
        var newHash = "#/" + Math.floor(x) + "/" + Math.floor(y) + "/" + Math.floor(z) + "/" + zoom + "/" + w + "/" + maptype;
        overviewer.util.lastHash = newHash; // this should not trigger initHash
        window.location.replace(newHash);
    },
    'updateHash': function() {
        var currTileset = overviewer.mapView.options.currentTileSet;
        if (currTileset == null) {return;}
        var coordinates = overviewer.util.fromLatLngToWorld(overviewer.map.getCenter().lat(), 
                overviewer.map.getCenter().lng(),
                currTileset);
        var zoom = overviewer.map.getZoom();
        var maptype = overviewer.map.getMapTypeId();

        // convert mapType into a index
        var currentWorldView = overviewer.mapModel.get("currentWorldView");
        var maptypeId = -1;
        for (id in currentWorldView.options.mapTypeIds) {
            if (currentWorldView.options.mapTypeIds[id] == maptype) {
                maptypeId = id;
            }
        }

        var worldId = -1;
        for (id in overviewer.collections.worldViews) {
            if (overviewer.collections.worldViews[id] == currentWorldView) {
                worldId = id;
            }
        }


        if (zoom >= currTileset.get('maxZoom')) {
            zoom = 'max';
        } else if (zoom <= currTileset.get('minZoom')) {
            zoom = 'min';
        } else {
            // default to (map-update friendly) negative zooms
            zoom -= currTileset.get('maxZoom');
        }
        overviewer.util.setHash(coordinates.x, coordinates.y, coordinates.z, zoom, worldId, maptypeId);
    },
    'goToHash': function() {
        // Note: the actual data begins at coords[1], coords[0] is empty.
        var coords = window.location.hash.split("/");


        var zoom;
        var worldid = -1;
        var maptyped = -1;
        // The if-statements try to prevent unexpected behaviour when using incomplete hashes, e.g. older links
        if (coords.length > 4) {
            zoom = coords[4];
        }
        if (coords.length > 6) {
            worldid = coords[5];
            maptypeid = coords[6];
        }
        var worldView = overviewer.collections.worldViews[worldid];
        overviewer.mapModel.set({currentWorldView: worldView});

        var maptype = worldView.options.mapTypeIds[maptypeid];
        overviewer.map.setMapTypeId(maptype);
        var tsetModel = worldView.model.get("tileSets").at(maptypeid);
        
        var latlngcoords = overviewer.util.fromWorldToLatLng(parseInt(coords[1]), 
                parseInt(coords[2]), 
                parseInt(coords[3]),
                tsetModel);

        if (zoom == 'max') {
            zoom = tsetModel.get('maxZoom');
        } else if (zoom == 'min') {
            zoom = tsetModel.get('minZoom');
        } else {
            zoom = parseInt(zoom);
            if (zoom < 0) {
                // if zoom is negative, treat it as a "zoom out from max"
                zoom += tsetModel.get('maxZoom');
            } else {
                // fall back to default zoom
                zoom = tsetModel.get('defaultZoom');
            }
        }

        // clip zoom
        if (zoom > tsetModel.get('maxZoom'))
            zoom = tsetModel.get('maxZoom');
        if (zoom < tsetModel.get('minZoom'))
            zoom = tsetModel.get('minZoom');

        overviewer.map.setCenter(latlngcoords);
        overviewer.map.setZoom(zoom);
        var locationmarker = new overviewer.views.LocationIconView();
        locationmarker.render();
    }
};
