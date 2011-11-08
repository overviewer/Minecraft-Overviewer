var overviewer = {
    /**
     * This holds the map, probably the most important var in this file
     */
    'map': null,
    /**
     * These are collections of data used in various places
     */
    'collections': {
        /**
         * A list of lists of raw marker data objects, this will allow for an
         * arbitrary number of marker data sources. This replaces the old
         * markerData var from markers.js. Now you can add markers by including
         * a file with:
         * overviewer.collections.markerDatas.push([<your list of markers>]);
         */
        'markerDatas':  [],
        /**
         * The actual Marker objects are stored here.
         */
        'markers':      {},
        /**
         * Same as markerDatas, list of lists of raw region objects.
         */
        'regionDatas':  [],
        /**
         * The actual Region objects.
         */
        'regions':      {},
        /**
         * Overlay mapTypes (like Spawn) will go in here.
         */
        'overlays':     [],
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
        'infoWindow':   null
    },
    'util': {
        /**
         * General initialization function, called when the page is loaded.
         * Probably shouldn't need changing unless some very different kind of new
         * feature gets added.
         */
        'initialize': function() {
            overviewer.util.initializeClassPrototypes();
            overviewer.util.initializeMapTypes();
            overviewer.util.initializeMap();
            overviewer.util.initializeMarkers();
            overviewer.util.initializeRegions();
            overviewer.util.createMapControls();
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
         * Setup the varous mapTypes before we actually create the map. This used
         * to be a bunch of crap down at the bottom of functions.js
         */
        'initializeMapTypes': function() {
            var mapOptions = {};
            for (i in overviewerConfig.mapTypes) {
                var view = overviewerConfig.mapTypes[i];
                var imageFormat = view.imgformat ? view.imgformat : 'png';
                
                if (view.shortname == null)
                    view.shortname = view.label.replace(/\s+/g, "");
                
                mapOptions[view.shortname] = {
                    'getTileUrl':   overviewer.gmap.getTileUrlGenerator(view.path,
                        view.base, imageFormat),
                    'tileSize':     new google.maps.Size(
                        overviewerConfig.CONST.tileSize,
                        overviewerConfig.CONST.tileSize),
                    'maxZoom':      overviewerConfig.map.maxZoom,
                    'minZoom':      overviewerConfig.map.minZoom,
                    'isPng':        imageFormat.toLowerCase() == 'png'
                }
                overviewer.collections.mapTypes[view.shortname] = new google.maps.ImageMapType(
                    mapOptions[view.shortname]);
                overviewer.collections.mapTypes[view.shortname].name = view.label;
                overviewer.collections.mapTypes[view.shortname].shortname = view.shortname;
                overviewer.collections.mapTypes[view.shortname].alt = 'Minecraft ' +
                    view.label + ' Map';
                overviewer.collections.mapTypes[view.shortname].projection  =
                    new overviewer.classes.MapProjection();
                if (view.overlay) {
                    overviewer.collections.overlays.push(
                        overviewer.collections.mapTypes[view.shortname]);
                } else {
                    overviewer.collections.mapTypeIds.push(
                        overviewerConfig.CONST.mapDivId + view.shortname);
                }
            }
        },
        /**
         * This is where the magic happens. We setup the map with all it's
         * options. The query string is also parsed here so we can know if
         * we should be looking at a particular point on the map or just use
         * the default view.
         */
        'initializeMap': function() {
            var defaultCenter = overviewer.util.fromWorldToLatLng(
                overviewerConfig.map.center[0], overviewerConfig.map.center[1],
                overviewerConfig.map.center[2]);
            var lat = defaultCenter.lat();
            var lng = defaultCenter.lng();
            var zoom = overviewerConfig.map.defaultZoom;
            var mapcenter;
            var queryParams = overviewer.util.parseQueryString();
            if (queryParams.debug) {
                overviewerConfig.map.debug=true;
            }
            if (queryParams.lat) {
                lat = parseFloat(queryParams.lat);
            }
            if (queryParams.lng) {
                lng = parseFloat(queryParams.lng);
            }
            if (queryParams.zoom) {
                if (queryParams.zoom == 'max') {
                    zoom = overviewerConfig.map.maxZoom;
                } else if (queryParams.zoom == 'min') {
                    zoom = overviewerConfig.map.minZoom;
                } else {
                    zoom = parseInt(queryParams.zoom);
                    if (zoom < 0 && zoom + overviewerConfig.map.maxZoom >= 0) {
                        //if zoom is negative, try to treat as "zoom out from max zoom"
                        zoom += overviewerConfig.map.maxZoom;
                    } else {
                        //fall back to default zoom
                        zoom = overviewerConfig.map.defaultZoom;
                    }
                }
            }
            if (queryParams.x && queryParams.y && queryParams.z) {
                mapcenter = overviewer.util.fromWorldToLatLng(queryParams.x,
                    queryParams.y, queryParams.z);
                // Add a market indicating the user-supplied position
                overviewer.collections.markerDatas.push([{
                    'msg':  'Coordinates ' + queryParams.x + ', ' +
                        queryParams.y + ', ' + queryParams.z,
                    'y':    parseFloat(queryParams.y),
                    'x':    parseFloat(queryParams.x),
                    'z':    parseFloat(queryParams.z),
                    'type': 'querypos'}]);
            } else {
                mapcenter = new google.maps.LatLng(lat, lng);
            }
            var mapOptions = {
                zoom:                   zoom,
                center:                 mapcenter,
                panControl:             overviewerConfig.map.controls.pan,
                scaleControl:           false,
                mapTypeControl:         overviewerConfig.map.controls.mapType &&
                    overviewer.collections.mapTypeIds.length > 1,
                mapTypeControlOptions: {
                    mapTypeIds: overviewer.collections.mapTypeIds
                },
                mapTypeId:              overviewer.util.getDefaultMapTypeId(),
                streetViewControl:      false,
                overviewMapControl:     true,
                zoomControl:            overviewerConfig.map.controls.zoom,
                backgroundColor:        overviewer.util.getMapTypeBackgroundColor(
                    overviewer.util.getDefaultMapTypeId())
            };
            overviewer.map = new google.maps.Map(document.getElementById(
                overviewerConfig.CONST.mapDivId), mapOptions);

            if (overviewerConfig.map.debug) {
                overviewer.map.overlayMapTypes.insertAt(0,
                    new overviewer.classes.CoordMapType(new google.maps.Size(
                        overviewerConfig.CONST.tileSize,
                        overviewerConfig.CONST.tileSize)));
                google.maps.event.addListener(overviewer.map, 'click', function(event) {
                    overviewer.util.debug('latLng: (' + event.latLng.lat() +
                        ', ' + event.latLng.lng() + ')');
                    var pnt = overviewer.map.getProjection().fromLatLngToPoint(event.latLng);
                    overviewer.util.debug('point: ' + pnt);
                    var pxx = pnt.x * overviewerConfig.CONST.tileSize *
                        Math.pow(2, overviewerConfig.map.maxZoom);
                    var pxy = pnt.y * overviewerConfig.CONST.tileSize *
                        Math.pow(2, overviewerConfig.map.maxZoom);
                    overviewer.util.debug('pixel: (' + pxx + ', ' + pxy + ')');
                });
            }

            // Now attach the coordinate map type to the map's registry
            for (i in overviewer.collections.mapTypes) {
                overviewer.map.mapTypes.set(overviewerConfig.CONST.mapDivId +
                    overviewer.collections.mapTypes[i].shortname,
                    overviewer.collections.mapTypes[i]);
            }
            
            // Jump to the hash if given
            overviewer.util.initHash();
            
            // Add live hash update listeners
            // Note: It is important to add them after jumping to the hash
            google.maps.event.addListener(overviewer.map, 'dragend', function() {
                overviewer.util.updateHash();
            });
            
            google.maps.event.addListener(overviewer.map, 'zoom_changed', function() {
                overviewer.util.updateHash();
            });
            google.maps.event.addListener(overviewer.map, 'dblclick', function() {
                overviewer.util.updateHash();
            });
            
            // Make the link again whenever the map changes
            google.maps.event.addListener(overviewer.map, 'maptypeid_changed', function() {
                $('#'+overviewerConfig.CONST.mapDivId).css(
                    'background-color', overviewer.util.getMapTypeBackgroundColor(
                        overviewer.map.getMapTypeId()));
                //smuggled this one in here for maptypeid hash generation --CounterPillow
                overviewer.util.updateHash();
            });
        },
        /**
         * Read through overviewer.collections.markerDatas and create Marker
         * objects and stick them in overviewer.collections.markers . This
         * should probably be done differently at some point so that we can
         * support markers that change position more easily.
         */
        'initializeMarkers': function() {
            //first, give all collections an empty array to work with
            for (i in overviewerConfig.objectGroups.signs) {
                overviewer.util.debug('Found sign group: ' +
                    overviewerConfig.objectGroups.signs[i].label);
                overviewer.collections.markers[
                    overviewerConfig.objectGroups.signs[i].label] = [];
            }
            for (i in overviewerConfig.objectGroups.animals) {
                overviewer.util.debug('Found animal group: ' +
                    overviewerConfig.objectGroups.animals[i].label);
                overviewer.collections.markers[
                    overviewerConfig.objectGroups.animals[i].label] = [];
            }
            for (i in overviewer.collections.markerDatas) {
                var markerData = overviewer.collections.markerDatas[i];
                for (j in markerData) {
                    var item = markerData[j];
                    // a default:
                    var iconURL = '';
                    if (item.type == 'spawn') { 
                        // don't filter spawn, always display
                        var marker = new google.maps.Marker({
                            'position': overviewer.util.fromWorldToLatLng(item.x,
                                item.y, item.z),
                             'map':     overviewer.map,
                             'title':   jQuery.trim(item.msg),
                             'icon':    overviewerConfig.CONST.image.spawnMarker
                        });
                        continue;
                    }

                    if (item.type == 'querypos') { 
                        // Set on page load if MC x/y/z coords are given in the
                        // query string
                        var marker = new google.maps.Marker({
                            'position': overviewer.util.fromWorldToLatLng(item.x,
                                item.y, item.z),
                             'map':     overviewer.map,
                             'title':   jQuery.trim(item.msg),
                             'icon':    overviewerConfig.CONST.image.queryMarker
                        });
                        google.maps.event.addListener(marker, 'click', function(){ marker.setVisible(false); });

                        continue;
                    }

                    if (item.type == 'cow') { 
                        var marker = new google.maps.Marker({
                            'position': overviewer.util.fromWorldToLatLng(item.x, item.y, item.z),
                             'map':     overviewer.map,
                             'icon':    overviewerConfig.CONST.image.cowMarker,
							 'visible':  false
                        });
						overviewer.collections.markers['ark'].push(marker);
						overviewer.collections.markers[item.type].push(marker);
                        continue;
                    }
                    if (item.type == 'sheep') { 
                        var marker = new google.maps.Marker({
                            'position': overviewer.util.fromWorldToLatLng(item.x, item.y, item.z),
                             'map':     overviewer.map,
                             'icon':    overviewerConfig.CONST.image.sheepMarker,
							 'visible':  false
                        });
						overviewer.collections.markers['ark'].push(marker);
						overviewer.collections.markers[item.type].push(marker);
                        continue;
                    }
                    if (item.type == 'pig') { 
                        var marker = new google.maps.Marker({
                            'position': overviewer.util.fromWorldToLatLng(item.x, item.y, item.z),
                             'map':     overviewer.map,
                             'icon':    overviewerConfig.CONST.image.pigMarker,
							 'visible':  false
                        });
						overviewer.collections.markers['ark'].push(marker);
						overviewer.collections.markers[item.type].push(marker);
                        continue;
                    }
                    if (item.type == 'chicken') { 
                        var marker = new google.maps.Marker({
                            'position': overviewer.util.fromWorldToLatLng(item.x, item.y, item.z),
                             'map':     overviewer.map,
                             'icon':    overviewerConfig.CONST.image.chickenMarker,
							 'visible':  false
                        });
						overviewer.collections.markers['ark'].push(marker);
						overviewer.collections.markers[item.type].push(marker);
                        continue;
                    }
                    if (item.type == 'squid') { 
                        var marker = new google.maps.Marker({
                            'position': overviewer.util.fromWorldToLatLng(item.x, item.y, item.z),
                             'map':     overviewer.map,
                             'icon':    overviewerConfig.CONST.image.squidMarker,
							 'visible':  false
                        });
						overviewer.collections.markers['ark'].push(marker);
						overviewer.collections.markers[item.type].push(marker);
                        continue;
                    }

                    var matched = false;
                    for (j in overviewerConfig.objectGroups.signs) {
                        var signGroup = overviewerConfig.objectGroups.signs[j];
                        var label = signGroup.label;
                        if (signGroup.match(item)) {
                            matched = true;
                            // can add custom types of images for externally defined
                            // item types, like 'command' here.
                            if (item.type == 'sign') {
                                iconURL = overviewerConfig.CONST.image.signMarker;
                            }
                            overviewer.util.debug('Sign icon: ' + signGroup.icon);
                            if (signGroup.icon) {
                                iconURL = signGroup.icon;
                            }
                            var marker = new google.maps.Marker({
                                'position': overviewer.util.fromWorldToLatLng(item.x,
                                    item.y, item.z),
                                'map':      overviewer.map,
                                'title':    jQuery.trim(item.msg), 
                                'icon':     iconURL,
                                'visible':  false
                            });
                            item.marker = marker;
                            overviewer.util.debug(label);
                            overviewer.collections.markers[label].push(marker);
                            if (item.type == 'sign') {
                                overviewer.util.createMarkerInfoWindow(marker);
                            }
                        }
                    }
					
                    if (!matched) {
                        // is this signpost doesn't match any of the groups in
                        // config.js, add it automatically to the "__others__" group
                        if (item.type == 'sign') {
                            iconURL = overviewerConfig.CONST.image.signMarker;
                        }
                        var marker = new google.maps.Marker({
                            'position': overviewer.util.fromWorldToLatLng(item.x,
                                item.y, item.z),
                            'map':      overviewer.map,
                            'title':    jQuery.trim(item.msg), 
                            'icon':     iconURL,
                            'visible':  false
                        });
                        item.marker = marker;
                        if (overviewer.collections.markers['__others__']) {
                            overviewer.collections.markers['__others__'].push(marker);
                        } else {
                            overviewer.collections.markers['__others__'] = [marker];
                        }
                        if (item.type == 'sign') {
                            overviewer.util.createMarkerInfoWindow(marker, item);
                        }
                    }
                }
            }
        },
        /**
         * Same as initializeMarkers() for the most part.
         */
        'initializeRegions': function() {
            for (i in overviewerConfig.objectGroups.regions) {
                overviewer.collections.regions[overviewerConfig.objectGroups.regions[i].label] = [];
            }
            for (i in overviewer.collections.regionDatas) {
                var regionData = overviewer.collections.regionDatas[i];
                for (j in regionData) {
                    var region = regionData[j];
                    // pull all the points out of the regions file.
                    var converted = new google.maps.MVCArray();
                    for (k in region.path) {
                        var point = region.path[k];
                        converted.push(overviewer.util.fromWorldToLatLng(
                            point.x, point.y, point.z));

                    }
                    
                    if (region.label) {
                        var name = region.label;
                    } else {
                        var name = "rawr";
                    }
                    
                    if(region.opacity) {
                        var strokeOpacity = region.opacity;
                        var fillOpacity = region.opacity * 0.25;
                    } else {
                        var strokeOpacity = region.strokeOpacity;
                        var fillOpacity = region.fillOpacity;
                    }
                    
                    var shapeOptions = {
                            'name':             name,
                            'geodesic':         false,
                            'map':              null,
                            'strokeColor':      region.color,
                            'strokeOpacity':    strokeOpacity,
                            'strokeWeight':     overviewerConfig.CONST.regionStrokeWeight,
                            'zIndex':           j
                    };
                    if (region.closed) {
                        shapeOptions["fillColor"] = region.color;
                        shapeOptions["fillOpacity"] = fillOpacity;
                        shapeOptions["paths"] = converted;
                    } else {
                        shapeOptions["path"] = converted;
                    }

                    var matched = false;

                    for (k in overviewerConfig.objectGroups.regions) {
                        var regionGroup = overviewerConfig.objectGroups.regions[k];
                        var clickable = regionGroup.clickable;
                        var label = regionGroup.label;
                        
                        if (!regionGroup.match(region))
                            continue;
                        matched = true;
                        
                        if (!region.label) {
                            clickable = false; // if it doesn't have a name, we dont have to show it.
                        }

                        if (region.closed) {
                            var shape = new google.maps.Polygon(shapeOptions);
                        } else {
                            var shape = new google.maps.Polyline(shapeOptions);
                        }

                        overviewer.collections.regions[label].push(shape); 

                        if (clickable) {
                            overviewer.util.createRegionInfoWindow(shape);
                        }
                    }
                    
                    // if we haven't matched anything, go ahead and add it
                    if (!matched) {
                        if (region.closed) {
                            var shape = new google.maps.Polygon(shapeOptions);
                        } else {
                            var shape = new google.maps.Polyline(shapeOptions);
                        }
                        
                        shape.setMap(overviewer.map);
                    }
                }
            }
        },
        /**
         * Change the map's div's background color according to the mapType's bg_color setting
         *
         * @param string mapTypeId
         * @return string
         */
        'getMapTypeBackgroundColor': function(mapTypeId) {
            for(i in overviewerConfig.mapTypes) {
                if( overviewerConfig.CONST.mapDivId +
                        overviewerConfig.mapTypes[i].shortname == mapTypeId ) {
                    overviewer.util.debug('Found background color for: ' +
                        overviewerConfig.mapTypes[i].bg_color);
                    return overviewerConfig.mapTypes[i].bg_color;
                }
            }
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
         * 
         * @return google.maps.LatLng
         */
        'fromWorldToLatLng': function(x, z, y) {
            // the width and height of all the highest-zoom tiles combined,
            // inverted
            var perPixel = 1.0 / (overviewerConfig.CONST.tileSize *
                Math.pow(2, overviewerConfig.map.zoomLevels));

            if(overviewerConfig.map.north_direction == 'upper-left'){
                temp = x;
                x = -y-1;
                y = temp;
            } else if(overviewerConfig.map.north_direction == 'upper-right'){
                x = -x-1;
                y = -y-1;
            } else if(overviewerConfig.map.north_direction == 'lower-right'){
                temp = x;
                x = y;
                y = -temp-1;
            }

            // This information about where the center column is may change with
            // a different drawing implementation -- check it again after any
            // drawing overhauls!

            // point (0, 0, 127) is at (0.5, 0.0) of tile (tiles/2 - 1, tiles/2)
            // so the Y coordinate is at 0.5, and the X is at 0.5 -
            // ((tileSize / 2) / (tileSize * 2^zoomLevels))
            // or equivalently, 0.5 - (1 / 2^(zoomLevels + 1))
            var lng = 0.5 - (1.0 / Math.pow(2, overviewerConfig.map.zoomLevels + 1));
            var lat = 0.5;

            // the following metrics mimic those in
            // chunk_render in src/iterate.c

            // each block on X axis adds 12px to x and subtracts 6px from y
            lng += 12 * x * perPixel;
            lat -= 6 * x * perPixel;

            // each block on Y axis adds 12px to x and adds 6px to y
            lng += 12 * y * perPixel;
            lat += 6 * y * perPixel;

            // each block down along Z adds 12px to y
            lat += 12 * (128 - z) * perPixel;

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
        'fromLatLngToWorld': function(lat, lng) {
            // Initialize world x/y/z object to be returned
            var point = Array();
            point.x = 0;
            point.y = 64;
            point.z = 0;

            // the width and height of all the highest-zoom tiles combined,
            // inverted
            var perPixel = 1.0 / (overviewerConfig.CONST.tileSize *
                Math.pow(2, overviewerConfig.map.zoomLevels));

            // Revert base positioning
            // See equivalent code in fromWorldToLatLng()
            lng -= 0.5 - (1.0 / Math.pow(2, overviewerConfig.map.zoomLevels + 1));
            lat -= 0.5;

            // I'll admit, I plugged this into Wolfram Alpha:
            //   a = (x * 12 * r) + (z * 12 * r), b = (z * 6 * r) - (x * 6 * r)
            // And I don't know the math behind solving for for X and Z given
            // A (lng) and B (lat).  But Wolfram Alpha did. :)  I'd welcome
            // suggestions for splitting this up into long form and documenting
            // it. -RF
            point.x = (lng - 2 * lat) / (24 * perPixel)
            point.z = (lng + 2 * lat) / (24 * perPixel)

            // Adjust for the fact that we we can't figure out what Y is given
            // only latitude and longitude, so assume Y=64.
            point.x += 64;
            point.z -= 64;

            if(overviewerConfig.map.north_direction == 'upper-left'){
                temp = point.z;
                point.z = -point.x;
                point.x = temp;
            } else if(overviewerConfig.map.north_direction == 'upper-right'){
                point.x = -point.x;
                point.z = -point.z;
            } else if(overviewerConfig.map.north_direction == 'lower-right'){
                temp = point.z;
                point.z = point.x;
                point.x = -temp;
            }

            return point;
        },
        /**
         * Create and draw the various map controls and other related things
         * like the compass, current view link, etc.
         */
        'createMapControls': function() {
            // compass rose, in the top right corner
            var compassDiv = document.createElement('DIV');
            compassDiv.style.padding = '5px';
            var compassImg = document.createElement('IMG');
            compassImg.src = overviewerConfig.CONST.image.compass;
            compassDiv.appendChild(compassImg);
            compassDiv.index = 0;
            // add it to the map, top right.
            if (overviewerConfig.map.controls.compass) {
                overviewer.map.controls[google.maps.ControlPosition.TOP_RIGHT].push(compassDiv);
            }

            // Spawn button
            var homeControlDiv = document.createElement('DIV');
            var homeControl = new overviewer.classes.HomeControl(homeControlDiv);  
            $(homeControlDiv).addClass('customControl');
            homeControlDiv.index = 1;
            if (overviewerConfig.map.controls.spawn) {
                overviewer.map.controls[google.maps.ControlPosition.TOP_RIGHT].push(homeControlDiv);
            }

            // Coords box
            var coordsDiv = document.createElement('DIV');
            coordsDiv.id = 'coordsDiv';
            coordsDiv.innerHTML = '';
            if (overviewerConfig.map.controls.coordsBox) {
                overviewer.map.controls[google.maps.ControlPosition.BOTTOM_LEFT].push(coordsDiv);
            }
            
            // Update coords on mousemove
            google.maps.event.addListener(overviewer.map, 'mousemove', function (event) {
                var worldcoords = overviewer.util.fromLatLngToWorld(event.latLng.lat(), event.latLng.lng());
                coordsDiv.innerHTML = "Coords: X " + Math.round(worldcoords.x) + ", Z " + Math.round(worldcoords.z);
            });

            // only need to create the control if there are items in the list.
            // as defined in config.js
            if (overviewerConfig.objectGroups.signs.length > 0) {
                // signpost display control
                var items = [];
                for (i in overviewerConfig.objectGroups.signs) {
                    var signGroup = overviewerConfig.objectGroups.signs[i];
                    // don't create an option for this group if empty
                    if (overviewer.collections.markers[signGroup.label].length == 0) {
                        continue;
                    }
                    
                    var iconURL = signGroup.icon;
                    if(!iconURL) {
                        iconURL = overviewerConfig.CONST.image.defaultMarker;
                    }
                    items.push({
                        'label': signGroup.label, 
                        'checked': signGroup.checked,
                        'icon': iconURL,
                        'action': function(n, item, checked) {
                            jQuery.each(overviewer.collections.markers[item.label],
                                        function(i, elem) {
                                            elem.setVisible(checked);
                                        }
                            );
                            overviewer.util.debug('Adding sign item: ' + item);
                        }
                    });
                }
                
                // only create drop down if there's used options
                if (items.length > 0) {
                    overviewer.util.createDropDown('Markers', items);
                }
            }


            // only need to create the control if there are items in the list.
            // as defined in config.js
            if (overviewerConfig.objectGroups.animals.length > 0) {
                // signpost display control
                var items = [];
                for (i in overviewerConfig.objectGroups.animals) {
                    var signGroup = overviewerConfig.objectGroups.animals[i];
                    // don't create an option for this group if empty
                    if (overviewer.collections.markers[signGroup.label].length == 0) {
                        continue;
                    }
                    
                    var iconURL = signGroup.icon;
                    if(!iconURL) {
                        iconURL = overviewerConfig.CONST.image.defaultMarker;
                    }
                    items.push({
                        'label': signGroup.label, 
                        'checked': signGroup.checked,
                        'icon': iconURL,
                        'action': function(n, item, checked) {
                            jQuery.each(overviewer.collections.markers[item.label],
                                        function(i, elem) {
                                            elem.setVisible(checked);
                                        }
                            );
                            overviewer.util.debug('Adding sign item: ' + item);
                        }
                    });
                }
                
                // only create drop down if there's used options
                if (items.length > 0) {
                    overviewer.util.createDropDown('Animals', items);
                }
            }

            // if there are any regions data, lets show the option to hide/show them.
            if (overviewerConfig.objectGroups.regions.length > 0) {
                // region display control
                var items = [];
                for (i in overviewerConfig.objectGroups.regions) {
                    var regionGroup = overviewerConfig.objectGroups.regions[i];
                    items.push({
                        'label': regionGroup.label, 
                        'checked': regionGroup.checked,
                        'action': function(n, item, checked) {
                            jQuery.each(overviewer.collections.regions[item.label],
                                function(i,elem) {
                                    // Thanks to LeastWeasel for this line!
                                    elem.setMap(checked ? overviewer.map : null);
                                });
                                overviewer.util.debug('Adding region item: ' + item);

                        }
                    });
                }
                overviewer.util.createDropDown('Regions', items);
            }

            if (overviewerConfig.map.controls.overlays && overviewer.collections.overlays.length > 0) {
                // overlay maps control
                var items = [];
                for (i in overviewer.collections.overlays) {
                    var overlay = overviewer.collections.overlays[i];
                    items.push({
                        'label':    overlay.name,
                        'checked':  false,
                        'overlay':  overlay,
                        'action':   function(i, item, checked) {
                            if (checked) {
                                overviewer.map.overlayMapTypes.push(item.overlay);
                            } else {
                                var idx_to_delete = -1;
                                overviewer.map.overlayMapTypes.forEach(function(e, j) {
                                    if (e == item.overlay) {
                                        idx_to_delete = j;
                                    }
                                });
                                if (idx_to_delete >= 0) {
                                    overviewer.map.overlayMapTypes.removeAt(idx_to_delete);
                                }
                            }
                        }
                    });
                }
                overviewer.util.createDropDown('Overlays', items);
            }
            
            // call out to create search box, as it's pretty complicated
            overviewer.util.createSearchBox();
        },
        /**
         * Reusable method for creating drop-down menus
         * 
         * @param string title
         * @param array items
         */
        'createDropDown': function(title, items) {
            var control = document.createElement('DIV');
            // let's let a style sheet do most of the styling here
            $(control).addClass('customControl');

            var controlText = document.createElement('DIV');
            controlText.innerHTML = title;

            var controlBorder = document.createElement('DIV');
            $(controlBorder).addClass('top');
            control.appendChild(controlBorder);
            controlBorder.appendChild(controlText);

            var dropdownDiv = document.createElement('DIV');
            $(dropdownDiv).addClass('dropDown');
            control.appendChild(dropdownDiv);
            dropdownDiv.innerHTML='';

            // add the functionality to toggle visibility of the items
            $(controlText).click(function() {
                    $(controlBorder).toggleClass('top-active');
                    $(dropdownDiv).toggle();
                });

            // add that control box we've made back to the map.
            overviewer.map.controls[google.maps.ControlPosition.TOP_RIGHT].push(control);

            for(i in items) {
                // create the visible elements of the item
                var item = items[i];
                overviewer.util.debug(item);
                var itemDiv = document.createElement('div');
                var itemInput = document.createElement('input');
                itemInput.type='checkbox';

                // give it a name
                $(itemInput).data('label',item.label);
                jQuery(itemInput).click((function(local_idx, local_item) {
                                     return function(e) {
                                         item.action(local_idx, local_item, e.target.checked);
                                     };
                                 })(i, item));

                // if its checked, its gotta do something, do that here.
                if (item.checked) {
                    itemInput.checked = true;
                    item.action(i, item, item.checked);
                }
                dropdownDiv.appendChild(itemDiv);
                itemDiv.appendChild(itemInput);
                var textNode = document.createElement('text');
                if(item.icon) {
                    textNode.innerHTML = '<img width="15" height="15" src="' + 
                        item.icon + '">' + item.label + '<br/>';
                } else {
                    textNode.innerHTML = item.label + '<br/>';
                }

                itemDiv.appendChild(textNode);
            }
        },
        /**
         * Create search box and dropdown in the top right google maps area.
        */
        'createSearchBox': function() {
            var searchControl = document.createElement("div");
            searchControl.id = "searchControl";

            var searchInput = document.createElement("input");
            searchInput.type = "text";
            searchInput.value = "Sign Search";
            searchInput.title = "Sign Search";
            $(searchInput).addClass("inactive");
            
            /* Hey dawg, I heard you like functions.
            * So we defined a function inside your function.
             */
            searchInput.onfocus = function() {
                if (searchInput.value == "Sign Search") {
                    searchInput.value = "";
                    $(searchInput).removeClass("inactive").addClass("active");
                }
            };
            searchInput.onblur = function() {
                if (searchInput.value == "") {
                    searchInput.value = "Sign Search";
                    $(searchInput).removeClass("active").addClass("inactive");
                }
            };

            searchControl.appendChild(searchInput);

            var searchDropDown = document.createElement("div");
            searchDropDown.id = "searchDropDown";
            searchControl.appendChild(searchDropDown);

            $(searchInput).keyup(function(e) {
                var newline_stripper = new RegExp("[\\r\\n]", "g")
                if(searchInput.value.length !== 0) {
                    $(searchDropDown).fadeIn();
                        
                    $(searchDropDown).empty();

                    overviewer.collections.markerDatas.forEach(function(marker_list) {
                        marker_list.forEach(function(sign) {
                            var regex = new RegExp(overviewer.util.pregQuote(searchInput.value), "mi");
                            if(sign.msg.match(regex)) {
                                if(sign.marker !== undefined && sign.marker.getVisible()) {
                                    var t = document.createElement("div");
                                    t.className = "searchResultItem";
                                    var i = document.createElement("img");
                                    i.src = sign.marker.getIcon();
                                    t.appendChild(i);
                                    var s = document.createElement("span");
                                    
                                    $(s).text(sign.msg.replace(newline_stripper, ""));
                                    t.appendChild(s);
                                    searchDropDown.appendChild(t);
                                    $(t).click(function(e) {
                                        $(searchDropDown).fadeOut();
                                        overviewer.map.setZoom(7);
                                        overviewer.map.setCenter(sign.marker.getPosition());
                                    });

                                }
                            }
                        });
                    });
                } else {
                    $(searchDropDown).fadeOut();
                }
            });
            
            if (overviewerConfig.map.controls.searchBox) {
                overviewer.map.controls[google.maps.ControlPosition.TOP_RIGHT].push(searchControl);
            }
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
            var windowContent = '<div class="infoWindow"><img src="' + marker.icon +
                '"/><p>' + marker.title.replace(/\n/g,'<br/>') + '</p></div>';
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
            if(window.location.hash.split("/").length > 1) {
                overviewer.util.goToHash();
                // Clean up the hash.
                overviewer.util.updateHash();
                
                // Add a marker indicating the user-supplied position
                var coordinates = overviewer.util.fromLatLngToWorld(overviewer.map.getCenter().lat(), overviewer.map.getCenter().lng());
                overviewer.collections.markerDatas.push([{
                    'msg': 'Coordinates ' + Math.floor(coordinates.x) + ', ' + Math.floor(coordinates.y) + ', ' + Math.floor(coordinates.z),
                    'x': coordinates.x,
                    'y': coordinates.y,
                    'z': coordinates.z,
                    'type': 'querypos'}]);
            }
        },
        'setHash': function(x, y, z, zoom, maptype)    {
            // remove the div prefix from the maptype (looks better)
            if (maptype)
            {
                maptype = maptype.replace(overviewerConfig.CONST.mapDivId, "");
            }
            window.location.replace("#/" + Math.floor(x) + "/" + Math.floor(y) + "/" + Math.floor(z) + "/" + zoom + "/" + maptype);
        },
        'updateHash': function() {
            var coordinates = overviewer.util.fromLatLngToWorld(overviewer.map.getCenter().lat(), overviewer.map.getCenter().lng());
            var zoom = overviewer.map.getZoom();
            var maptype = overviewer.map.getMapTypeId();
            if (zoom == overviewerConfig.map.maxZoom) {
                zoom = 'max';
            } else if (zoom == overviewerConfig.map.minZoom) {
                zoom = 'min';
            } else {
                // default to (map-update friendly) negative zooms
                zoom -= overviewerConfig.map.maxZoom;
            }
            overviewer.util.setHash(coordinates.x, coordinates.y, coordinates.z, zoom, maptype);
        },
        'goToHash': function() {
            // Note: the actual data begins at coords[1], coords[0] is empty.
            var coords = window.location.hash.split("/");
            var latlngcoords = overviewer.util.fromWorldToLatLng(parseInt(coords[1]), parseInt(coords[2]), parseInt(coords[3]));
            var zoom;
            var maptype = '';
            // The if-statements try to prevent unexpected behaviour when using incomplete hashes, e.g. older links
            if (coords.length > 4) {
                zoom = coords[4];
            }
            if (coords.length > 5) {
                maptype = coords[5];
            }
            
            if (zoom == 'max') {
                zoom = overviewerConfig.map.maxZoom;
            } else if (zoom == 'min') {
                zoom = overviewerConfig.map.minZoom;
            } else {
                zoom = parseInt(zoom);
                if (zoom < 0 && zoom + overviewerConfig.map.maxZoom >= 0) {
                    // if zoom is negative, treat it as a "zoom out from max"
                    zoom += overviewerConfig.map.maxZoom;
                } else {
                    // fall back to default zoom
                    zoom = overviewerConfig.map.defaultZoom;
                }
            }
            // If the maptype isn't set, set the default one.
            if (maptype == '') {
                // We can now set the map to use the 'coordinate' map type
                overviewer.map.setMapTypeId(overviewer.util.getDefaultMapTypeId());
            } else {
                // normalize the map type (this supports old-style,
                // 'mcmapLabel' style map types, converts them to 'shortname'
                if (maptype.lastIndexOf(overviewerConfig.CONST.mapDivId, 0) === 0) {
                    maptype = maptype.replace(overviewerConfig.CONST.mapDivId, "");
                    for (i in overviewer.collections.mapTypes) {
                        var type = overviewer.collections.mapTypes[i];
                        if (type.name == maptype) {
                            maptype = type.shortname;
                            break;
                        }
                    }
                }
                
                overviewer.map.setMapTypeId(overviewerConfig.CONST.mapDivId + maptype);
            }
            
            overviewer.map.setCenter(latlngcoords);
            overviewer.map.setZoom(zoom);
        }
    },
    /**
     * The various classes needed in this file.
     */
    'classes': {
        /**
         * This is the button that centers the map on spawn. Not sure why we
         * need a separate class for this and not some of the other controls.
         * 
         * @param documentElement controlDiv
         */
        'HomeControl': function(controlDiv) {
            controlDiv.style.padding = '5px';
            // Set CSS for the control border
            var control = document.createElement('DIV');
            $(control).addClass('top');
            control.title = 'Click to center the map on the Spawn';
            controlDiv.appendChild(control);

            // Set CSS for the control interior
            var controlText = document.createElement('DIV');
            controlText.innerHTML = 'Spawn';
            $(controlText).addClass('button');
            control.appendChild(controlText);

            // Setup the click event listeners: simply set the map to map center
            // as definned below
            google.maps.event.addDomListener(control, 'click', function() {
                    overviewer.map.panTo(overviewer.util.fromWorldToLatLng(
                        overviewerConfig.map.center[0],
                        overviewerConfig.map.center[1],
                        overviewerConfig.map.center[2]));
                    overviewer.util.updateHash();
                });
        },
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
    },
    /**
     * Stuff that we give to the google maps code instead of using ourselves
     * goes in here.
     * 
     * Also, why do I keep writing these comments as if I'm multiple people? I
     * should probably stop that.
     */
    'gmap': {
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
                } else if(zoom == 0) {
                    url += '/base';
                } else {
                    for(var z = zoom - 1; z >= 0; --z) {
                        var x = Math.floor(tile.x / Math.pow(2, z)) % 2;
                        var y = Math.floor(tile.y / Math.pow(2, z)) % 2;
                        url += '/' + (x + 2 * y);
                    }
                }
                url = url + '.' + pathExt;
                if(overviewerConfig.map.cacheMinutes > 0) {
                    var d = new Date();
                    url += '?c=' + Math.floor(d.getTime() /
                        (1000 * 60 * overviewerConfig.map.cacheMinutes));
                }
                return(urlBase + url);
            }
        }
    }
};
