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
        overviewer.util.initializePolyfills();

        overviewer.coordBoxClass = L.Control.extend({
            options: {
                position: 'bottomleft',
            },
            initialize: function() {
                this.coord_box = L.DomUtil.create('div', 'coordbox');
            },
            render: function(latlng) {
                var currWorld = overviewer.current_world;
                if (currWorld == null) {return;}

                var currTileset = overviewer.current_layer[currWorld];
                if (currTileset == null) {return;}

                var ovconf = currTileset.tileSetConfig;

                w_coords = overviewer.util.fromLatLngToWorld(latlng.lat, latlng.lng, ovconf);

                var r_x = Math.floor(Math.floor(w_coords.x / 16.0) / 32.0);
                var r_z = Math.floor(Math.floor(w_coords.z / 16.0) / 32.0);
                var r_name = "r." + r_x + "." + r_z + ".mca";

                this.coord_box.innerHTML = "X " +
                                           Math.round(w_coords.x) +
                                           " Z " + Math.round(w_coords.z) +
                                           " (" + r_name + ")";
            },
            onAdd: function() {
                return this.coord_box;
            }
        });
        overviewer.progressClass = L.Control.extend({
            options: {
                position: 'bottomright'
            },
            initialize: function() {
                this.progress = L.DomUtil.create("div", "progress");
                this.progress.innerHTML = 'Current render progress';
                this.progress.style.visibility = 'hidden';
            },
            update: function() {
                fetch("progress.json")
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Response was not ok');
                        }
                        return response.json();
                    })
                    .then(data => {
                        this.progress.innerHTML = data.message;
                        if (data.update > 0) {
                            setTimeout(this.update.bind(this), data.update);
                            this.progress.style.visibility = '';
                        } else {
                            setTimeout(this.update.bind(this), 60000);
                            this.progress.innerHTML = 'Hidden - data.update < 0';
                            this.progress.style.visibility = 'hidden';
                        }
                    })
                    .catch(error => {
                        this.progress.innerHtml = 'Hidden - no data';
                        this.progress.style.visibility = 'hidden';
                        console.info('Error getting progress; hiding control', error);
                    });
            },
            onAdd: function() {
                // Not all browsers may have this
                if ('fetch' in window) {
                    setTimeout(this.update.bind(this), 0);
                }
                return this.progress;
            }
        });
        overviewer.compassClass = L.Control.extend({
            initialize: function(imagedict, options) {
                L.Util.setOptions(this, options);
                this.compass_img = L.DomUtil.create('img', 'compass');
                this.imagedict = imagedict;
            },
            render: function(direction) {
                this.compass_img.src = this.imagedict[direction];
            },
            onAdd: function() {
                return this.compass_img;
            }
        });
        overviewer.control = L.Control.extend({
            initialize: function(options) {
                L.Util.setOptions(this, options);

                this.container = L.DomUtil.create('div', 'worldcontrol');
                this.select = L.DomUtil.create('select');
                this.select.onchange = this.onChange;
                this.container.appendChild(this.select);
            },
            addWorld: function(world) {
                var option = L.DomUtil.create('option');
                option.value = world;
                option.innerText = world;
                this.select.appendChild(option);
            },
            onChange: function(ev) {
                overviewer.util.debug(ev.target);
                overviewer.util.debug(ev.target.value);
                var selected_world = ev.target.value;


                // save current view for the current_world
                let current_center = [overviewer.map.getCenter(), overviewer.map.getZoom()];
                let current_layer = overviewer.current_layer[overviewer.current_world] ||
                    Object.values(overviewer.collections.mapTypes[overviewer.current_world])[0];
                let layer_name = current_layer.tileSetConfig.path;
                overviewer.collections.centers[overviewer.current_world][layer_name] = current_center;

                overviewer.layerCtrl.remove();

                var base_layers = {};
                var overlay_layers = {};
                for (var bl in overviewer.collections.mapTypes[selected_world]) {
                    var bl_o = overviewer.collections.mapTypes[selected_world][bl];
                    base_layers[bl_o.tileSetConfig.name] = bl_o;
                }
                for (var ol in overviewer.collections.overlays[selected_world]) {
                    var ol_o = overviewer.collections.overlays[selected_world][ol];
                    overlay_layers[ol_o.tileSetConfig.name] = ol_o;
                }

                overviewer.layerCtrl = L.control.layers(
                    base_layers,
                    overlay_layers,
                    {collapsed: false})
                .addTo(overviewer.map);

                for (var world_name in overviewer.collections.mapTypes) {
                    for (var tset_name in overviewer.collections.mapTypes[world_name]) {
                        var lyr = overviewer.collections.mapTypes[world_name][tset_name];
                        if (world_name != selected_world) {
                            if (overviewer.map.hasLayer(lyr))
                                overviewer.map.removeLayer(lyr);
                        }
                        if (lyr.tileSetConfig.marker_groups) {
                            for (var marker_group in lyr.tileSetConfig.marker_groups) {
                                lyr.tileSetConfig.marker_groups[marker_group].remove();
                            }
                        }
                        if (lyr.tileSetConfig.markerCtrl) {
                            lyr.tileSetConfig.markerCtrl.remove();
                        }
                    }

                    for (var tset_name in overviewer.collections.overlays[world_name]) {
                        var lyr = overviewer.collections.overlays[world_name][tset_name];
                        if (world_name != selected_world) {
                            if (overviewer.map.hasLayer(lyr))
                                overviewer.map.removeLayer(lyr);
                        }
                    }
                }

                let selected_layer_name = overviewer.collections.mapTypes[selected_world] && overviewer.current_layer[selected_world] ?
                    overviewer.current_layer[selected_world].tileSetConfig.path :
                    Object.keys(overviewer.collections.mapTypes[selected_world])[0];

                let center = overviewer.collections.centers[selected_world][selected_layer_name];
                overviewer.map.setView(center[0], center[1]);

                overviewer.current_world = selected_world;
                overviewer.map.addLayer(overviewer.collections.mapTypes[selected_world][selected_layer_name]);
            },
            onAdd: function() {
                return this.container
            }
        });



        overviewer.map = L.map('mcmap', {crs: L.CRS.Simple});

        overviewer.map.attributionControl.setPrefix(
            '<a href="https://overviewer.org">Overviewer/Leaflet</a>');

        overviewer.map.on('baselayerchange', function(ev) {
            
            // when changing the layer, ensure coordinates remain correct
            if (overviewer.current_layer[overviewer.current_world]) {
                const center = overviewer.map.getCenter();
                const currentWorldCoords = overviewer.util.fromLatLngToWorld(
                        center.lat, 
                        center.lng, 
                        overviewer.current_layer[overviewer.current_world].tileSetConfig);
                    
                const newMapCoords = overviewer.util.fromWorldToLatLng(
                        currentWorldCoords.x, 
                        currentWorldCoords.y, 
                        currentWorldCoords.z, 
                        ev.layer.tileSetConfig);
                        
                overviewer.map.setView(
                        newMapCoords,
                        overviewer.map.getZoom(),
                        { animate: false });
            }
            
            // before updating the current_layer, remove the marker control, if it exists
            if (overviewer.current_world && overviewer.current_layer[overviewer.current_world]) {
                let tsc = overviewer.current_layer[overviewer.current_world].tileSetConfig;

                if (tsc.markerCtrl)
                    tsc.markerCtrl.remove();
                if (tsc.marker_groups) {
                    for (var marker_group in tsc.marker_groups) {
                        tsc.marker_groups[marker_group].remove();
                    }
                }
            }
            overviewer.current_layer[overviewer.current_world] = ev.layer;
            var ovconf = ev.layer.tileSetConfig;

            // Change the compass
            overviewer.compass.render(ovconf.north_direction);

            // Set the background colour
            document.getElementById("mcmap").style.backgroundColor = ovconf.bgcolor;

            if (overviewer.collections.locationMarker) {
                overviewer.collections.locationMarker.remove();
            }
            // Remove old spawn marker, add new one
            if (overviewer.collections.spawnMarker) {
                overviewer.collections.spawnMarker.remove();
            }
            if (typeof(ovconf.spawn) == "object") {
                var spawnIcon = L.icon({
                    iconUrl: overviewerConfig.CONST.image.spawnMarker,
                    iconRetinaUrl: overviewerConfig.CONST.image.spawnMarker2x,
                    iconSize: [32, 37],
                    iconAnchor: [15, 33],
                });
                var latlng = overviewer.util.fromWorldToLatLng(ovconf.spawn[0],
                                                               ovconf.spawn[1],
                                                               ovconf.spawn[2],
                                                               ovconf);
                var ohaimark = L.marker(latlng, {icon: spawnIcon, title: "Spawn"});
                ohaimark.on('click', function(ev) {
                    overviewer.map.setView(ev.latlng);
                });
                overviewer.collections.spawnMarker = ohaimark
                overviewer.collections.spawnMarker.addTo(overviewer.map);
            } else {
                overviewer.collections.spawnMarker = null;
            }

            // reset the markers control with the markers for this layer
            if (ovconf.marker_groups) {
                overviewer.util.debug("markers for", ovconf.marker_groups);
                ovconf.markerCtrl = L.control.layers(
                        [],
                        ovconf.marker_groups, {collapsed: false}).addTo(overviewer.map);
            }
            for (var marker_group in ovconf.marker_groups) {
                var mg = ovconf.marker_groups[marker_group];
                if (mg.options.default_checked) {
                    mg.addTo(overviewer.map);
                }
            }
            // Update overlays
            for (var olw in overviewer.collections.overlays) {
                for (var ol in overviewer.collections.overlays[olw]) {
                    var ol_o = overviewer.collections.overlays[olw][ol];
                    if (ol_o.tileSetConfig.isOverlay.includes(ovconf.path)) {
                        if (!overviewer.util.isInLayerCtrl(overviewer.layerCtrl, ol_o)) {
                            overviewer.layerCtrl.addOverlay(ol_o, ol_o.tileSetConfig.name);
                        }
                    } else {
                        if (overviewer.util.isInLayerCtrl(overviewer.layerCtrl, ol_o)) {
                            overviewer.layerCtrl.removeLayer(ol_o);
                        }
                    }
                }
            }


            overviewer.util.updateHash();
        });

        overviewer.map.on('moveend', function(ev) {
            overviewer.util.updateHash();
        });

        var tset = overviewerConfig.tilesets[0];

        overviewer.map.on("click", function(e) {
            var point = overviewer.util.fromLatLngToWorld(e.latlng.lat, e.latlng.lng, tset);
        });

        var tilesetLayers = {}

        overviewer.worldCtrl = new overviewer.control();
        overviewer.compass = new overviewer.compassClass(
            overviewerConfig.CONST.image.compass);
        overviewer.coord_box = new overviewer.coordBoxClass();
        overviewer.progress = new overviewer.progressClass();


        overviewerConfig.worlds.forEach(function(world_name, idx) {
            overviewer.collections.mapTypes[world_name] = {}
            overviewer.collections.overlays[world_name] = {}
            overviewer.worldCtrl.addWorld(world_name);
        });

        overviewer.compass.addTo(overviewer.map);
        overviewer.worldCtrl.addTo(overviewer.map);
        overviewer.coord_box.addTo(overviewer.map);
        overviewer.progress.addTo(overviewer.map);

        overviewer.map.on('mousemove', function(ev) {
            overviewer.coord_box.render(ev.latlng);
        });

        overviewerConfig.tilesets.forEach(function(obj, idx) {
            var myLayer = new L.tileLayer('', {
                tileSize: overviewerConfig.CONST.tileSize,
                noWrap: true,
                maxZoom: obj.maxZoom,
                minZoom: obj.minZoom,
                errorTileUrl: obj.base + obj.path + "/blank." + obj.imgextension,
            });
            myLayer.getTileUrl = overviewer.util.getTileUrlGenerator(obj.path, obj.base, obj.imgextension);

            if (obj.isOverlay) {
                overviewer.collections.overlays[obj.world][obj.path] = myLayer;
            } else {
                overviewer.collections.mapTypes[obj.world][obj.path] = myLayer;
            }

            obj.marker_groups = undefined;

            if (overviewer.collections.haveSigns == true) {
                // if there are markers for this tileset, create them now
                if ((typeof markers !== 'undefined') && (obj.path in markers)) {
                    overviewer.util.debug("this tileset has markers:", obj);
                    obj.marker_groups = {};

                    // For every group of markers
                    for (var mkidx = 0; mkidx < markers[obj.path].length; mkidx++) {
                        // Create a Leaflet layer group
                        var marker_group = new L.layerGroup();
                        var marker_entry = markers[obj.path][mkidx];
                        L.Util.setOptions(marker_group, {default_checked: marker_entry.checked});
                        var icon =  L.divIcon({html: `<img class="ov-marker" src="${marker_entry.icon}">`});

                        // For every marker in group
                        for (var dbidx = 0; dbidx < markersDB[marker_entry.groupName].raw.length; dbidx++) {
                            let db = markersDB[marker_entry.groupName].raw[dbidx];
                            var layerObj = undefined;

                            // Shape or marker?
                            if ('points' in db) {
                                // Convert all coords
                                plLatLng = db['points'].map(function(p) {
                                    return overviewer.util.fromWorldToLatLng(p.x, p.y, p.z, obj);
                                });
                                options = {
                                    color: db['strokeColor'],
                                    weight: db['strokeWeight'],
                                    fill: db['fill']
                                };
                                layerObj = db['isLine'] ? L.polyline(plLatLng, options) : L.polygon(plLatLng, options);
                                if (db['hovertext']) {
                                    layerObj.bindTooltip(db['hovertext'], {sticky: true});
                                }
                                // TODO: add other config options (fill color, fill opacity)
                            } else {
                                // Convert coords
                                let latlng = overviewer.util.fromWorldToLatLng(db.x, db.y, db.z, obj);
                                // Set icon and use default icon if not specified
                                let m_icon = L.divIcon({html: `<img class="ov-marker" src="${db.icon == undefined ? marker_entry.icon : db.icon}">`});
                                // Create marker
                                layerObj = new L.marker(latlng, {icon: m_icon, title: db.hovertext});
                            }
                            // Add popup to marker
                            if (marker_entry.createInfoWindow && db.text) {
                                layerObj.bindPopup(db.text);
                            }
                            // Add the polyline or marker to the layer
                            marker_group.addLayer(layerObj);
                        }
                        // Save marker group
                        var layer_name_html;
                        if (marker_entry.showIconInLegend) {
                            layer_name_html = marker_entry.displayName +
                                '<img class="ov-marker-legend" src="' + marker_entry.icon + '"></img>';
                        }
                        else {
                            layer_name_html = marker_entry.displayName;
                        }
                        obj.marker_groups[layer_name_html] = marker_group;
                    }
                }
            }

            myLayer["tileSetConfig"] = obj;

            if (!overviewer.collections.centers[obj.world]) {
                overviewer.collections.centers[obj.world] = {};
            }

            if (typeof(obj.center) == "object") {
                var latlng = overviewer.util.fromWorldToLatLng(obj.center[0], obj.center[1], obj.center[2], obj);
                overviewer.collections.centers[obj.world][obj.path] = [ latlng, obj.defaultZoom ];
            } else {
                overviewer.collections.centers[obj.world][obj.path] = [ [0, 0], obj.defaultZoom ];
            }

        });

        overviewer.layerCtrl = L.control.layers(
                overviewer.collections.mapTypes[overviewerConfig.worlds[0]],
                overviewer.collections.overlays[overviewerConfig.worlds[0]],
                {collapsed: false})
            .addTo(overviewer.map);
        overviewer.current_world = overviewerConfig.worlds[0];

        let default_layer_name = Object.keys(overviewer.collections.mapTypes[overviewer.current_world])[0];
        let center = overviewer.collections.centers[overviewer.current_world][default_layer_name];
        overviewer.map.setView(center[0], center[1]);

        if (!overviewer.util.initHash()) {
            overviewer.worldCtrl.onChange({target: {value: overviewer.current_world}});
        }

        overviewer.util.runReadyQueue();
        overviewer.util.isReady = true;
    },

    'injectMarkerScript': function(url) {
        var m = document.createElement('script'); m.type = 'text/javascript'; m.async = false;
        m.src = url;
        var s = document.getElementsByTagName('script')[0]; s.parentNode.appendChild(m);
    },

    /** Any polyfills needed to improve browser compatibility
     */
    'initializePolyfills': function() {
        // From https://developer.mozilla.org/en-US/docs/Web/API/ChildNode/remove
        // IE is missing this
        if (!('remove' in Element.prototype)) {
            Element.prototype.remove = function() {
                if (this.parentNode) {
                    this.parentNode.removeChild(this);
                }
            };
        }

    },

    /**
     * onready function for other scripts that rely on overviewer
     * usage: overviewer.util.ready(function(){ // do stuff });
     *
     *
     */
    'ready': function(callback){
        if (typeof callback !== 'function') return;
        if (overviewer.util.isReady){ // run instantly if overviewer already is ready
            overviewer.util.readyQueue.push(callback);
            overviewer.util.runReadyQueue();
        } else {
            overviewer.util.readyQueue.push(callback); // wait until initialize is finished
        }
    },
    'runReadyQueue': function(){
        if(overviewer.util.readyQueue.length === 0) return;
        overviewer.util.readyQueue.forEach(function(callback){
            callback();
        });
        overviewer.util.readyQueue = [];
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
     * Gee, I wonder what this does.
     *
     * @param string msg
     */
    'debug': function(...args) {
        if (overviewerConfig.map.debug) {
            console.log(...args);
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
     * @return array
     */
    'fromWorldToLatLng': function(x, y, z, tset) {

        var zoomLevels = tset.zoomLevels;
        var north_direction = tset.north_direction;

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

        return [-lat*overviewerConfig.CONST.tileSize, lng*overviewerConfig.CONST.tileSize]
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
    'fromLatLngToWorld': function(lat, lng, tset) {
        var zoomLevels = tset.zoomLevels;
        var north_direction = tset.north_direction;

        lat = -lat/overviewerConfig.CONST.tileSize;
        lng = lng/overviewerConfig.CONST.tileSize;

        // lat lng will always be between (0,0) -- top left corner
        //                                (-384, 384) -- bottom right corner

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
    'initHash': function() {
        var newHash = window.location.hash;
        if (overviewer.util.lastHash !== newHash) {
            overviewer.util.lastHash = newHash;
            if(newHash.split("/").length > 1) {
                overviewer.util.goToHash();
                // Clean up the hash.
                overviewer.util.updateHash();
                return true;
            } else {
                return false; // signal to caller that we didn't goto any hash
            }
        }
    },
    'setHash': function(x, y, z, zoom, w, maptype)    {
        // save this info is a nice easy to parse format
        var newHash = "#/" + Math.floor(x) + "/" + Math.floor(y) + "/" + Math.floor(z) + "/" + zoom + "/" + encodeURI(w) + "/" + encodeURI(maptype);
        overviewer.util.lastHash = newHash; // this should not trigger initHash
        window.location.replace(newHash);
    },
    'updateHash': function() {
        // name of current world
        var currWorld = overviewer.current_world;
        if (currWorld == null) {return;}

        var currTileset = overviewer.current_layer[currWorld];
        if (currTileset == null) {return;}

        var ovconf = currTileset.tileSetConfig;

        var coordinates = overviewer.util.fromLatLngToWorld(overviewer.map.getCenter().lat,
                overviewer.map.getCenter().lng,
                ovconf);
        var zoom = overviewer.map.getZoom();

        if (zoom >= ovconf.maxZoom) {
            zoom = 'max';
        } else if (zoom <= ovconf.minZoom) {
            zoom = 'min';
        } else {
            // default to (map-update friendly) negative zooms
            zoom -= ovconf.maxZoom;
        }
        overviewer.util.setHash(coordinates.x, coordinates.y, coordinates.z, zoom, currWorld, ovconf.path);
    },
    'goToHash': function() {
        // Note: the actual data begins at coords[1], coords[0] is empty.
        var coords = window.location.hash.split("/");


        var zoom;
        var world_name = null;
        var tileset_name = null;
        // The if-statements try to prevent unexpected behaviour when using incomplete hashes, e.g. older links
        if (coords.length > 4) {
            zoom = coords[4];
        }
        if (coords.length > 6) {
            world_name = decodeURI(coords[5]);
            tileset_name = decodeURI(coords[6]);
        }

        var target_layer = overviewer.collections.mapTypes[world_name][tileset_name];
        if (!target_layer) {
            var default_tset_name = Object.keys(
                overviewer.collections.mapTypes[world_name])[0];
            target_layer = overviewer.collections.mapTypes[world_name][default_tset_name];
        }
        var ovconf = target_layer.tileSetConfig;

        var latlngcoords = overviewer.util.fromWorldToLatLng(parseInt(coords[1]),
                parseInt(coords[2]),
                parseInt(coords[3]),
                ovconf);

        if (zoom == 'max') {
            zoom = ovconf.maxZoom;
        } else if (zoom == 'min') {
            zoom = ovconf.minZoom;
        } else {
            zoom = parseInt(zoom);
            if (zoom < 0) {
                // if zoom is negative, treat it as a "zoom out from max"
                zoom += ovconf.maxZoom;
            } else {
                // fall back to default zoom
                zoom = ovconf.defaultZoom;
            }
        }

        // clip zoom
        if (zoom > ovconf.maxZoom)
            zoom = ovconf.maxZoom;
        if (zoom < ovconf.minZoom)
            zoom = ovconf.minZoom;

        // build a fake event for the world switcher control
        overviewer.current_layer[world_name] = target_layer;
        overviewer.worldCtrl.onChange({target: {value: world_name}});
        overviewer.worldCtrl.select.value = world_name;

        overviewer.map.setView(latlngcoords, zoom);

        if (ovconf.showlocationmarker) {
            var locationIcon = L.icon({
                iconUrl: overviewerConfig.CONST.image.queryMarker,
                iconRetinaUrl: overviewerConfig.CONST.image.queryMarker2x,
                iconSize: [32, 37],
                iconAnchor: [15, 33],
            });
            var locationm = L.marker(latlngcoords, {  icon: locationIcon,
                                                title: "Linked location"});
            overviewer.collections.locationMarker = locationm
            overviewer.collections.locationMarker.on('contextmenu', function(ev) {
               overviewer.collections.locationMarker.remove();
            });
            overviewer.collections.locationMarker.on('click', function(ev) {
                overviewer.map.setView(ev.latlng);
            });
            overviewer.collections.locationMarker.addTo(overviewer.map);
        }
    },
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
    },
    'isInLayerCtrl': function(ctrl, layer) {
        for (var l in ctrl._layers) {
            if (ctrl._layers[l].layer.tileSetConfig.path == layer.tileSetConfig.path) {
                return true;
            }
        }
        return false;
    }
};
