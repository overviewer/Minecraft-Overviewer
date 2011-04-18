// var def
var map; // god of the overviewer... bow before the google api.
var markerCollection = {}; // holds groups of markers
var markersInit = false; // only have to load the markers once, this just makes sure we only do it once
var regionCollection = {}; // holds groups of regions
var regionsInit = false; // only have to load the regions once, this just makes sure we only do it once

var prevInfoWindow = null;

// add a popup info window to the marker and then the marker to the map.
// marker is the clickable image on the map with all data.
// item is just the same item in the markers.js
function prepareSignMarker(marker, item) {
    var c = "<div class=\"infoWindow\"><img src=\"signpost.png\" /><p>" + item.msg.replace(/\n/g,"<br/>") + "</p></div>";
    var infowindow = new google.maps.InfoWindow({content: c
            });
    google.maps.event.addListener(marker, 'click', function() {
            if (prevInfoWindow)
                prevInfoWindow.close()
            infowindow.open(map,marker);
            prevInfoWindow = infowindow
        });
}

// reusable function for making drop down menus.
// title = string
// items = array 
function createDropDown(title, items) {
    var control = document.createElement("DIV");
    control.id = "customControl"; // let's let a style sheet do most of the styling here

    var controlText = document.createElement("DIV");
    controlText.innerHTML = title;

    var controlBorder = document.createElement("DIV");
    controlBorder.id="top";
    control.appendChild(controlBorder);
    controlBorder.appendChild(controlText);

    var dropdownDiv = document.createElement("DIV");
    dropdownDiv.id="dropDown";
    control.appendChild(dropdownDiv);
    dropdownDiv.innerHTML="";

    // add the functionality to toggle visibility of the items
    $(controlText).click(function() {
            $(dropdownDiv).toggle();
        });
    
    // add that control box we've made back to the map.
    map.controls[google.maps.ControlPosition.TOP_RIGHT].push(control);

    for (idx in items) {
        // create the visible elements of the item
        var item = items[idx];
        //console.log(item); // debug
        var d = document.createElement("div");
        var n = document.createElement("input");
        n.type="checkbox";

        // give it a name
        $(n).data("label",item.label);
        jQuery(n).click((function(local_idx, local_item) {
                             return function(e) {
                                 item.action(local_idx, local_item, e.target.checked);
                             };
                         })(idx, item));

        // if its checked, its gotta do something, do that here.
        if (item.checked) {
            n.checked = true;
            item.action(idx, item.label, item.checked);
        }
        dropdownDiv.appendChild(d);
        d.appendChild(n)
        var textNode = document.createElement("text");
        if(item.icon) {
            textNode.innerHTML = "<img width='15' height='15' src='"+item.icon+"'>" + item.label + "<br/>";
        } else {
            textNode.innerHTML = item.label + "<br/>";
        }

        d.appendChild(textNode);
    }
}

function HomeControl(controlDiv, map) {
 
    controlDiv.style.padding = '5px';

    // Set CSS for the control border
    var control = document.createElement('DIV');
    control.id='top';
    control.title = 'Click to center the map on the Spawn';
    controlDiv.appendChild(control);

    // Set CSS for the control interior
    var controlText = document.createElement('DIV');
    controlText.innerHTML = 'Spawn';
    controlText.id='button';
    control.appendChild(controlText);

    // Setup the click event listeners: simply set the map to map center as definned below
    google.maps.event.addDomListener(control, 'click', function() {
            map.panTo(fromWorldToLatLng(config.center[0],
                                        config.center[1],
                                        config.center[2]));
        });
 
}


// need to define the controls including the compass and layer controls. top right!
// input variables are for chumps... and reusable functions. this is neither.
function drawMapControls() {

    // viewstate link (little link to where you're looking at the map, normally bottom left)
    var viewStateDiv = document.createElement('DIV');
    viewStateDiv.id="link";
    // add it to the map, bottom left.
    map.controls[google.maps.ControlPosition.BOTTOM_LEFT].push(viewStateDiv);

    // compass rose, in the top right corner
    var compassDiv = document.createElement('DIV');
    compassDiv.style.padding = '5px';
    var compassImg = document.createElement('IMG');
    compassImg.src="compass.png";
    compassDiv.appendChild(compassImg);
    compassDiv.index = 0;
    // add it to the map, top right.
    map.controls[google.maps.ControlPosition.TOP_RIGHT].push(compassDiv);
    
    // Spawn button
    var homeControlDiv = document.createElement('DIV');
    var homeControl = new HomeControl(homeControlDiv, map);  
    homeControlDiv.id = "customControl";
    homeControlDiv.index = 1;
    map.controls[google.maps.ControlPosition.TOP_RIGHT].push(homeControlDiv);


    // only need to create the control if there are items in the list. as definned in config.js
    if (signGroups.length > 0) {
        // signpost display control
        
        var items = [];
        for (idx in signGroups) {
            var signGroup = signGroups[idx];
            var iconURL = signGroup.icon;
            if (!iconURL) { iconURL = 'signpost.png'; }
            items.push({
                "label": signGroup.label, 
                "checked": signGroup.checked,
                "icon": iconURL,
                "action": function(n, item, checked) {
                    jQuery.each(markerCollection[item.label], function(i,elem) {
                            elem.setVisible(checked);
                        });
                    //alert(item.label);
                }
            });
        }
        createDropDown("Signposts", items);
    }
    
    // if there are any regions data, lets show the option to hide/show them.
    if (regionGroups.length > 0) {
        // region display control
        
        var items = [];
        for (idx in regionGroups) {
            var regionGroup = regionGroups[idx];
            items.push({
                "label": regionGroup.label, 
                "checked": regionGroup.checked,
                "action": function(n, item, checked) {
                        jQuery.each(regionCollection[item.label], function(i,elem) {
                                elem.setMap(checked ? map : null); // Thanks to LeastWeasel for this line!
                            });
                    }
                });
        }
        createDropDown("Regions", items);
    }
    
    if (overlayMapTypes.length > 0) {
        // overlay maps control
        
        var items = [];
        for (idx in overlayMapTypes) {
            var overlay = overlayMapTypes[idx];
            items.push({"label": overlay.name, "checked": false, "overlay": overlay,
                    "action": function(i, item, checked) {
                        if (checked) {
                            map.overlayMapTypes.push(item.overlay);
                        } else {
                            var idx_to_delete = -1;
                            map.overlayMapTypes.forEach(function(e, j) {
                                    if (e == item.overlay) { idx_to_delete = j; }
                                });
                            if (idx_to_delete >= 0) {
                                map.overlayMapTypes.removeAt(idx_to_delete);
                            }
                        }
                }});
        }
        createDropDown("Overlays", items);
    }
}

// will be recoded by pi, currently always displays all regions all the time.
// parse the data as definned in the regions.js
function initRegions() {
    if (regionsInit) { return; }
    regionsInit = true;
    
    for (i in regionGroups) {
        regionCollection[regionGroups[i].label] = [];
    }

    for (i in regionData) {
        var region = regionData[i];
        
        // pull all the points out of the regions file.
        var converted = new google.maps.MVCArray();
        var infoPoint = "";
        for (j in region.path) {
            var point = region.path[j];
            converted.push(fromWorldToLatLng(point.x, point.y, point.z));
            
        }
        
        for (idx in regionGroups) {
            var regionGroup = regionGroups[idx];
            var testfunc = regionGroup.match;
            var clickable = regionGroup.clickable
            var label = regionGroup.label;
            
            if(region.label) {
                var name = region.label
            } else {
                var name = 'rawr';
                clickable = false; // if it doesn't have a name, we dont have to show it.
            }

            if (region.closed) {
                var shape = new google.maps.Polygon({
                        name: name,
                        clickable: clickable,
                        geodesic: false,
                        map: null,
                        strokeColor: region.color,
                        strokeOpacity: region.opacity,
                        strokeWeight: 2,
                        fillColor: region.color,
                        fillOpacity: region.opacity * 0.25,
                        zIndex: i,
                        paths: converted
                    });
            } else {
                var shape = new google.maps.Polyline({
                        name: name,
                        clickable: clickable,
                        geodesic: false,
                        map: null,
                        strokeColor: region.color,
                        strokeOpacity: region.opacity,
                        strokeWeight: 2,
                        zIndex: i,
                        path: converted
                    });
            }
            regionCollection[label].push(shape); 
            
            if (clickable) {
                // add the region infowindow popup
                infowindow = new google.maps.InfoWindow();
                google.maps.event.addListener(shape, 'click', function(e,i) {
                        
                        var contentString = "<b>Region: "+this.name+"</b><br />";
                        contentString += "Clicked Location: <br />" + e.latLng.lat() + "," + e.latLng.lng() + "<br />";

                        // Replace our Info Window's content and position
                        infowindow.setContent(contentString);
                        infowindow.setPosition(e.latLng);

                        infowindow.open(map);

                    });
            }
        }
    }
}

// will initalize all the markers data as found in markers.js
// may need to be reviewed by agrif or someone else... little finicky right now.
function initMarkers() {
    if (markersInit) { return; } // oh, we've already done this? nevermind, exit the function.
    markersInit = true; // now that we've started, dont have to do it twice.
    
    // first, give all collections an empty array to work with
    for (i in signGroups) {
        markerCollection[signGroups[i].label] = [];
    }
        
    
    for (i in markerData) {
        var item = markerData[i];

        // a default:
        var iconURL = '';
        if (item.type == 'spawn') { 
            // don't filter spawn, always display

            iconURL = 'http://google-maps-icons.googlecode.com/files/home.png';
            var converted = fromWorldToLatLng(item.x, item.y, item.z);
            var marker = new google.maps.Marker({position: converted,
                    map: map,
                    title: jQuery.trim(item.msg), 
                    icon: iconURL
                });
            continue;
        }

        if (item.type == 'querypos') { 
            // Set on page load if MC x/y/z coords are given in the query string

            iconURL = 'http://google-maps-icons.googlecode.com/files/regroup.png';
            var converted = fromWorldToLatLng(item.x, item.y, item.z);
            var marker = new google.maps.Marker({position: converted,
                    map: map,
                    title: jQuery.trim(item.msg), 
                    icon: iconURL
                    });

            continue;
        }

        var matched = false;
        for (idx in signGroups) {
            var signGroup = signGroups[idx];
            var testfunc = signGroup.match;
            var label = signGroup.label;

            if (testfunc(item)) {
                matched = true;

                // can add custom types of images for externally definned item types, like 'command' here.
                if (item.type == 'sign') { iconURL = 'signpost_icon.png'; }

                //console.log(signGroup.icon); //debug
                if (signGroup.icon) { iconURL = signGroup.icon; }

                var converted = fromWorldToLatLng(item.x, item.y, item.z);
                var marker = new google.maps.Marker({position: converted,
                        map: map,
                        title: jQuery.trim(item.msg), 
                        icon: iconURL,
                        visible: false
                    });
                
                markerCollection[label].push(marker);

                if (item.type == 'sign') {
                    prepareSignMarker(marker, item);
                }
            }
        }
        
        if (!matched) {
            // is this signpost doesn't match any of the groups in config.js, add it automatically to the "__others__" group
            if (item.type == 'sign') { iconURL = 'signpost_icon.png';}

            var converted = fromWorldToLatLng(item.x, item.y, item.z);
            var marker = new google.maps.Marker({position: converted,
                    map: map,
                    title: jQuery.trim(item.msg), 
                    icon: iconURL,
                    visible: false
                    });
            if (markerCollection["__others__"]) {
                markerCollection["__others__"].push(marker);
            } else {
                markerCollection["__others__"] = [marker];
            }

            if (item.type == 'sign') {
                prepareSignMarker(marker, item);
            }
        }
        


    }
}

// update the link in the viewstate. 
function makeLink() {
    var displayZoom = map.getZoom();
    if (displayZoom == config.maxZoom) {
        displayZoom = "max";
    } else {
        displayZoom -= config.maxZoom;
    }
    var xyz;
    var xyz = fromLatLngToWorld(map.getCenter().lat(), map.getCenter().lng());
    var a=location.href.substring(0,location.href.lastIndexOf(location.search))
        + "?x=" + Math.floor(xyz.x)
        + "&y=" + Math.floor(xyz.y)
        + "&z=" + Math.floor(xyz.z)
        + "&zoom=" + displayZoom;
    document.getElementById("link").innerHTML = a;
}

// load the map up and add all the functions relevant stuff to the map.
function initialize() {

    var query = location.search.substring(1);

    var defaultCenter = fromWorldToLatLng(config.center[0],
                                          config.center[1],
                                          config.center[2]);
    var lat = defaultCenter.lat();
    var lng = defaultCenter.lng();
    
    var zoom = config.defaultZoom;
    var hasquerypos = false;
    var queryx = 0;
    var queryy = 64;
    var queryz = 0;
    var mapcenter;
    var pairs = query.split("&");
    for (var i=0; i<pairs.length; i++) {
        // break each pair at the first "=" to obtain the argname and value
        var pos = pairs[i].indexOf("=");
        var argname = pairs[i].substring(0,pos).toLowerCase();
        var value = pairs[i].substring(pos+1).toLowerCase();

        // process each possible argname
        if (argname == "lat") {lat = parseFloat(value);}
        if (argname == "lng") {lng = parseFloat(value);}
        if (argname == "zoom") {
            if (value == "max") {
                zoom = config.maxZoom;
            } else {
                zoom = parseInt(value);
                // If negative, treat as a "zoom out from max zoom" value
                if (zoom < 0) {zoom = config.maxZoom + zoom;}
                // If still negative, fall back to default zoom
                if (zoom < 0) {zoom = config.defaultZoom;}
            }
        }
        if (argname == "x") {queryx = parseFloat(value); hasquerypos = true;}
        if (argname == "y") {queryy = parseFloat(value); hasquerypos = true;}
        if (argname == "z") {queryz = parseFloat(value); hasquerypos = true;}
    }

    if (hasquerypos) {
        mapcenter = fromWorldToLatLng(queryx, queryy, queryz);
        // Add a market indicating the user-supplied position
        markerData.push({"msg": "Coordinates " + queryx + ", " + queryy + ", " + queryz, "y": queryy, "x": queryx, "z": queryz, "type": "querypos"})
    } else {
        mapcenter = new google.maps.LatLng(lat, lng);
    }

    var mapTyepControlToggle = false
    if (mapTypeIds.length > 1) {
      mapTyepControlToggle = true
    }
    var mapOptions = {
        zoom: zoom,
        center: mapcenter,
        navigationControl: true,
        scaleControl: false,
        mapTypeControl: mapTyepControlToggle,
        mapTypeControlOptions: {
            mapTypeIds: mapTypeIds
        },
        mapTypeId: mapTypeIdDefault,
        streetViewControl: false,
        backgroundColor: config.bg_color,
    };
    map = new google.maps.Map(document.getElementById('mcmap'), mapOptions);

    if(config.debug) {
        map.overlayMapTypes.insertAt(0, new CoordMapType(new google.maps.Size(config.tileSize, config.tileSize)));

        google.maps.event.addListener(map, 'click', function(event) {
                //console.log("latLng; " + event.latLng.lat() + ", " + event.latLng.lng());

                var pnt = map.getProjection().fromLatLngToPoint(event.latLng);
                //console.log("point: " + pnt);

                var pxx = pnt.x * config.tileSize * Math.pow(2, config.maxZoom);
                var pxy = pnt.y * config.tileSize * Math.pow(2, config.maxZoom);
                //console.log("pixel: " + pxx + ", " + pxy);
                });
    }

    // Now attach the coordinate map type to the map's registry
    for (idx in MCMapType) {
      map.mapTypes.set('mcmap' + MCMapType[idx].name, MCMapType[idx]);
    }
    
    // We can now set the map to use the 'coordinate' map type
    map.setMapTypeId(mapTypeIdDefault);

    // initialize the markers and regions
    initMarkers();
    initRegions();
    drawMapControls();

    //makeLink();

    // Make the link again whenever the map changes
    google.maps.event.addListener(map, 'zoom_changed', function() {
        makeLink();
    });
    google.maps.event.addListener(map, 'center_changed', function() {
        makeLink();
    });

}


// our custom projection maps Latitude to Y, and Longitude to X as normal,
// but it maps the range [0.0, 1.0] to [0, tileSize] in both directions
// so it is easier to position markers, etc. based on their position
// (find their position in the lowest-zoom image, and divide by tileSize)
function MCMapProjection() {
    this.inverseTileSize = 1.0 / config.tileSize;
}
  
MCMapProjection.prototype.fromLatLngToPoint = function(latLng) {
    var x = latLng.lng() * config.tileSize;
    var y = latLng.lat() * config.tileSize;
    return new google.maps.Point(x, y);
};

MCMapProjection.prototype.fromPointToLatLng = function(point) {
    var lng = point.x * this.inverseTileSize;
    var lat = point.y * this.inverseTileSize;
    return new google.maps.LatLng(lat, lng);
};
  
// helper to get map LatLng from world coordinates
// takes arguments in X, Y, Z order
// (arguments are *out of order*, because within the function we use
// the axes like the rest of Minecraft Overviewer -- with the Z and Y
// flipped from normal minecraft usage.)
function fromWorldToLatLng(x, z, y)
{
    // the width and height of all the highest-zoom tiles combined, inverted
    var perPixel = 1.0 / (config.tileSize * Math.pow(2, config.maxZoom));

    // This information about where the center column is may change with a different
    // drawing implementation -- check it again after any drawing overhauls!

    // point (0, 0, 127) is at (0.5, 0.0) of tile (tiles/2 - 1, tiles/2)
    // so the Y coordinate is at 0.5, and the X is at 0.5 - ((tileSize / 2) / (tileSize * 2^maxZoom))
    // or equivalently, 0.5 - (1 / 2^(maxZoom + 1))
    var lng = 0.5 - (1.0 / Math.pow(2, config.maxZoom + 1));
    var lat = 0.5;

    // the following metrics mimic those in ChunkRenderer.chunk_render in "chunk.py"
    // or, equivalently, chunk_render in src/iterate.c

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
}

// NOTE: X, Y and Z in this function are Minecraft world definitions
// (that is, X is horizontal, Y is altitude and Z is vertical).
function fromLatLngToWorld(lat, lng)
{
    // Initialize world x/y/z object to be returned
    var xyz = Array();
    xyz.x = 0;
    xyz.y = 64;
    xyz.z = 0;

    // the width and height of all the highest-zoom tiles combined, inverted
    var perPixel = 1.0 / (config.tileSize * Math.pow(2, config.maxZoom));

    // Revert base positioning
    // See equivalent code in fromWorldToLatLng()
    lng -= 0.5 - (1.0 / Math.pow(2, config.maxZoom + 1));
    lat -= 0.5;

    // I'll admit, I plugged this into Wolfram Alpha:
    //   a = (x * 12 * r) + (z * 12 * r), b = (z * 6 * r) - (x * 6 * r)
    // And I don't know the math behind solving for for X and Z given
    // A (lng) and B (lat).  But Wolfram Alpha did. :)  I'd welcome
    // suggestions for splitting this up into long form and documenting
    // it. -RF
    xyz.x = (lng - 2 * lat) / (24 * perPixel)
    xyz.z = (lng + 2 * lat) / (24 * perPixel)

    // Adjust for the fact that we we can't figure out what Y is given
    // only latitude and longitude, so assume Y=64.
    xyz.x += 64 + 1;
    xyz.z -= 64 + 2;
    
    return xyz;
}

function getTileUrlGenerator(path, path_base, path_ext) {
    return function(tile, zoom) {
        var url = path;
        var url_base = ( path_base ? path_base : '' );
        if(tile.x < 0 || tile.x >= Math.pow(2, zoom) || tile.y < 0 || tile.y >= Math.pow(2, zoom)) {
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
        url = url + '.' + path_ext;
        if(config.cacheMinutes > 0) {
            var d = new Date();
            url += '?c=' + Math.floor(d.getTime() / (1000 * 60 * config.cacheMinutes));
        }
        return(url_base + url);
    }
}

var MCMapOptions = new Array;
var MCMapType = new Array;
var mapTypeIdDefault = null;
var mapTypeIds = [];
var overlayMapTypes = [];
for (idx in mapTypeData) {
    var view = mapTypeData[idx];
    var imgformat = view.imgformat ? view.imgformat : 'png';

    MCMapOptions[view.label] = {
        getTileUrl: getTileUrlGenerator(view.path, view.base, imgformat),
        tileSize: new google.maps.Size(config.tileSize, config.tileSize),
        maxZoom:  config.maxZoom,
        minZoom:  0,
        isPng:    !(imgformat.match(/^png$/i) == null)
    };
  
    MCMapType[view.label] = new google.maps.ImageMapType(MCMapOptions[view.label]);
    MCMapType[view.label].name = view.label;
    MCMapType[view.label].alt = "Minecraft " + view.label + " Map";
    MCMapType[view.label].projection = new MCMapProjection();

    if (view.overlay) {
        overlayMapTypes.push(MCMapType[view.label]);
    } else {
    if (mapTypeIdDefault == null) {
        mapTypeIdDefault = 'mcmap' + view.label;
    }
    mapTypeIds.push('mcmap' + view.label);
  }
}
  
function CoordMapType() {
}
  
function CoordMapType(tileSize) {
    this.tileSize = tileSize;
}
  
CoordMapType.prototype.getTile = function(coord, zoom, ownerDocument) {
    var div = ownerDocument.createElement('DIV');
    div.innerHTML = "(" + coord.x + ", " + coord.y + ", " + zoom + ")";
    div.innerHTML += "<br />";
    div.innerHTML += MCMapOptions.getTileUrl(coord, zoom);
    div.style.width = this.tileSize.width + 'px';
    div.style.height = this.tileSize.height + 'px';
    div.style.fontSize = '10';
    div.style.borderStyle = 'solid';
    div.style.borderWidth = '1px';
    div.style.borderColor = '#AAAAAA';
    return div;
};
