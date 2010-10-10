  var config = {
    path:        'tiles',
    fileExt:     '{imgformat}',
    tileSize:     384,
    defaultZoom:  {defaultzoom},
    markerZoom:   {markerzoom},
    maxZoom:      {maxzoom},
    cacheMinutes: 0, // Change this to have browsers automatically requiest new images every x minutes
    debug:        false
  };



  function imgError(source){
	source.src = "http://maps.gstatic.com/intl/en_us/mapfiles/transparent.png";
	source.onerror = "";
	return true;
}
  
  var markers = new Array();
  var tiles = new Object();

        var urlParams = {};

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

    // each block on X axis adds 12px to x and subtracts 6px from y
    lng += 12 * x * perPixel;
    lat -= 6 * x * perPixel;

    // each block on Y axis adds 12px to x and adds 6px to y
    lng += 12 * y * perPixel;
    lat += 6 * y * perPixel;

    // each block down along Z adds 12px to y
    lat += 12 * (128 - z) * perPixel;

    // add on 12 px to the X coordinate and 18px to the Y to center our point
    lng += 12 * perPixel;
    lat += 18 * perPixel;

    return new google.maps.LatLng(lat, lng);
  }

  function updateTiles() {

	for(tile in tiles) {
		tiles[tile].div.firstChild.src = mcMapType.getTileUrl(tiles[tile].coord, tiles[tile].zoom)+"?" + (new Date).getTime();
		tiles[tile].div.firstChild.onerror = 'imgError(this)';
		tiles[tile].div.firstChild.style.border = '1px solid red';
		setTimeout( function() {
			for(tile in tiles) {
				tiles[tile].div.firstChild.style.border = '0px';
			} }, 500);
		}
  }
  
  function MCMapType() {
}

MCMapType.prototype.tileSize = new google.maps.Size(config.tileSize, config.tileSize);
MCMapType.prototype.maxZoom = config.maxZoom;
MCMapType.prototype.minZoom = 0;
MCMapType.prototype.isPng =  !(config.fileExt.match(/^png$/i) == null);

MCMapType.prototype.getTile = function(coord, zoom, ownerDocument) {
    var div = ownerDocument.createElement('DIV');
    //div.innerHTML = "(" + coord.x + ", " + coord.y + ", " + zoom + ")";
    //div.innerHTML += "<br />";
    //div.innerHTML += mcMapType.getTileUrl(coord, zoom);

	div.innerHTML += "<img src='"+mcMapType.getTileUrl(coord, zoom)+"' onerror='imgError(this)' />";
	
	//src="http://maps.gstatic.com/intl/en_us/mapfiles/transparent.png"
	//div.innerHTML += "<img src='test' />";
    div.style.width = this.tileSize.width + 'px';
    div.style.height = this.tileSize.height + 'px';
    //div.style.fontSize = '10';
    //div.style.borderStyle = 'solid';
    //div.style.borderWidth = '1px';
    //div.style.borderColor = '#AAAAAA';
	// Add coodrs to live list, so when reload is hit, it gets reloaded
	div.tileId = "" + coord.x+","+coord.y+","+zoom
	tiles[div.tileId] = {coord : coord, zoom : zoom, div : div};
	
    return div;
	}

MCMapType.prototype.releaseTile = function(tile) {
	// Remove coodrs from live list
	
	delete tiles[tile.tileId];
	
    }
	
MCMapType.prototype.getTileUrl = function(tile, zoom) {
      var url = config.path;
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
      url = url + '.' + config.fileExt;
      if(config.cacheMinutes > 0) {
        var d = new Date();
        url += '?c=' + Math.floor(d.getTime() / (1000 * 60 * config.cacheMinutes));
      }
      return(url);
};


MCMapType.prototype.name = "MC Map";
MCMapType.prototype.alt = "Minecraft Map";
MCMapType.prototype.projection = new MCMapProjection();

var map;
var mcMapType = new MCMapType();
  
/**
 * The HomeControl adds a control to the map that simply
 * returns the user to Chicago. This constructor takes
 * the control DIV as an argument.
 */

function ReloadControl(controlDiv, map) {

  // Set CSS styles for the DIV containing the control
  // Setting padding to 5 px will offset the control
  // from the edge of the map
  controlDiv.style.padding = '5px';

  // Set CSS for the control border
  var controlUI = document.createElement('DIV');
  controlUI.style.backgroundColor = 'white';
  controlUI.style.borderStyle = 'solid';
  controlUI.style.borderWidth = '2px';
  controlUI.style.cursor = 'pointer';
  controlUI.style.textAlign = 'center';
  controlUI.title = 'Click to reload the images in the current viewport';
  controlDiv.appendChild(controlUI);

  // Set CSS for the control interior
  var controlText = document.createElement('DIV');
  controlText.style.fontFamily = 'Arial,sans-serif';
  controlText.style.fontSize = '12px';
  controlText.style.paddingLeft = '4px';
  controlText.style.paddingRight = '4px';
  controlText.innerHTML = 'Reload';
  controlUI.appendChild(controlText);

  // Setup the click event listeners: simply set the map to Chicago
  google.maps.event.addDomListener(controlUI, 'click', function() {
    updateTiles();
  });
}
  
  var markersInit = false;

  function initMarkers() {
    if (markersInit) { return; }

    markersInit = true;

    if (markerData == null) return;
    for (i in markerData) {
      addMarker(markerData[i]);

   }
  }


  function delMarker(markername) {
        marker = markers[markername];

    if (marker) {
                marker.setMap(null);
                markers[markername].title = "";
                $('#mcmarkerlist div[name='+markername+']').remove();
        }
   }


  function addMarker(item) {
        // Add marker if it doesnt exist
        // if it does, update position
		
		if ("no_markers" in urlParams)
			return;

        if ("no_players" in urlParams && item.id == 4)
                return;

        if ("no_places" in urlParams && item.id == 3)
                return;

        var converted = fromWorldToLatLng(item.x, item.y, item.z);
        marker = markers[item.msg+item.id];
        if (marker) {
                marker.setPosition(converted);
        }
        else {

       var marker = new google.maps.Marker({
        position: converted,
        map: map,
        title: item.msg

		});
        markers[item.msg+item.id] = marker;
		  
		$('#mcmarkerlist div[name=mcmarkers'+item.id+']').append('<div class="mcmarker" name="'+item.msg+item.id+'">'+item.msg+'</div>');

        $('#mcmarkerlist div[name=mcmarkers'+item.id+'] div[name="'+item.msg+item.id+'"]').click(function() {
				map.panTo(markers[$(this).attr("name")].getPosition());
                map.setZoom(config.markerZoom);
			});
          }
   }


  function refreshMarkers(){
                $.getJSON('markers.json', function(data) {

                        if (data == null)
                                return;

                        for (marker in markers) {
                                var found = false;
                                for (item in data) {
                                        if (marker == data[item].msg + data[item].id)
                                                found = true;

                                }
                                if (!found)
                                        delMarker(marker);
                        }

                        for (item in data) {
                                addMarker(data[item]);
                        }




                });

        }



  function initialize() {
  
    var mapOptions = {
      zoom: config.defaultZoom,
      center: new google.maps.LatLng(0.5, 0.5),
      navigationControl: true,
      scaleControl: true,
      mapTypeControl: false,
      mapTypeId: 'mcmap'
    };
    map = new google.maps.Map(document.getElementById("mcmap"), mapOptions);
	//map = new google.maps.Map(document.getElementById("mcmap"), mapOptions);

    if(config.debug) {
      map.overlayMapTypes.insertAt(0, new CoordMapType(new google.maps.Size(config.tileSize, config.tileSize)));

          google.maps.event.addListener(map, 'click', function(event) {
            console.log("latLng; " + event.latLng.lat() + ", " + event.latLng.lng());

            var pnt = map.getProjection().fromLatLngToPoint(event.latLng);
            console.log("point: " + pnt);

            var pxx = pnt.x * config.tileSize * Math.pow(2, config.maxZoom);
            var pxy = pnt.y * config.tileSize * Math.pow(2, config.maxZoom);
            console.log("pixel: " + pxx + ", " + pxy);
          });
    }

    // Now attach the coordinate map type to the map's registry
    map.mapTypes.set('mcmap', mcMapType);

    // We can now set the map to use the 'coordinate' map type
    map.setMapTypeId('mcmap');

	// initialize the markers
	initMarkers();


	//var refreshInterval = setInterval(refreshMarkers, 3 * 1000);
	refreshMarkers();

	// Set initial position to spawn
	setTimeout(map.panTo(markers["Spawn0"].getPosition()),2000);
	
	// Create the DIV to hold the control and call the HomeControl() constructor
	// passing in this DIV.
	var reloadControlDiv = document.createElement('DIV');
	var reloadControl = new ReloadControl(reloadControlDiv, map);

	reloadControl.index = 1;
	map.controls[google.maps.ControlPosition.TOP_RIGHT].push(reloadControlDiv);
  }


$(document).ready(function() {

        (function () {
            var e,
                d = function (s) { return decodeURIComponent(s.replace(/\+/g, " ")); },
                q = window.location.search.substring(1),
                r = /([^&=]+)=?([^&]*)/g;

            while (e = r.exec(q))
               urlParams[d(e[1])] = d(e[2]);
        })();

        if ("no_overlay" in urlParams)
                $('#mcmarkerlist').hide();
        if ("no_markers" in urlParams)
                $('#mcmarkerlist').hide();
        if ("no_places" in urlParams)
                $('div[name=mcmarkers3]').hide();
        if ("no_players" in urlParams)
                $('div[name=mcmarkers4]').hide();
        initialize();
});


