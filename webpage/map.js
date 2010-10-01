  var config = {
    path:        'tiles',
    fileExt:     'png',
    tileSize:     384,
    defaultZoom:  {defaultzoom},
    markerZoom:   {markerzoom},
    maxZoom:      {maxzoom},
    cacheMinutes: 0, // Change this to have browsers automatically requiest new images every x minutes
    debug:        false
  };

  var markers = new Array();

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

  var MCMapOptions = {
    getTileUrl: function(tile, zoom) {
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
    },
    tileSize: new google.maps.Size(config.tileSize, config.tileSize),
    maxZoom:  config.maxZoom,
    minZoom:  0,
    isPng:    !(config.fileExt.match(/^png$/i) == null)
  };

  var MCMapType = new google.maps.ImageMapType(MCMapOptions);
  MCMapType.name = "MC Map";
  MCMapType.alt = "Minecraft Map";
  MCMapType.projection = new MCMapProjection();

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

  var map;

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
      scaleControl: false,
      mapTypeControl: false,
      mapTypeId: 'mcmap'
    };
    map = new google.maps.Map(document.getElementById("mcmap"), mapOptions);

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
    map.mapTypes.set('mcmap', MCMapType);

    // We can now set the map to use the 'coordinate' map type
    map.setMapTypeId('mcmap');

        // initialize the markers
        initMarkers();


        var refreshInterval = setInterval(refreshMarkers, 3 * 1000);
        refreshMarkers();

        // Set initial position to spawn
        map.panTo(markers["Spawn0"].getPosition());
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

