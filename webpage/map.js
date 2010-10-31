
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
  
  var MCMapOptionsUnlit = {
    getTileUrl: function(tile, zoom) {
      var url = config.path+"/unlit";
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
  
   var MCMapOptionsDay = {
    getTileUrl: function(tile, zoom) {
      var url = config.path+"/day";
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
  
   var MCMapOptionsNight= {
    getTileUrl: function(tile, zoom) {
      var url = config.path+"/night";
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
  
  var MCMapTypeUnlit = new google.maps.ImageMapType(MCMapOptionsUnlit);
  MCMapTypeUnlit.name = "Unlit";
  MCMapTypeUnlit.alt = "Minecraft Map";
  MCMapTypeUnlit.projection = new MCMapProjection();
  
  var MCMapTypeDay = new google.maps.ImageMapType(MCMapOptionsDay);
  MCMapTypeDay.name = "Day";
  MCMapTypeDay.alt = "Minecraft Map";
  MCMapTypeDay.projection = new MCMapProjection();
  
  var MCMapTypeNight = new google.maps.ImageMapType(MCMapOptionsNight);
  MCMapTypeNight.name = "Night";
  MCMapTypeNight.alt = "Minecraft Map";
  MCMapTypeNight.projection = new MCMapProjection();
  
  function CoordMapType() {
  }
  
  function CoordMapType(tileSize) {
    this.tileSize = tileSize;
  }
  
  CoordMapType.prototype.getTile = function(coord, zoom, ownerDocument) {
    var div = ownerDocument.createElement('DIV');
    div.innerHTML = "(" + coord.x + ", " + coord.y + ", " + zoom + ")";
    div.innerHTML += "<br />";
    div.innerHTML += MCMapOptionsUnlit.getTileUrl(coord, zoom);
    div.style.width = this.tileSize.width + 'px';
    div.style.height = this.tileSize.height + 'px';
    div.style.fontSize = '10';
    div.style.borderStyle = 'solid';
    div.style.borderWidth = '1px';
    div.style.borderColor = '#AAAAAA';
    return div;
  };
  
  function makeLink() {
    var a=location.href.substring(0,location.href.lastIndexOf("/")+1)
        + "?lat=" + map.getCenter().lat().toFixed(6)
        + "&lng=" + map.getCenter().lng().toFixed(6)
        + "&zoom=" + map.getZoom();
    document.getElementById("link").innerHTML = a;
}
  
  var map;
  
  
  function initialize() {
      var query = location.search.substring(1);

    var lat = 0.5;
    var lng = 0.5;
    var zoom = config.defaultZoom;
    var pairs = query.split("&");
    for (var i=0; i<pairs.length; i++) {
        // break each pair at the first "=" to obtain the argname and value
        var pos = pairs[i].indexOf("=");
        var argname = pairs[i].substring(0,pos).toLowerCase();
        var value = pairs[i].substring(pos+1).toLowerCase();

        // process each possible argname
        if (argname == "lat") {lat = parseFloat(value);}
        if (argname == "lng") {lng = parseFloat(value);}
        if (argname == "zoom") {zoom = parseInt(value);}
    }
      
    var mapOptions = {
      zoom: config.defaultZoom,
      navigationControl: true,
      scaleControl: false,
      mapTypeControl: true,
      mapTypeControlOptions: {
          mapTypeIds: [ 'mcmapunlit','mcmapday','mcmapnight' ]
        },
      streetViewControl: false,
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
    
     
    // Now attach the seperate lighting map types to the map's registry
    map.mapTypes.set('mcmapunlit', MCMapTypeUnlit);
    map.mapTypes.set('mcmapday', MCMapTypeDay);
    map.mapTypes.set('mcmapnight', MCMapTypeNight);
    
    // We can now set the map to use the 'unlit' map type
    map.setMapTypeId('mcmapunlit');
	
    // Set the mapCenter (used as spawn
    
    map.setCenter(config.mapCenter);
    
     makeLink();

    // Make the link again whenever the map changes
    google.maps.event.addListener(map, 'zoom_changed', function() {
        makeLink();
    });
    google.maps.event.addListener(map, 'center_changed', function() {
        makeLink();
    });
	
  }
