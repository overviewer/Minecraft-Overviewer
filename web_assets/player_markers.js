var playerMarkers = null;
var warpMarkers = [];
var PlayerNames = null;
var PlayerCount = 0;

function deletePlayerMarkers() {
  if (playerMarkers) {
    for (i in playerMarkers) {
      playerMarkers[i].setMap(null);
    }
    playerMarkers = null;
	PlayerNames = null;
	PlayerCount = 0;
  }
}

setInterval(loadPlayerMarkers, 1000 * 5);
setTimeout(loadPlayerMarkers, 1000);

function preparePlayerMarker(marker,item) {
	var c = "<div class=\"infoWindow\" style='width: 300px'><img src='player-avatar.php?player="+item.msg+"&s=3&bc=000&bw=1&format=flat'/><h1>"+item.msg+"</h1></div>";
	var infowindow = new google.maps.InfoWindow({content: c});
	google.maps.event.addListener(marker, 'click', function() {
		infowindow.open(map,marker);
	});
}

function loadPlayerMarkers() {
    $.getJSON('markers.json', function(data) {
        deletePlayerMarkers();
        playerMarkers = [];
		PlayerNames = [];
		PlayerCount = 0;

        for (i in data) {
            var item = data[i];
            var converted = fromWorldToLatLng(item.x, item.y, item.z);
			
			var perPixel = 1.0 / (config.tileSize * Math.pow(2, config.maxZoom));

			var lng = 0.5 - (1.0 / Math.pow(2, config.maxZoom + 1));
			var lat = 0.5;
					
			lng += 12 * item.x * perPixel;
			lat -= 6 * item.x * perPixel;
					
			lng += 12 * item.z * perPixel;
			lat += 6 * item.z * perPixel;
					
			lat += 12 * (128 - item.y) * perPixel;

			lng += 12 * perPixel;
			lat += 18 * perPixel;
			
			PlayerNames.push('<!-- ' + item.msg.toLowerCase() + ' -->&nbsp;<a href="index.html?lat=' + lat +'&lng=' + lng + '&zoom=6" target="_parent" class="playerlink"><img src="./head.php?player=' + item.msg + '&usage=list" border="0" />&nbsp;' + item.msg + '</a><br /> ');
			PlayerCount++;
            
			var marker = new google.maps.Marker({
                    position: converted,
                    map: map,
                    title: item.msg,
                    icon: "player-avatar.php?player="+item.msg+"&s=1&bc=fff&bw=1&format=flat",
					visible: true,
					zIndex: 999
            });
			playerMarkers.push(marker);
			preparePlayerMarker(marker, item);
         }
		
		$("#Players").empty();

		if(PlayerCount == 0)
		{
			$("#Players").html('&nbsp;<a href="index.html" target="_parent"><img src="./home-list.png" border="0" /></a>&nbsp;<font color="lightgreen">' + PlayerCount + '</font> players online');
		}
		else
		{
			PlayerNames.sort();
			
			$("#Players").html('&nbsp;<a href="index.html" target="_parent"><img src="./home-list.png" border="0" /></a>&nbsp;<font color="lightgreen">' + PlayerCount + '</font> Players online:<br /><br />' + PlayerNames.join(" "));
		}
	});
}

loadPlayerMarkers();
