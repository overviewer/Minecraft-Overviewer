// Global variables
var playerMarkers = new Array();
var urlParams = {};
var regionsInit = false;
  
var reg = /(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})/; // !TODO!need to keep synced with mapmarkers format

  
function initRegions() {
	if (regionsInit) { return; }

	regionsInit = true;

	for (i in regionData) {
		var region = regionData[i];
		var converted = new google.maps.MVCArray();
		for (j in region.path) {
			var point = region.path[j];
			converted.push(fromWorldToLatLng(point.x, point.y, point.z));
		}

		if (region.closed) {
			new google.maps.Polygon({
				clickable: false,
				geodesic: false,
				map: map,
				strokeColor: region.color,
				strokeOpacity: region.opacity,
				strokeWeight: 2,
				fillColor: region.color,
				fillOpacity: region.opacity * 0.25,
				zIndex: i,
				paths: converted
			});
		} else {
			new google.maps.Polyline({
				clickable: false,
				geodesic: false,
				map: map,
				strokeColor: region.color,
				strokeOpacity: region.opacity,
				strokeWeight: 2,
				zIndex: i,
				path: converted
			});
		}
	}
}
		
function gotoPlayer(index)
{
	map.setCenter(playerMarkers[index].position);
	map.setZoom(6);
}

  function delMarker(markername) {
        marker = playerMarkers[markername];

    if (marker) {
                marker.setMap(null);
                delete playerMarkers[markername];
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
				
		m = reg.exec(item.timestamp),
		ts = new Date(m[1],m[2]-1,m[3],m[4],m[5],m[6]),
		d = new Date(),
		diff = d.getTime() - ts.getTime(),
		var converted = fromWorldToLatLng(item.x, item.y, item.z);
		marker = playerMarkers[item.msg+item.id];
		
		if (marker) {
			marker.setPosition(converted);
		}
		else {
			if( diff < 10 * 1000*60 ) {
			
				var marker = new google.maps.Marker({
					position: converted,
					map: map,
					title: item.msg,
					icon: 'User.png'
				});
				$('#plist').append("<span onClick='gotoPlayer(" + i + ")'>" + item.msg + "</span><br />");
				playerMarkers[item.msg+item.id] = marker;
			}
			else {
				var marker = new google.maps.Marker({
				position: converted,
				map: map,
				title: item.msg + " - Idle since " + ts.toString(),,
				icon: 'User.png'
			});
			$('#plist').append("<span onClick='gotoPlayer(" + i + ")' class='idle'>" + item.msg + "</span><br />");
			playerMarkers[item.msg+item.id] = marker;
			}
		}
		
		
   }


  function refreshMarkers(){
                $.getJSON('markers.json', function(data) {

                        if (data == null || data.length == 0) {
							$('#plist').html('[No players online]');
                            return;
						}

						for (i in data) {
							var item = data[i],
							
						}
		
                        for (marker in playerMarkers) {
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



  function mapMarkersInit() {
    // initRegions(); //!TODO!Get MapRegions to write regions.json from cuboids
	

	var refreshInterval = setInterval(refreshMarkers, 3 * 1000);
	refreshMarkers();
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
        mapMarkersInit();
});
