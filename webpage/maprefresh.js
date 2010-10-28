// Global variables
var playerMarkers = new Array();
var urlParams = {};
var regionsInit = false;
  
var reg = /(\d{4})(\d{2})(\d{2}) (\d{2}):(\d{2}):(\d{2})/; // !TODO!need to keep synced with mapmarkers format

  
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
	map.setZoom(config.markerZoom);
}

  function delMarker(markername) {
        marker = playerMarkers[markername];

    if (marker) {
                marker.setVisible(false);
                //delete playerMarkers[markername];
                $('#plist span[name='+markername+']').remove();
				$('#plist br[name='+markername+']').remove();
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
				
		var m = reg.exec(item.timestamp),
		ts = new Date(m[1],m[2]-1,m[3],m[4],m[5],m[6]),
		d = new Date(),
		diff = d.getTime() - ts.getTime(),
		converted = fromWorldToLatLng(item.x, item.y, item.z);
		marker = playerMarkers[item.msg+item.id];
		
		if (marker) {
			if (!marker.getVisible()) {
				marker.setVisible(true);
				if( diff < 10 * 1000*60 ) {
					$('#plist').append("<span name='" + item.msg+item.id + "' onClick='gotoPlayer(\"" + item.msg+item.id + "\")'>" + item.msg + "</span><br name='" + item.msg+item.id + "' />");
				}
				else {
					$('#plist').append("<span name='" + item.msg+item.id + "' onClick='gotoPlayer(\"" + item.msg+item.id + "\")' class='idle'>" + item.msg + "</span><br name='" + item.msg+item.id + "' />");
				}
			}
			marker.setPosition(converted);
		}
		else {
			if( diff < 10 * 1000*60 ) {
			
				var marker = new google.maps.Marker({
					position: converted,
					map: map,
					title: item.msg,
					icon: 'smiley.gif'
				});
				$('#plist').append("<span name='" + item.msg+item.id + "' onClick='gotoPlayer(\"" + item.msg+item.id + "\")'>" + item.msg + "</span><br name='" + item.msg+item.id + "' />");
				playerMarkers[item.msg+item.id] = marker;
			}
			else {
				var marker = new google.maps.Marker({
				position: converted,
				map: map,
				title: item.msg + " - Idle since " + ts.toString(),
				icon: 'smiley.gif'
			});
			$('#plist').append("<span name='" + item.msg+item.id + "' onClick='gotoPlayer(\"" + item.msg+item.id + "\")' class='idle'>" + item.msg + "</span><br name='" + item.msg+item.id + "' />");
			playerMarkers[item.msg+item.id] = marker;
			}
		}
		
		
   }


  function refreshMarkers(){
                $.getJSON('markers.json', function(data) {
					try {
                        if (data == null || data.length == 0) {
							$('#plist').html('[No players online]');
                            return;
						}
		
                        for (marker in playerMarkers) {
                                var found = false;
                                for (item in data) {
                                        if (marker == data[item].msg + data[item].id)
                                                found = true;

                                }
                                if (!found && playerMarkers[marker].getVisible())
                                        delMarker(marker);
                        }

                        for (item in data) {
							if (data[item].id == 4)
                                addMarker(data[item]); // Only player markers, ignore other for now
                        }
					}

					catch(err)
					{
						// Do nothing
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
