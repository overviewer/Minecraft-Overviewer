// Global variables
var markers = new Array();
var urlParams = {};
var markersInit = false;
var regionsInit = false;
 

function initMarkers() {
	if (markersInit) { return; }
	markersInit = true;

	if (markerData == null) return;
	
	for (i in markerData) {
		addMarker(markerData[i]);
	}
}
  
function initRegions() {
	if (regionsInit) { return; }

	regionsInit = true;

	for (i in regionData) {
		var region = regionData[i];
		var converted = new google.maps.MVCArray();
		for (j in region.path) {
			ar point = region.path[j];
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
		


  function delMarker(markername) {
        marker = markers[markername];

    if (marker) {
                marker.setMap(null);
                delete markers[markername];
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



  function mapMarkersInit() {
    
	// initialize the markers
	initMarkers();
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
