var overviewerConfig = {
    //These will probably never change
    'CONST': {
        'tileSize':             384,
        'image': {
            'defaultMarker':    'signpost.png',
            'signMarker':       'signpost_icon.png',
            'compass':          'compass.png',
            'spawnMarker':      'http://google-maps-icons.googlecode.com/files/home.png',
            'queryMarker':      'http://google-maps-icons.googlecode.com/files/regroup.png'
        },
        'mapDivId':             'mcmap',
        'regionStrokeWeight':   2
    },
    'map': {
        'controls': {
            'navigation':   true
        },
        'defaultZoom':  0,
        'minZoom':      {minzoom},
        'maxZoom':      {maxzoom},
        'center':       {spawn_coords},
        'cacheMinutes': 0,
        'debug':        false,
    },
    'objectGroups': {
        'signs': [
            {
                'label':    'All',
                'match':    function(sign) {
                    return true;
                }
            }
        ],
        'regions': []
    },
    'mapTypes':         {maptypedata}
};