overviewer.views= {}


overviewer.views.WorldView = Backbone.View.extend({
    initialize: function(opts) {
        this.options.mapTypes = [];
        this.options.mapTypeIds = [];
        this.model.get("tileSets").each(function(tset, index, list) {
            var ops = {
                getTileUrl: overviewer.gmap.getTileUrlGenerator(tset.get("path"), tset.get("base"), tset.get("imgextension")),
                'tileSize':     new google.maps.Size(
                                    overviewerConfig.CONST.tileSize,
                                    overviewerConfig.CONST.tileSize),
                'maxZoom':      tset.get("maxZoom"),
                'minZoom':      tset.get("minZoom"),
                'isPng':        (tset.get("imgextension")=="png")
            };
            var newMapType = new google.maps.ImageMapType(ops);
            newMapType.name = tset.get("name");
            newMapType.shortname = tset.get("name");
            newMapType.alt = "Minecraft " + tset.get("name") + " Map";
            newMapType.projection = new overviewer.classes.MapProjection();
    
            this.options.mapTypes.push(newMapType);
            this.options.mapTypeIds.push(overviewerConfig.CONST.mapDivId + this.model.get("name") + tset.get("name"));

        }, this);
    },
});



overviewer.views.WorldSelectorView = Backbone.View.extend({
    initialize: function() {
        if(overviewer.collections.worldViews.length > 1) {
            // a div will have already been created for us, we just
            // need to register it with the google maps control
            var selectBox = document.createElement('select');
            $.each(overviewer.collections.worldViews, function(index, elem) {
                var o = document.createElement("option");
                o.value = elem.model.get("name");
                o.innerHTML = elem.model.get("name");
                $(o).data("viewObj", elem);
                selectBox.appendChild(o);

            });

            this.el.appendChild(selectBox);
            overviewer.map.controls[google.maps.ControlPosition.TOP_LEFT].push(this.el);
        }
    },
    events: {
        "change select":  "changeWorld"
    },
    changeWorld: function() {
        var selectObj = this.$("select")[0];
        var selectedOption = selectObj.options[selectObj.selectedIndex]; 

        overviewer.mapModel.set({currentWorldView: $(selectedOption).data("viewObj")});
        //
     },
    render: function(t) {
        //console.log("WorldSelectorView::render() TODO implement this (low priority)");
    }
});



overviewer.views.CompassView = Backbone.View.extend({
    initialize: function() {
        this.el.index=0;
        var compassImg = document.createElement('IMG');
        compassImg.src = overviewerConfig.CONST.image.compass;
        this.el.appendChild(compassImg);

        overviewer.map.controls[google.maps.ControlPosition.TOP_RIGHT].push(this.el);
    },
    /**
     * CompassView::render
     */
    render: function() {
        var tsetModel = overviewer.mapView.options.currentTileSet;
        var northdir = tsetModel.get("north_direction");
        if (northdir == overviewerConfig.CONST.UPPERLEFT)
            this.$("IMG").attr("src","compass_upper-left.png");
        if (northdir == overviewerConfig.CONST.UPPERRIGHT)
            this.$("IMG").attr("src", "compass_upper-right.png");
        if (northdir == overviewerConfig.CONST.LOWERLEFT)
            this.$("IMG").attr("src", "compass_lower-left.png");
        if (northdir == overviewerConfig.CONST.LOWERRIGHT)
            this.$("IMG").attr("src", "compass_lower-right.png");
    }
});


overviewer.views.CoordboxView = Backbone.View.extend({
    initialize: function() {
        // Coords box
        this.el.id = 'coordsDiv';
        this.el.innerHTML = 'coords here';
        overviewer.map.controls[google.maps.ControlPosition.BOTTOM_LEFT].push(this.el);
    },
    updateCoords: function(latLng) {
        var worldcoords = overviewer.util.fromLatLngToWorld(latLng.lat(), 
        latLng.lng(),
        overviewer.mapView.options.currentTileSet);
        this.el.innerHTML = "Coords: X " + Math.round(worldcoords.x) + ", Z " + Math.round(worldcoords.z);
    }
});



/* GoogleMapView is responsible for dealing with the GoogleMaps API to create the 
 */

overviewer.views.GoogleMapView = Backbone.View.extend({
    initialize: function(opts) {
        this.options.map = null;
        var curWorld = this.model.get("currentWorldView").model;

        var curTset = curWorld.get("tileSets").at(0);

        /*
           var defaultCenter = overviewer.util.fromWorldToLatLng(
           overviewerConfig.map.center[0], 
           overviewerConfig.map.center[1],
           overviewerConfig.map.center[2],
           curTset.get("defaultZoom"));
           */
        var lat = 0.62939453125;// TODO defaultCenter.lat();
        var lng = 0.38525390625; // TODO defaultCenter.lng();
        var mapcenter = new google.maps.LatLng(lat, lng);

        this.options.mapTypes=[];
        this.options.mapTypeIds=[];
        var opts = this.options;

        var mapOptions = {};
    // 
        // init the map with some default options.  use the first tileset in the first world
        this.options.mapOptions = {
            zoom:                   curTset.get("defaultZoom"),
            center:                 mapcenter,
            panControl:             true,
            scaleControl:           false,
            mapTypeControl:         true,
            //mapTypeControlOptions: {
                //mapTypeIds: this.options.mapTypeIds
            //},
            mapTypeId:              '',
            streetViewControl:      false,
            overviewMapControl:     true,
            zoomControl:            true,
            backgroundColor:        curTset.get("bgcolor")
        };

    
        overviewer.map = new google.maps.Map(this.el, this.options.mapOptions);

        // register every ImageMapType with the map
        $.each(overviewer.collections.worldViews, function( index, worldView) {
            $.each(worldView.options.mapTypes, function(i_index, maptype) {
                overviewer.map.mapTypes.set(overviewerConfig.CONST.mapDivId + 
                    worldView.model.get("name") + maptype.shortname , maptype);
            });
        });
        
    },
    /* GoogleMapView::render()
     * Should be called when the current world has changed in GoogleMapModel
     */
    render: function() {
        var view = this.model.get("currentWorldView");
        this.options.mapOptions.mapTypeControlOptions = {
            mapTypeIds: view.options.mapTypeIds};
        this.options.mapOptions.mapTypeId = view.options.mapTypeIds[0];
        overviewer.map.setOptions(this.options.mapOptions);


        return this;
    },
    /**
     * GoogleMapView::updateCurrentTileset()
     * Keeps track of the currently visible tileset
     */
    updateCurrentTileset: function() {
        var currentWorldView = this.model.get("currentWorldView");
        var gmapCurrent = overviewer.map.getMapTypeId();
        for (id in currentWorldView.options.mapTypeIds) {
            if (currentWorldView.options.mapTypeIds[id] == gmapCurrent) {
                this.options.currentTileSet = currentWorldView.model.get("tileSets").at(id);
            }
        }

        // for this world, remember our current viewport (as worldcoords, not LatLng)
        //

    }

});




/**
 * SignControlView
 */
overviewer.views.SignControlView = Backbone.View.extend({
    /** SignControlView::initialize
     */
    initialize: function(opts) {
        $(this.el).addClass("customControl");
        overviewer.map.controls[google.maps.ControlPosition.TOP_RIGHT].push(this.el);

    },
    registerEvents: function(me) {
        google.maps.event.addListener(overviewer.map, 'maptypeid_changed', function(event) {
            overviewer.mapView.updateCurrentTileset();

            // workaround IE issue.  bah!
            if (typeof markers=="undefined") { return; }
            me.render();
            // hide markers, if necessary
            // for each markerSet, check:
            //    if the markerSet isnot part of this tileset, hide all of the markers
            var curMarkerSet = overviewer.mapView.options.currentTileSet.attributes.path;
            var dataRoot = markers[curMarkerSet];
            if (!dataRoot) { 
                // this tileset has no signs, so hide all of them
                for (markerSet in markersDB) {
                    if (markersDB[markerSet].created) {
                        jQuery.each(markersDB[markerSet].raw, function(i, elem) {
                            elem.markerObj.setVisible(false);
                        });
                    }
                }

                return; 
            }
            var groupsForThisTileSet = jQuery.map(dataRoot, function(elem, i) { return elem.groupName;})
            for (markerSet in markersDB) {
                if (jQuery.inArray(markerSet, groupsForThisTileSet) == -1){
                    // hide these
                    if (markersDB[markerSet].created) {
                        jQuery.each(markersDB[markerSet].raw, function(i, elem) {
                            elem.markerObj.setVisible(false);
                        });
                    }
                    markersDB[markerSet].checked=false;
                }
                // make sure the checkboxes checked if necessary
                $("[_mc_groupname=" + markerSet + "]").attr("checked", markersDB[markerSet].checked);

            }

        });

    },
    /**
     * SignControlView::render
     */
    render: function() {

        var curMarkerSet = overviewer.mapView.options.currentTileSet.attributes.path;
        //var dataRoot = overviewer.collections.markerInfo[curMarkerSet];
        var dataRoot = markers[curMarkerSet];

        this.el.innerHTML=""
        
        // if we have no markerSets for this tileset, do nothing:
        if (!dataRoot) { return; }


        var controlText = document.createElement('DIV');
        controlText.innerHTML = "Signs";

        var controlBorder = document.createElement('DIV');
        $(controlBorder).addClass('top');
        this.el.appendChild(controlBorder);
        controlBorder.appendChild(controlText);

        var dropdownDiv = document.createElement('DIV');
        $(dropdownDiv).addClass('dropDown');
        this.el.appendChild(dropdownDiv);
        dropdownDiv.innerHTML='';

        // add the functionality to toggle visibility of the items
        $(controlText).click(function() {
                $(controlBorder).toggleClass('top-active');
                $(dropdownDiv).toggle();
        });


        // add some menus
        for (i in dataRoot) {
            var group = dataRoot[i];
            this.addItem({label: group.displayName, groupName:group.groupName, action:function(this_item, checked) {
                markersDB[this_item.groupName].checked = checked;
                jQuery.each(markersDB[this_item.groupName].raw, function(i, elem) {
                    elem.markerObj.setVisible(checked);
                });
            }});
        }

        //dataRoot['markers'] = [];
        //
        for (i in dataRoot) {
            var groupName = dataRoot[i].groupName;
            if (!markersDB[groupName].created) {
                for (j in markersDB[groupName].raw) {
                    var entity = markersDB[groupName].raw[j];
                    if (entity['id'] == 'Player') {
                        iconURL = "http://overviewer.org/avatar/"
                            + entity['EntityId'];
                    } else if (entity['id'] == 'PlayerSpawn') {
                        iconURL = overviewerConfig.CONST.image.bedMarker;
                    } else if (entity['id'] == 'Sign') {
                        iconURL = overviewerConfig.CONST.image.signMarker;
                    } else {
                        iconURL = overviewerConfig.CONST.image.defaultMarker;
                    }
                    var marker = new google.maps.Marker({
                            'position': overviewer.util.fromWorldToLatLng(entity.x,
                                entity.y, entity.z, overviewer.mapView.options.currentTileSet),
                            'map':      overviewer.map,
                            'title':    jQuery.trim(entity.Text1 + "\n" + entity.Text2 + "\n" + entity.Text3 + "\n" + entity.Text4), 
                            'icon':     iconURL,
                            'visible':  false
                    }); 
                    overviewer.util.createMarkerInfoWindow(marker);
                    jQuery.extend(entity, {markerObj: marker});
                }
                markersDB[groupName].created = true;
            }
        }


    },
    addItem: function(item) {
        var itemDiv = document.createElement('div');
        var itemInput = document.createElement('input');
        itemInput.type='checkbox';

        // give it a name
        $(itemInput).data('label',item.label);
        $(itemInput).attr("_mc_groupname", item.groupName);
        jQuery(itemInput).click((function(local_item) {
            return function(e) {
                item.action(local_item, e.target.checked);
            };
        })(item));

        this.$(".dropDown")[0].appendChild(itemDiv);
        itemDiv.appendChild(itemInput);
        var textNode = document.createElement('text');
        if(item.icon) {
            textNode.innerHTML = '<img width="15" height="15" src="' + 
                item.icon + '">' + item.label + '<br/>';
        } else {
            textNode.innerHTML = item.label + '<br/>';
        }

        itemDiv.appendChild(textNode);


    },
});

/**
 * SpawnIconView
 */
overviewer.views.SpawnIconView = Backbone.View.extend({
    render: function() {
        // 
        var curTileSet = overviewer.mapView.options.currentTileSet;
        if (overviewer.collections.spawnMarker) {
            overviewer.collections.spawnMarker.setMap(null);
            overviewer.collections.spawnMarker = null;
        }
        var spawn = curTileSet.get("spawn");
        if (spawn) {
            overviewer.collections.spawnMarker = new google.maps.Marker({
                'position': overviewer.util.fromWorldToLatLng(spawn[0],
                    spawn[1], spawn[2], overviewer.mapView.options.currentTileSet),
                'map':      overviewer.map,
                'title':    'spawn',
                'icon':     overviewerConfig.CONST.image.spawnMarker,
                'visible':  false
                }); 
            overviewer.collections.spawnMarker.setVisible(true);
        }
    }
});

