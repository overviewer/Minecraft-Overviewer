overviewer.views= {}


overviewer.views.WorldView = Backbone.View.extend({
    initialize: function(opts) {
        this.options.mapTypes = [];
        this.options.overlayMapTypes = [];
        this.options.mapTypeIds = [];
        this.options.overlayMapTypeIds = [];

        var curTileSet = this.model.get("tileSets").at(0);
        var spawn = curTileSet.get("spawn");
        if (spawn == "false") {
            var spawn = [0,64,0];
        }
        this.options.lastViewport = [spawn[0],spawn[1],spawn[2],curTileSet.get("defaultZoom")];

        this.model.get("tileSets").each(function(tset, index, list) {
            // ignore overlays:
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
            newMapType._ov_tileSet = tset;

            if (tset.get("isOverlay")) {
                newMapType.tiles = tset.get("tilesets");
                this.options.overlayMapTypes.push(newMapType);
                this.options.overlayMapTypeIds.push(overviewerConfig.CONST.mapDivId + this.model.get("name") + tset.get("name"));
            } else {
                this.options.mapTypes.push(newMapType);
                this.options.mapTypeIds.push(overviewerConfig.CONST.mapDivId + this.model.get("name") + tset.get("name"));
            }

        }, this);

        this.model.get("tileSets").each(function(tset, index, list) {
            // ignore non-overlays:
            if (!tset.get("isOverlay")) { return; };

        });
    }
});



overviewer.views.WorldSelectorView = Backbone.View.extend({
    initialize: function() {
        if(overviewer.collections.worldViews.length > 1) {
            $(this.el).addClass("customControl");
            
            // a div will have already been created for us, we just
            // need to register it with the google maps control
            var selectBox = document.createElement('select');
            $.each(overviewer.collections.worldViews, function(index, elem) {
                var o = document.createElement("option");
                o.value = elem.model.get("name");
                o.innerHTML = elem.model.get("name");
                if (elem.model == overviewer.mapModel.get("currentWorldView").model) {
                    o.selected=true;
                }
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
        compassImg.src = '';  // this will be set properly in the render function (below)
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

        var regionfileX = Math.floor(Math.floor(worldcoords.x / 16.0) / 32.0);
        var regionfileZ = Math.floor(Math.floor(worldcoords.z / 16.0) / 32.0);
        var regionfilename = "r." + regionfileX + "." + regionfileZ + ".mca";

        this.el.innerHTML = "<strong>X</strong> " + Math.round(worldcoords.x) +
                            " <strong>Z</strong> " + Math.round(worldcoords.z) +
                            " (" + regionfilename + ")";
    }
});

overviewer.views.ProgressView = Backbone.View.extend({
    initialize: function() {
        this.el.id = 'progressDiv';
        this.el.innerHTML = 'Current Render Progress';
        overviewer.map.controls[google.maps.ControlPosition.BOTTOM_RIGHT].push(this.el);
        $(this.el).hide();
        $.ajaxSetup({cache: false});
    },
    updateProgress: function() {
        e = this;
        $.getJSON('progress.json', null, function(d){
            if (!(d == null||d=='')) {
                $(e.el).show();
                e.el.innerHTML = d['message'];
                if (d.update > 0) {
                    setTimeout("e.updateProgress()", d.update);
                } else {
                    setTimeout("e.updateProgress()", 60000);
                    e.el.innerHTML="Hidden - d.update < 0";
                    $(e.el).hide();
                }
            } else {
                e.el.innerHTML="Hidden - !!d==false";
                $(e.el).hide();
            }
        });
    }
});

/* GoogleMapView is responsible for dealing with the GoogleMaps API to create the 
 */

overviewer.views.GoogleMapView = Backbone.View.extend({
    initialize: function(opts) {
        this.options.map = null;
        var curWorld = this.model.get("currentWorldView").model;

        var curTset = curWorld.get("tileSets").at(0);
        var spawn = curTset.get("spawn");
        if (spawn == "false") {
            var spawn = [0,64,0];
        }
        var mapcenter = overviewer.util.fromWorldToLatLng(
           spawn[0],
           spawn[1],
           spawn[2],
           curTset);


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
                this.options.currentTileSet = currentWorldView.options.mapTypes[id]._ov_tileSet;
            }
        }

        // for this world, remember our current viewport (as worldcoords, not LatLng)
        //

    }

});


/**
 * OverlayControlView
 */
overviewer.views.OverlayControlView = Backbone.View.extend({
    /** OverlayControlVIew::initialize
     */
    initialize: function(opts) {
        $(this.el).addClass("customControl");
        overviewer.map.controls[google.maps.ControlPosition.TOP_RIGHT].push(this.el); 
    },
    registerEvents: function(me) {
        overviewer.mapModel.bind("change:currentWorldView", me.render, me);
    },

    /**
     * OverlayControlView::render
     */
    render: function() {
        this.el.innerHTML="";
        
        // hide all visible overlays:
        overviewer.map.overlayMapTypes.clear()

        // if this world has no overlays, don't create this control
        var mapTypes = overviewer.mapModel.get('currentWorldView').options.overlayMapTypes;
        if (mapTypes.length == 0) { return; }

        var controlText = document.createElement('DIV');
        controlText.innerHTML = "Overlays";
        
        var controlBorder = document.createElement('DIV');
        $(controlBorder).addClass('top');
        this.el.appendChild(controlBorder);
        controlBorder.appendChild(controlText);
        
        var dropdownDiv = document.createElement('DIV');
        $(dropdownDiv).addClass('dropDown');
        this.el.appendChild(dropdownDiv);
        dropdownDiv.innerHTML='';
        
        $(controlText).click(function() {
                $(controlBorder).toggleClass('top-active');
                $(dropdownDiv).toggle();
        });

        var currentTileSetPath = overviewer.mapView.options.currentTileSet.get('path');
        
        for (i in mapTypes) {
            var mt = mapTypes[i];
            // if this overlay specifies a list of valid tilesets, then skip over any invalid tilesets 
            if ((mt.tiles.length > 0) && (mt.tiles.indexOf(currentTileSetPath) ==-1)) {
                continue;
            }
            this.addItem({label: mt.name,
                    name: mt.name,
                    mt: mt,

                    action: function(this_item, checked) {
                        if (checked) {
                            overviewer.map.overlayMapTypes.push(this_item.mt);
                        } else {
                            var idx_to_delete = -1;
                            overviewer.map.overlayMapTypes.forEach(function(e, j) {
                                if (e == this_item.mt) {
                                    idx_to_delete = j;
                                }
                            });
                            if (idx_to_delete >= 0) {
                                overviewer.map.overlayMapTypes.removeAt(idx_to_delete);
                            }
                        }

                    }
            });
        }


    },

    addItem: function(item) {
        var itemDiv = document.createElement('div');
        var itemInput = document.createElement('input');
        itemInput.type='checkbox';

        // if this overlay is already visible, set the checkbox
        // to checked
        overviewer.map.overlayMapTypes.forEach(function(e, j) {
            if (e == item.mt) {
                itemInput.checked=true;
            }
        });
        
        // give it a name
        $(itemInput).attr("_mc_overlayname", item.name);
        jQuery(itemInput).click((function(local_item) {
            return function(e) {
                item.action(local_item, e.target.checked);
            };
        })(item));

        this.$(".dropDown")[0].appendChild(itemDiv);
        itemDiv.appendChild(itemInput);
        var textNode = document.createElement('text');
        textNode.innerHTML = item.label + '<br/>';

        itemDiv.appendChild(textNode);
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


            // hide markers that are part of other tilesets than this
            // for each markerSet, check:
            //    if the markerSet isnot part of this tileset, hide all of the markers
            var curMarkerSet = overviewer.mapView.options.currentTileSet.get("path");
            var dataRoot = markers[curMarkerSet];

            jQuery.each(markers, function(key, markerSet) {
                if (key != curMarkerSet) {
                    jQuery.each(markerSet, function(i, markerGroup) {
                        if (typeof markerGroup.markerObjs != "undefined") {
                            jQuery.each(markerGroup.markerObjs, function(j, markerObj) {
                                markerObj.setVisible(false);
                            });
                        }
                    });
                }
            });

            return;

        });

    },
    /**
     * SignControlView::render
     */
    render: function() {

        var curMarkerSet = overviewer.mapView.options.currentTileSet.get("path");
        //var dataRoot = overviewer.collections.markerInfo[curMarkerSet];
        var dataRoot = markers[curMarkerSet];

        this.el.innerHTML="";
        
        // if we have no markerSets for this tileset, do nothing:
        if (!dataRoot) { return; }


        var controlText = document.createElement('DIV');
        controlText.innerHTML = overviewer.mapView.options.currentTileSet.get("poititle");

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



        //dataRoot['markers'] = [];
        //
        for (i in dataRoot) {
            var groupName = dataRoot[i].groupName;
            if (!dataRoot[i].created) {
                dataRoot[i].markerObjs = [];
                for (j in markersDB[groupName].raw) {
                    var entity = markersDB[groupName].raw[j];
                    if (entity['icon']) {
                        iconURL = entity['icon'];
                    } else {
                        iconURL = dataRoot[i].icon;
                    }
                    var marker = new google.maps.Marker({
                            'position': overviewer.util.fromWorldToLatLng(entity.x,
                                entity.y, entity.z, overviewer.mapView.options.currentTileSet),
                            'map':      overviewer.map,
                            'title':    jQuery.trim(entity.hovertext), 
                            'content':  jQuery.trim(entity.text),
                            'icon':     iconURL,
                            'visible':  false
                    }); 
                    if(entity['createInfoWindow'] == true) {
                        overviewer.util.createMarkerInfoWindow(marker);
                    } else {
                        if(dataRoot[i].createInfoWindow == true) {
                            overviewer.util.createMarkerInfoWindow(marker);
                        }
                    }
                    dataRoot[i].markerObjs.push(marker);
                    // Polyline stuff added by FreakusGeekus. Probably needs work.
                    if (typeof entity['polyline'] != 'undefined') {
						var polypath = new Array();
                        for (point in entity.polyline) {
                            polypath.push(overviewer.util.fromWorldToLatLng(entity.polyline[point].x, entity.polyline[point].y, entity.polyline[point].z, overviewer.mapView.options.currentTileSet));
                        }
                        
                        var polyline = new google.maps.Polyline({
								'path': polypath,
                                'clickable': false,
                                'map': overviewer.map,
                                'visible': false,
								'strokeColor': entity['strokeColor']
                        });
						dataRoot[i].markerObjs.push(polyline);
                    }
					
					// Polygons
                    if (typeof entity['polygon'] != 'undefined') {
						var polypath = new Array();
                        for (point in entity.polygon) {
                            polypath.push(overviewer.util.fromWorldToLatLng(entity.polygon[point].x, entity.polygon[point].y, entity.polygon[point].z, overviewer.mapView.options.currentTileSet));
                        }
                        
                        var polygon = new google.maps.Polygon({
								'clickable': false,
								'fillColor': entity['fillColor'],
								'fillOpacity': entity['fillOpacity'],
								'map': overviewer.map,
								'path': polypath,
								'strokeColor': entity['strokeColor'],
								'strokeOpacity': entity['strokeOpacity'],
								'visible': false
                        });
						dataRoot[i].markerObjs.push(polygon);
                    }
                }
                dataRoot[i].created = true;
            }
        }
        
        // add some menus
        for (i in dataRoot) {
            var group = dataRoot[i];
            this.addItem({group: group, action:function(this_item, checked) {
                this_item.group.checked = checked;
                jQuery.each(this_item.group.markerObjs, function(i, markerObj) {
                    markerObj.setVisible(checked);
                });
            }});
            if (group.checked) {
                jQuery.each(group.markerObjs, function(i, markerObj) {
                    markerObj.setVisible(true);
                });
            }
        }


    },
    addItem: function(item) {
        var itemDiv = document.createElement('div');
        var itemInput = document.createElement('input');
        itemInput.type='checkbox';

        if (item.group.checked) {
            itemInput.checked="true";
        }

        // give it a name
        $(itemInput).data('label',item.group.displayName);
        $(itemInput).attr("_mc_groupname", item.group.gropuName);
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
                item.icon + '">' + item.group.displayName + '&nbsp;<br/>';
        } else {
            textNode.innerHTML = item.group.displayName + '&nbsp;<br/>';
        }

        itemDiv.appendChild(textNode);
        itemDiv.style.whiteSpace = "nowrap";

    }
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

overviewer.views.LocationIconView = Backbone.View.extend({
    render: function() {
        // 
    if (overviewer.collections.locationMarker) {
        overviewer.collections.locationMarker.setMap(null);
        overviewer.collections.locationMarker = null;
    }
    overviewer.collections.locationMarker = new google.maps.Marker({
        'position': overviewer.map.getCenter(), 
        'map':      overviewer.map,
        'title':    'location',
        'icon':     overviewerConfig.CONST.image.queryMarker,
        'visible':  false
    }); 
    overviewer.collections.locationMarker.setVisible(overviewer.mapView.options.currentTileSet.get("showlocationmarker"));

    }
});

