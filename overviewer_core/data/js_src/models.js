overviewer.models = {};

/* WorldModel
 * Primarily has a collection of TileSets
 */
overviewer.models.WorldModel = Backbone.Model.extend({
    initialize: function(attrs) {
        attrs.tileSets = new overviewer.models.TileSetCollection();
        this.set(attrs);
    }
});


/* WorldCollection
 * A collection of WorldModels
 */
overviewer.models.WorldCollection = Backbone.Collection.extend({
    model: overviewer.models.WorldModel
});


/* TileSetModel
 */
overviewer.models.TileSetModel = Backbone.Model.extend({
    defaults: {
        markers: []
    },
    initialize: function(attrs) {
        // this implies that the Worlds collection must be
        // initialized before any TIleSetModels are created
        attrs.world = overviewer.collections.worlds.get(attrs.world);
        this.set(attrs);
    }
});

overviewer.models.TileSetCollection = Backbone.Collection.extend({
    model: overviewer.models.TileSetModel
});


overviewer.models.GoogleMapModel = Backbone.Model.extend({
    initialize: function(attrs) {
        attrs.currentWorldView = overviewer.collections.worldViews[0];
        this.set(attrs);
    }
});

