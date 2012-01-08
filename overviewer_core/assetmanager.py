#    This file is part of the Minecraft Overviewer.
#
#    Minecraft Overviewer is free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or (at
#    your option) any later version.
#
#    Minecraft Overviewer is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#    Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with the Overviewer.  If not, see <http://www.gnu.org/licenses/>.

import json
import os
import codecs
from PIL import Image

class AssetManager(object):
    """\
These objects provide an interface to metadata and persistent data, and at the
same time, controls the generated javascript files in the output directory.
There should only be one instances of these per execution.
    """

    def __init__(self, outputdir):
        """\
Initializes the AssetManager with the top-level output directory.  
It can read/parse and write/dump the overviewerConfig.js file into this top-level
directory. 
        """
        self.outputdir = outputdir
        self.renders = dict()

        #  stores Points Of Interest to be mapped with markers
        #  This is a dictionary of lists of dictionaries
        #  Each regionset's name is a key in this dictionary
        self.POI = dict()

    def found_poi(self, regionset, poi_type, contents, chunkX, chunkY):
        if regionset.name not in self.POI.keys():
            POI[regionset.name] = []
        # TODO based on the type, so something
        POI[regionset.name].append


    def finalize(self, tilesets):

        # dictionary to hold the overviewerConfig.js settings that we will dumps
        dump = dict()
        dump['CONST'] = dict(tileSize=384)
        dump['CONST']['image'] = {
                'defaultMarker':    'signpost.png',
                'signMarker':       'signpost_icon.png',
                'compass':          'compass_lower-left.png',
                'spawnMarker':      'http://google-maps-icons.googlecode.com/files/home.png',
                'queryMarker':      'http://google-maps-icons.googlecode.com/files/regroup.png'
                }
        dump['CONST']['mapDivId'] = 'mcmap'
        dump['CONST']['regionStrokeWeight'] = 2

        # based on the tilesets we have, group them by worlds
        worlds = []
        for tileset in tilesets:
            if tileset.options.get('worldname_orig') not in worlds:
                worlds.append(tileset.options.get('worldname_orig'))

        dump['worlds'] = worlds
        dump['map'] = dict()
        dump['map']['debug'] = True
        dump['map']['center'] = [-314, 67, 94]
        dump['map']['controls'] = {
            'pan': True,
            'zoom': True,
            'spawn': True,
            'compass': True,
            'mapType': True,
            'overlays': True,
            'coordsBox': True,
            'searchBox': True
            }


        dump['tilesets'] = []

        def bgcolorformat(color):
            return "#%02x%02x%02x" % color[0:3]

        for tileset in tilesets:
            js_tileset = dict()
            js_tileset['name'] = tileset.options.get('name')
            js_tileset['zoomLevels'] = tileset.treedepth
            js_tileset['minZoom'] = 0
            js_tileset['defaultZoom'] = 1
            js_tileset['maxZoom'] = tileset.treedepth
            js_tileset['path'] = tileset.options.get('name')
            js_tileset['base'] = ''
            js_tileset['bgcolor'] = bgcolorformat(tileset.options.get('bgcolor'))
            dump['tilesets'].append(js_tileset)

            # write a blank image
            blank = Image.new("RGBA", (1,1), tileset.options.get('bgcolor'))
            blank.save(os.path.join(self.outputdir, tileset.options.get('name'), "blank."+tileset.options.get('imgformat')))


        jsondump = json.dumps(dump, indent=4)
        with codecs.open(os.path.join(self.outputdir, 'overviewerConfig.js'), 'w', encoding='UTF-8') as f:
            f.write("var overviewerConfig = " + jsondump + ";\n")

          
        
        # copy web assets into destdir:
        global_assets = os.path.join(util.get_program_path(), "overviewer_core", "data", "web_assets")
        if not os.path.isdir(global_assets):
            global_assets = os.path.join(util.get_program_path(), "web_assets")
        mirror_dir(global_assets, self.destdir)
        
        # do the same with the local copy, if we have it
        if self.web_assets_path:
            mirror_dir(self.web_assets_path, self.destdir)



        # helper function to get a label for the given rendermode
        def get_render_mode_label(rendermode):
            info = get_render_mode_info(rendermode)
            if 'label' in info:
                return info['label']
            return rendermode.capitalize()

        
        # create generated map type data, from given quadtrees
        maptypedata = map(lambda q: {'label' : get_render_mode_label(q.rendermode),
                                     'shortname' : q.rendermode,
                                     'path' : q.tiledir,
                                     'bg_color': self.bg_color,
                                     'overlay' : 'overlay' in get_render_mode_inheritance(q.rendermode),
                                     'imgformat' : q.imgformat},
                          self.quadtrees)
        config = config.replace("{maptypedata}", json.dumps(maptypedata))
        
        with codecs.open(os.path.join(self.destdir, "overviewerConfig.js"), 'w', encoding='UTF-8') as output:
            output.write(config)
   


        # Add time and version in index.html
        indexpath = os.path.join(self.destdir, "index.html")

        index = codecs.open(indexpath, 'r', encoding='UTF-8').read()
        index = index.replace("{title}", "%s Map - Minecraft Overviewer" % self.world.name)
        index = index.replace("{time}", strftime("%a, %d %b %Y %H:%M:%S %Z", localtime()).decode(locale.getpreferredencoding()))
        versionstr = "%s (%s)" % (overviewer_version.VERSION, overviewer_version.HASH[:7])
        index = index.replace("{version}", versionstr)

        with codecs.open(os.path.join(self.destdir, "index.html"), 'w', encoding='UTF-8') as output:
            output.write(index)


        # the following bit of code came from world.py

        # if it exists, open overviewer.dat, and read in the data structure
        # info self.persistentData.  This dictionary can hold any information
        # that may be needed between runs.
        # Currently only holds into about POIs (more more details, see quadtree)
        self.pickleFile = os.path.join(self.outputdir, "overviewer.dat")
        
        ### TODO: Deal with old picklefiles that still live in the world directory

        if os.path.exists(self.pickleFile):
            self.persistentDataIsNew = False
            with open(self.pickleFile,"rb") as p:
                self.persistentData = cPickle.load(p)
                if not self.persistentData.get('north_direction', False):
                    # this is a pre-configurable-north map, so add the north_direction key
                    self.persistentData['north_direction'] = 'lower-left'
        else:
            # some defaults, presumably a new map
            self.persistentData = dict(POI=[], north_direction='lower-left')
            self.persistentDataIsNew = True # indicates that the values in persistentData are new defaults, and it's OK to override them
        
        # the following bit of code came from googlemap.py
        
        # since we will only discover PointsOfInterest in chunks that need to be 
        # [re]rendered, POIs like signs in unchanged chunks will not be listed
        # in self.world.POI.  To make sure we don't remove these from markers.js
        # we need to merge self.world.POI with the persistant data in world.PersistentData

        self.world.POI += filter(lambda x: x['type'] != 'spawn', self.world.persistentData['POI'])

        if self.nosigns:
            markers = filter(lambda x: x['type'] != 'sign', self.world.POI)
        else:
            markers = self.world.POI


        # save persistent data
        self.world.persistentData['POI'] = self.world.POI
        self.world.persistentData['north_direction'] = self.world.north_direction
        with open(self.world.pickleFile,"wb") as f:
            cPickle.dump(self.world.persistentData,f)

        ### TODO find a better mechanism than --skipjs
        if self.skipjs:
            if self.web_assets_hook:
                self.web_assets_hook(self)

        # write out the default marker table
        with codecs.open(os.path.join(self.outputdir, "markers.js"), 'w', encoding='UTF-8') as output:
            output.write("// This is a generated file. Please do not edit it!\n")
            output.write("overviewer.collections.markerDatas.push(\n")
            output.write("// --start marker json dump--\n")
            json.dump(markers, output, indent=1)
            output.write("\n// --end marker json dump--\n")
            output.write(");\n")
