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
import locale
import time
import logging

from PIL import Image

import world
import util
import overviewer_version

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

        # look for overviewerConfig in self.outputdir
        try:
            with open(os.path.join(self.outputdir, "overviewerConfig.js")) as c:
                overviewerConfig_str = "{" + "\n".join(c.readlines()[1:-1]) + "}"
            self.overviewerConfig = json.loads(overviewerConfig_str)
        except Exception, e:
            if os.path.exists(os.path.join(self.outputdir, "overviewerConfig.js")):
                logging.warning("A previous overviewerConfig.js was found, but I couldn't read it for some reason. Continuing with a blank config")
            self.overviewerConfig = dict(tilesets=dict())

    def get_tileset_config(self, name):
        "Return the correct dictionary from the parsed overviewerConfig.js"
        for conf in self.overviewerConfig['tilesets']:
            if conf['path'] == name:
                return conf
        return dict()
        


    def found_poi(self, regionset, poi_type, contents, chunkX, chunkY):
        if regionset.name not in self.POI.keys():
            POI[regionset.name] = []
        # TODO based on the type, so something
        POI[regionset.name].append

    def initialize(self, tilesets):
        """Similar to finalize() but calls the tilesets' get_initial_data()
        instead of get_persistent_data() to compile the generated javascript
        config.

        """
        return self.finalize(tilesets, True)

    def finalize(self, tilesets, initial=False):
        """Called to output the generated javascript and all static files to
        the output directory

        """
        if not initial:
            get_data = lambda tileset: tileset.get_persistent_data()
        else:
            get_data = lambda tileset: tileset.get_initial_data()

        # dictionary to hold the overviewerConfig.js settings that we will dumps
        dump = dict()
        dump['CONST'] = dict(tileSize=384)
        dump['CONST']['image'] = {
                'defaultMarker':    'signpost.png',
                'signMarker':       'signpost_icon.png',
                'compass':          'compass_upper-left.png',
                'spawnMarker':      'http://google-maps-icons.googlecode.com/files/home.png',
                'queryMarker':      'http://google-maps-icons.googlecode.com/files/regroup.png'
                }
        dump['CONST']['mapDivId'] = 'mcmap'
        dump['CONST']['regionStrokeWeight'] = 2
        dump['CONST']['UPPERLEFT']  = world.UPPER_LEFT;
        dump['CONST']['UPPERRIGHT'] = world.UPPER_RIGHT;
        dump['CONST']['LOWERLEFT']  = world.LOWER_LEFT;
        dump['CONST']['LOWERRIGHT'] = world.LOWER_RIGHT;

        # based on the tilesets we have, group them by worlds
        worlds = []
        for tileset in tilesets:
            full_name = get_data(tileset)['world']
            if full_name not in worlds:
                worlds.append(full_name)

        dump['worlds'] = worlds
        dump['map'] = dict()
        dump['map']['debug'] = True
        dump['map']['cacheTag'] = str(int(time.time()))
        dump['map']['north_direction'] = 'lower-left' # only temporary
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


        for tileset in tilesets:
            dump['tilesets'].append(get_data(tileset))

            # write a blank image
            blank = Image.new("RGBA", (1,1), tileset.options.get('bgcolor'))
            blank.save(os.path.join(self.outputdir, tileset.options.get('name'), "blank." + tileset.options.get('imgformat')))


        jsondump = json.dumps(dump, indent=4)
        with util.FileReplacer(os.path.join(self.outputdir, "overviewerConfig.js")) as tmpfile:
            with codecs.open(tmpfile, 'w', encoding='UTF-8') as f:
                f.write("var overviewerConfig = " + jsondump + ";\n")

        # copy web assets into destdir:
        global_assets = os.path.join(util.get_program_path(), "overviewer_core", "data", "web_assets")
        if not os.path.isdir(global_assets):
            global_assets = os.path.join(util.get_program_path(), "web_assets")
        util.mirror_dir(global_assets, self.outputdir)

        # create overviewer.js from the source js files
        js_src = os.path.join(util.get_program_path(), "overviewer_core", "data", "js_src")
        if not os.path.isdir(js_src):
            js_src = os.path.join(util.get_program_path(), "js_src")
        with util.FileReplacer(os.path.join(self.outputdir, "overviewer.js")) as tmpfile:
            with open(tmpfile, "w") as fout:
                # first copy in js_src/overviewer.js
                with open(os.path.join(js_src, "overviewer.js"), 'r') as f:
                    fout.write(f.read())
                # now copy in the rest
                for js in os.listdir(js_src):
                    if not js.endswith("overviewer.js") and js.endswith(".js"):
                        with open(os.path.join(js_src,js)) as f:
                            fout.write(f.read())
        # helper function to get a label for the given rendermode
        def get_render_mode_label(rendermode):
            info = get_render_mode_info(rendermode)
            if 'label' in info:
                return info['label']
            return rendermode.capitalize()


        # Add time and version in index.html
        indexpath = os.path.join(self.outputdir, "index.html")

        index = codecs.open(indexpath, 'r', encoding='UTF-8').read()
        index = index.replace("{title}", "Minecraft Overviewer")
        index = index.replace("{time}", time.strftime("%a, %d %b %Y %H:%M:%S %Z", time.localtime()).decode(locale.getpreferredencoding()))
        versionstr = "%s (%s)" % (overviewer_version.VERSION, overviewer_version.HASH[:7])
        index = index.replace("{version}", versionstr)

        with util.FileReplacer(indexpath) as indexpath:
            with codecs.open(indexpath, 'w', encoding='UTF-8') as output:
                output.write(index)
