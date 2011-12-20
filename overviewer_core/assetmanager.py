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

        #  stores Points Of Interest to be mapped with markers
        #  This is a dictionary of lists of dictionaries
        #  Each regionset's name is a key in this dictionary
        self.POI = dict()

    def found_poi(self, regionset, poi_type, contents, chunkX, chunkY):
        if regionset.name not in self.POI.keys():
            POI[regionset.name] = []
        # TODO based on the type, so something
        POI[regionset.name].append


    def finalize(self):

        # the following bit of code came from googlemap.py
        zoomlevel = self.p

        bgcolor = (int(self.bg_color[1:3],16), int(self.bg_color[3:5],16), int(self.bg_color[5:7],16), 0)
        blank = Image.new("RGBA", (1,1), bgcolor)
        # Write a blank image
        for quadtree in self.quadtrees:
            tileDir = os.path.join(self.destdir, quadtree.tiledir)
            if not os.path.exists(tileDir): os.mkdir(tileDir)
            blank.save(os.path.join(tileDir, "blank."+quadtree.imgformat))
        
        # copy web assets into destdir:
        global_assets = os.path.join(util.get_program_path(), "overviewer_core", "data", "web_assets")
        if not os.path.isdir(global_assets):
            global_assets = os.path.join(util.get_program_path(), "web_assets")
        mirror_dir(global_assets, self.destdir)
        
        # do the same with the local copy, if we have it
        if self.web_assets_path:
            mirror_dir(self.web_assets_path, self.destdir)

        # replace the config js stuff
        # TODO don't replace stuff, but generate it from scratch
        config = codecs.open(os.path.join(self.outputdir, 'overviewerConfig.js'), 'r', encoding='UTF-8').read()
        config = config.replace(
                "{minzoom}", str(0))
        config = config.replace(
                "{maxzoom}", str(zoomlevel))
        config = config.replace(
                "{zoomlevels}", str(zoomlevel))
        config = config.replace(
                "{north_direction}", self.north_direction)
        
        config = config.replace("{spawn_coords}",
                                json.dumps(list(self.world.spawn)))



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
