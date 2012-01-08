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

import os
import os.path
import stat
import cPickle
import Image
import shutil
from time import strftime, localtime
import json
import locale
import codecs

import util
from c_overviewer import get_render_mode_inheritance, get_render_mode_info
import overviewer_version

"""
This module has routines related to generating a Google Maps-based
interface out of a set of tiles.

"""


class MapGen(object):
    def __init__(self, quadtrees, configInfo):
        """Generates a Google Maps interface for the given list of
        quadtrees. All of the quadtrees must have the same destdir,
        image format, and world. 

        Note:tiledir for each quadtree should be unique. By default the tiledir
        is determined by the rendermode
        
        """
        
        self.skipjs = configInfo.get('skipjs', False)
        self.nosigns = configInfo.get('nosigns', False)
        self.web_assets_hook = configInfo.get('web_assets_hook', None)
        self.web_assets_path = configInfo.get('web_assets_path', None)
        self.bg_color = configInfo.get('bg_color')
        self.north_direction = configInfo.get('north_direction', 'lower-left')
        
        if not len(quadtrees) > 0:
            raise ValueError("there must be at least one quadtree to work on")
        
        self.destdir = quadtrees[0].destdir
        self.regionobj = quadtrees[0].regionobj
        self.p = quadtrees[0].p
        for i in quadtrees:
            if i.destdir != self.destdir or i.regionobj != self.regionobj:
                raise ValueError("all the given quadtrees must have the same destdir and world")
        
        self.quadtrees = quadtrees
    
    def go(self, procs):
        """Writes out overviewerConfig.js and does copying of the other static web assets

        """
### TODO remove this method?  It has been moved into assetmanager.py

        pass


    def finalize(self):
        """Write out persistent data file and marker listings file
        """
### TODO remove this method?  It has been moved into assetmanager.py
        pass 


        
