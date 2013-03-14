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

import os.path
import os
import logging
import zipfile

import OIL

"""
This file contains the Textures class, which is responsible for
finding, loading, and returning named textures to be used in the
renderer. It also contains a function to return a Textures object
associated with the default Minecraft textures, if the Minecraft
client is installed.
"""

class TextureException(Exception):
    """To be thrown when a texture is not found."""
    pass

class Textures(object):
    """An object that finds and returns OIL Images for named
    textures. It is initialized with a texture pack path; this can be
    either a directory, zip file, or jar file (like the client jar).
    """
    
    def __init__(self, path):
        if not os.path.exists(path):
            raise ValueError("texture path does not exist")
        
        self.path = path
        self.zip = None
        if os.path.isfile(path) and zipfile.is_zipfile(path):
            self.zip = zipfile.ZipFile(path)
        if not (self.zip or os.path.isdir(path)):
            raise ValueError("texture path must either be a directory or a zip file")
        
        self.cache = {}
    
    def __getstate__(self):
        # Images don't pickle, so, just use the path
        return self.path
    
    def __setstate__(self, path):
        self.__init__(path)
    
    def _find_file(self, filename):
        if self.zip:
            try:
                self.zip.getinfo(filename)
                logging.debug("Found '%s' in zip '%s'", filename, self.path)
                return self.zip.open(filename)
            except (KeyError, IOError), e:
                raise TextureException("could not find '%s'" % (filename,))
        else:
            full_filename = os.path.join(self.path, filename)
            if not os.path.isfile(full_filename):
                raise TextureException("could not find '%s'" % (filename,))
            logging.debug("Found '%s' in directory '%s'", filename, self.path)
            return open(full_filename, 'rb')
        
    def load(self, filename):
        # Textures may be animated, this method un-animates them
        if filename in self.cache:
            return self.cache[filename]
        
        img = OIL.Image.load(self._find_file(filename))
        w, h = img.size
        if w != h:
            dest = OIL.Image(w, w)
            dest.composite(img, 255, 0, 0)
            img = dest
        
        self.cache[filename] = img
        return img

def get_default():
    jarpaths = []
    if "APPDATA" in os.environ:
        jarpaths.append(os.path.join(os.environ["APPDATA"], ".minecraft", "bin", "minecraft.jar"))
    if "HOME" in os.environ:
        jarpaths.append(os.path.join(os.environ["HOME"], "Library", "Application Support", "minecraft", "bin", "minecraft.jar"))
        jarpaths.append(os.path.join(os.environ["HOME"], ".minecraft", "bin", "minecraft.jar"))
    
    for jar in jarpaths:
        if zipfile.is_zipfile(jar):
            logging.debug("Using minecraft.jar at '%s'", jar)
            return Textures(jar)
    raise RuntimeError("could not locate Minecraft client for textures")
