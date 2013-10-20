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
import sys
import logging
import zipfile

from overviewer import util
from overviewer.oil import Image

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
    
    def __init__(self, local_path=None):
        if local_path and not os.path.exists(local_path):
            raise ValueError("texture path does not exist")

        self.local_path = local_path
        self.jar = None
        self.jarpath = ""

        self.cache = {}
    
    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes["jar"]
        del attributes["cache"]
        return attributes

    def __setstate__(self, attributes):
        self.__dict__ = attributes
        self.jar = None
        self.cache = {}

    def _find_file(self, filename, mode="rb", verbose=False):
        """Searches for the given file and returns an open handle to it.
        This searches the following locations in this order:

        * In the directory local_path given in the initializer
        * The program dir (same dir as overviewer.py) for extracted textures
        * On Darwin, in /Applications/Minecraft for extracted textures
        * Inside a minecraft client jar. Client jars are searched for in the
          following location depending on platform:

            * On Windows, at %APPDATA%/.minecraft/versions/
            * On Darwin, at
                $HOME/Library/Application Support/minecraft/versions
            * at $HOME/.minecraft/versions/

          Only the latest non-snapshot version >1.6 is used

        * The overviewer/data/textures dir

        In all of these, files are searched for in '.', 'anim', 'misc/', and
        'environment/'.

        """
        if verbose: logging.info("Starting search for {0}".format(filename))

        # a list of subdirectories to search for a given file,
        # after the obvious '.'
        search_dirs = ['anim', 'misc', 'environment', 'item']
        search_zip_paths = [filename,] + [d + '/' + filename for d in search_dirs]
        def search_dir(base):
            """Search the given base dir for filename, in search_dirs."""
            for path in [os.path.join(base, d, filename) for d in ['',] + search_dirs]:
                if os.path.isfile(path):
                    return path
            return None

        # we've sucessfully loaded something from here before, so let's quickly try
        # this before searching again
        if self.jar is not None:
            for jarfilename in search_zip_paths:
                try:
                    self.jar.getinfo(jarfilename)
                    if verbose: logging.info("Found (cached) %s in '%s'", jarfilename, self.jarpath)
                    return self.jar.open(jarfilename)
                except (KeyError, IOError), e:
                    pass

        # A texture path was given on the command line. Search this location
        # for the file first.
        if self.local_path:
            if os.path.isdir(self.local_path):
                path = search_dir(self.local_path)
                if path:
                    if verbose: logging.info("Found %s in '%s'", filename, path)
                    return open(path, mode)
            elif os.path.isfile(self.local_path):
                # Must be a resource pack. Look for the requested file within
                # it.
                try:
                    pack = zipfile.ZipFile(self.local_path)
                    for packfilename in search_zip_paths:
                        try:
                            # pack.getinfo() will raise KeyError if the file is
                            # not found.
                            pack.getinfo(packfilename)
                            if verbose: logging.info("Found %s in '%s'", packfilename, self.local_path)
                            return pack.open(packfilename)
                        except (KeyError, IOError):
                            pass
                except (zipfile.BadZipfile, IOError):
                    pass

        # If we haven't returned at this point, then the requested file was NOT
        # found in the user-specified texture path or resource pack.
        if verbose: logging.info("Did not find the file in specified texture path")

        # Look in the location of the overviewer executable for the given path
        programdir = util.get_program_path()
        path = search_dir(programdir)
        if path:
            if verbose: logging.info("Found %s in '%s'", filename, path)
            return open(path, mode)

        if sys.platform.startswith("darwin"):
            path = search_dir("/Applications/Minecraft")
            if path:
                if verbose: logging.info("Found %s in '%s'", filename, path)
                return open(path, mode)

        if verbose: logging.info("Did not find the file in overviewer executable directory")
        if verbose: logging.info("Looking for installed minecraft jar files...")

        # Find an installed minecraft client jar and look in it for the texture
        # file we need.
        versiondir = None
        if "APPDATA" in os.environ and sys.platform.startswith("win"):
            versiondir = os.path.join(os.environ['APPDATA'], ".minecraft", "versions")
        elif "HOME" in os.environ:
            # For linux:
            versiondir = os.path.join(os.environ['HOME'], ".minecraft", "versions")
            if not os.path.exists(versiondir) and sys.platform.startswith("darwin"):
                # For Mac:
                versiondir = os.path.join(os.environ['HOME'], "Library",
                    "Application Support", "minecraft", "versions")

        try:
            versions = os.listdir(versiondir)
            if verbose: logging.info("Found these versions: {0}".format(versions))
        except OSError:
            # Directory doesn't exist? Ignore it. It will find no versions and
            # fall through the checks below to the error at the bottom of the
            # method.
            versions = []

        most_recent_version = [0,0,0]
        for version in versions:
            # Look for the latest non-snapshot that is at least 1.6. This
            # version is only compatible with >=1.6, and we cannot in general
            # tell if a snapshot is more or less recent than a release.

            # Allow two component names such as "1.6" and three component names
            # such as "1.6.1"
            if version.count(".") not in (1,2):
                continue
            try:
                versionparts = [int(x) for x in version.split(".")]
            except ValueError:
                continue

            if versionparts < [1,6]:
                continue

            if versionparts > most_recent_version:
                most_recent_version = versionparts

        if most_recent_version != [0,0,0]:
            if verbose: logging.info("Most recent version >=1.6.0: {0}. Searching it for the file...".format(most_recent_version))

            jarname = ".".join(str(x) for x in most_recent_version)
            jarpath = os.path.join(versiondir, jarname, jarname + ".jar")

            if os.path.isfile(jarpath):
                jar = zipfile.ZipFile(jarpath)
                for jarfilename in search_zip_paths:
                    try:
                        jar.getinfo(jarfilename)
                        if verbose: logging.info("Found %s in '%s'", jarfilename, jarpath)
                        self.jar, self.jarpath = jar, jarpath
                        return jar.open(jarfilename)
                    except (KeyError, IOError), e:
                        pass

            if verbose: logging.info("Did not find file {0} in jar {1}".format(filename, jarpath))
        else:
            if verbose: logging.info("Did not find any non-snapshot minecraft jars >=1.6.0")

        # Last ditch effort: look for the file is stored in with the overviewer
        # installation. We include a few files that aren't included with Minecraft
        # textures. This used to be for things such as water and lava, since
        # they were generated by the game and not stored as images. Nowdays I
        # believe that's not true, but we still have a few files distributed
        # with overviewer.
        if verbose: logging.info("Looking for texture in overviewer/data/textures")
        path = search_dir(os.path.join(programdir, "overviewer", "data", "textures"))
        if path:
            if verbose: logging.info("Found %s in '%s'", filename, path)
            return open(path, mode)
        elif hasattr(sys, "frozen") or imp.is_frozen("__main__"):
            # windows special case, when the package dir doesn't exist
            path = search_dir(os.path.join(programdir, "textures"))
            if path:
                if verbose: logging.info("Found %s in '%s'", filename, path)
                return open(path, mode)

        raise TextureException("Could not find the textures while searching for '{0}'. Try specifying the 'texturepath' option in your config file.\nSet it to the path to a Minecraft Resource pack.\nAlternately, install the Minecraft client (which includes textures)\nAlso see <http://docs.overviewer.org/en/latest/running/#installing-the-textures>\n(Remember, this version of Overviewer requires a 1.6-compatible resource pack)\n(Also note that I won't automatically use snapshots; you'll have to use the texturepath option to use a snapshot jar)".format(filename))

    def load(self, filename):
        """Returns an overviewer.oil.Image object from the given texture
        name
        
        """
        # Textures may be animated, this method un-animates them
        if filename in self.cache:
            return self.cache[filename]
        
        img = Image.load(self._find_file(filename))
        w, h = img.size
        if w != h:
            dest = Image(w, w)
            dest.composite(img, 255, 0, 0)
            img = dest
        
        self.cache[filename] = img
        return img
