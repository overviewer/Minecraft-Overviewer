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

import imp
import os.path
import sys

def get_data_path(*args):
    """Create a path inside overviewer/data/, wherever that ends up when
    installed.

    """
    
    if hasattr(sys, "frozen") or imp.is_frozen("__main__"):
        return os.path.join(os.path.dirname(sys.executable), *args)
    else:
        try:
            # normally, we're in ./overviewer/util.py
            # we want ./overviewer/data/...
            return os.path.join(os.path.dirname(__file__), "data", *args)
        except NameError:
            return os.path.join(os.path.dirname(sys.argv[0]), *args)

class InstalledMinecraftError(Exception):
    pass

def get_minecraft_path(*args):
    """Look in various system-specific places for an installed minecraft,
    then construct a path off that location if found. Otherwise, raise
    InstalledMinecraftError.

    """
    
    def generator():
        if 'APPDATA' in os.environ and sys.platform.startswith('win'):
            yield os.path.join(os.environ['APPDATA'], '.minecraft')
        if 'HOME' in os.environ:
            if sys.platform.startswith('darwin'):
                yield os.path.join(os.environ['HOME'], 'Library', 'Application Support', 'minecraft')
            yield os.path.join(os.environ['HOME'], '.minecraft')
    
    paths = list(generator())
    for path in paths:
        if os.path.exists(path):
            return os.path.join(path, *args)
    
    raise InstalledMinecraftError("Could not find an installed minecraft in: " + " or ".join(paths))
