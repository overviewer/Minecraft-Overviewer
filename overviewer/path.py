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
    if hasattr(sys, "frozen") or imp.is_frozen("__main__"):
        return os.path.join(os.path.dirname(sys.executable), *args)
    else:
        try:
            # normally, we're in ./overviewer/util.py
            # we want ./overviewer/data/...
            return os.path.join(os.path.dirname(__file__), "data", *args)
        except NameError:
            return os.path.join(os.path.dirname(sys.argv[0]), *args)
