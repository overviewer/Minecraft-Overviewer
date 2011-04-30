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

"""
Misc utility routines used by multiple files that don't belong anywhere else
"""

import imp
import os
import os.path
import sys

def get_program_path():
    if hasattr(sys, "frozen") or imp.is_frozen("__main__"):
        return os.path.dirname(sys.executable)
    else:
        try:
            return os.path.dirname(__file__)
        except NameError:
            return os.path.dirname(sys.argv[0])



def findGitVersion():
    this_dir = get_program_path()
    if os.path.exists(os.path.join(this_dir,".git")):
        with open(os.path.join(this_dir,".git","HEAD")) as f:
            data = f.read().strip()
        if data.startswith("ref: "):
            if not os.path.exists(os.path.join(this_dir, ".git", data[5:])):
                return data
            with open(os.path.join(this_dir, ".git", data[5:])) as g:
                return g.read().strip()
        else:
            return data
    else:
        try:
            import overviewer_version
            return overviewer_version.VERSION
        except:
            return "unknown"
