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
import subprocess
import shlex

pngcrush = "pngcrush"
optipng = "optipng"
advdef = "advdef"

def check_programs(level):
    path = os.environ.get("PATH").split(os.pathsep)

    for prog,l in [(pngcrush,1), (optipng,2), (advdef,2)]:
        if l <= level:
            result = filter(lambda x: os.path.exists(os.path.join(x, prog)), path)
            if len(result) == 0:
                raise Exception("Optimization prog %s for level %d not found!" % (prog, l))

def optimize_image(imgpath, imgformat, optimizeimg):
    if imgformat == 'png':
        if optimizeimg >= 1:
            # we can't do an atomic replace here because windows is terrible
            # so instead, we make temp files, delete the old ones, and rename
            # the temp files. go windows!
            subprocess.Popen(shlex.split(pngcrush +" " + imgpath + " " + imgpath + ".tmp"),
                stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
            os.remove(imgpath)
            os.rename(imgpath+".tmp", imgpath)

        if optimizeimg >= 2:
            subprocess.Popen(shlex.split(optipng + " " + imgpath), stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE).communicate()[0]
            subprocess.Popen(shlex.split(advdef + " -z4 " + imgpath), stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE).communicate()[0]

