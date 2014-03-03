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

class Optimizer:
    binaryname = ""

    def __init__(self):
        raise NotImplementedError("I can't let you do that, Dave.")

    def optimize(self, img):
        raise NotImplementedError("I can't let you do that, Dave.")
    
    def fire_and_forget(self, args):
        subprocess.Popen(args, stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]

class NonAtomicOptimizer(Optimizer):
    def cleanup(self, img):
        os.rename(img + ".tmp", img)

    def fire_and_forget(self, args, img):
        subprocess.Popen(args, stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
        self.cleanup(img)

class PNGOptimizer:
    def __init__(self):
        raise NotImplementedError("I can't let you do that, Dave.")

class JPEGOptimizer:
    def __init__(self):
        raise NotImplementedError("I can't let you do that, Dave.")

class pngnq(NonAtomicOptimizer, PNGOptimizer):
    binaryname = "pngnq"

    def __init__(self, sampling=3, dither="n"):
        if sampling < 1 or sampling > 10:
            raise Exception("Invalid sampling value '%d' for pngnq!" % sampling)

        if dither not in ["n", "f"]:
            raise Exception("Invalid dither method '%s' for pngnq!" % dither)

        self.sampling = sampling
        self.dither = dither
    
    def optimize(self, img):
        if img.endswith(".tmp"):
            extension = ".tmp"
        else:
            extension = ".png.tmp"

        NonAtomicOptimizer.fire_and_forget(self, [self.binaryname, "-s", str(self.sampling),
                                "-Q", self.dither, "-e", extension, img], img)

class pngcrush(NonAtomicOptimizer, PNGOptimizer):
    binaryname = "pngcrush"
    # really can't be bothered to add some interface for all
    # the pngcrush options, it sucks anyway
    def __init__(self, brute=False):
        self.brute = brute
        
    def optimize(self, img):
        args = [self.binaryname, img, img + ".tmp"]
        if self.brute == True:  # Was the user an idiot?
            args.insert(1, "-brute")

        NonAtomicOptimizer.fire_and_forget(self, args, img)

class optipng(Optimizer, PNGOptimizer):
    binaryname = "optipng"

    def __init__(self, olevel=2):
        self.olevel = olevel
    
    def optimize(self, img):
        Optimizer.fire_and_forget(self, [self.binaryname, "-o" + str(self.olevel), "-quiet", img])
        

def check_programs(optimizers):
    path = os.environ.get("PATH").split(os.pathsep)
    
    def exists_in_path(prog):
        result = filter(lambda x: os.path.exists(os.path.join(x, prog)), path)
        return len(result) != 0
    
    for opt in optimizers:
        if (not exists_in_path(opt.binaryname)) and (not exists_in_path(opt.binaryname + ".exe")):
            raise Exception("Optimization program '%s' was not found!" % opt.binaryname)

def optimize_image(imgpath, imgformat, optimizers):
        for opt in optimizers:
            if imgformat == 'png':
                if isinstance(opt, PNGOptimizer):
                    opt.optimize(imgpath)
            elif imgformat == 'jpg':
                if isinstance(opt, JPEGOptimizer):
                    opt.optimize(imgpath)
