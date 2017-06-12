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


class Optimizer:
    binaryname = ""
    binarynames = []

    def __init__(self):
        raise NotImplementedError("I can't let you do that, Dave.")

    def optimize(self, img):
        raise NotImplementedError("I can't let you do that, Dave.")

    def fire_and_forget(self, args):
        subprocess.check_call(args)

    def check_availability(self):
        path = os.environ.get("PATH").split(os.pathsep)

        def exists_in_path(prog):
            result = filter(lambda x: os.path.exists(os.path.join(x, prog)),
                            path)
            return len(result) != 0

        binaries = self.binarynames + [x + ".exe" for x in self.binarynames]
        for b in binaries:
            if (exists_in_path(b)):
                self.binaryname = b
                break
        else:
            raise Exception("Optimization programs '%s' were not found!" %
                            binaries)

    def is_crusher(self):
        """Should return True if the optimization is lossless, i.e. none of the
        actual image data will be changed."""
        raise NotImplementedError("I'm so abstract I can't even say whether "
                                  "I'm a crusher.")


class NonAtomicOptimizer(Optimizer):
    def cleanup(self, img):
        os.remove(img)
        os.rename(img + ".tmp", img)

    def fire_and_forget(self, args, img):
        subprocess.check_call(args)
        self.cleanup(img)


class PNGOptimizer:
    def __init__(self):
        raise NotImplementedError("I can't let you do that, Dave.")


class JPEGOptimizer:
    def __init__(self):
        raise NotImplementedError("I can't let you do that, Dave.")


class pngnq(NonAtomicOptimizer, PNGOptimizer):
    binarynames = ["pngnq-s9", "pngnq"]

    def __init__(self, sampling=3, dither="n"):
        if sampling < 1 or sampling > 10:
            raise Exception("Invalid sampling value '%d' for pngnq!" %
                            sampling)
        if dither not in ["n", "f"]:
            raise Exception("Invalid dither method '%s' for pngnq!" % dither)
        self.sampling = sampling
        self.dither = dither

    def optimize(self, img):
        if img.endswith(".tmp"):
            extension = ".tmp"
        else:
            extension = ".png.tmp"

        args = [self.binaryname, "-s", str(self.sampling), "-f", "-e",
                extension, img]
        # Workaround for poopbuntu 12.04 which ships an old broken pngnq
        if self.dither != "n":
            args.insert(1, "-Q")
            args.insert(2, self.dither)

        NonAtomicOptimizer.fire_and_forget(self, args, img)

    def is_crusher(self):
        return False


class pngcrush(NonAtomicOptimizer, PNGOptimizer):
    binarynames = ["pngcrush"]

    # really can't be bothered to add some interface for all
    # the pngcrush options, it sucks anyway
    def __init__(self, brute=False):
        self.brute = brute

    def optimize(self, img):
        args = [self.binaryname, img, img + ".tmp"]
        if self.brute:  # Was the user an idiot?
            args.insert(1, "-brute")

        NonAtomicOptimizer.fire_and_forget(self, args, img)

    def is_crusher(self):
        return True


class optipng(Optimizer, PNGOptimizer):
    binarynames = ["optipng"]

    def __init__(self, olevel=2):
        self.olevel = olevel

    def optimize(self, img):
        Optimizer.fire_and_forget(self, [self.binaryname, "-o" +
                                         str(self.olevel), "-quiet", img])

    def is_crusher(self):
        return True


class advpng(Optimizer, PNGOptimizer):
    binarynames = ["advpng"]
    crusher = True

    def __init__(self, olevel=3):
        self.olevel = olevel

    def optimize(self, img):
        Optimizer.fire_and_forget(self, [self.binaryname, "-z" +
                                         str(self.olevel), "-q", img])

    def is_crusher(self):
        return True


class jpegoptim(Optimizer, JPEGOptimizer):
    binarynames = ["jpegoptim"]
    crusher = True
    quality = None
    target_size = None

    def __init__(self, quality=None, target_size=None):
        if quality is not None:
            if quality < 0 or quality > 100:
                raise Exception("Invalid target quality %d for jpegoptim" %
                                quality)
            self.quality = quality

        if target_size is not None:
            self.target_size = target_size

    def optimize(self, img):
        args = [self.binaryname, "-q", "-p"]
        if self.quality is not None:
            args.append("-m" + str(self.quality))

        if self.target_size is not None:
            args.append("-S" + str(self.target_size))

        args.append(img)

        Optimizer.fire_and_forget(self, args)

    def is_crusher(self):
        # Technically, optimisation is lossless if input image quality
        # is below target quality, but this is irrelevant in this case
        if (self.quality is not None) or (self.target_size is not None):
            return False
        else:
            return True


class oxipng(Optimizer, PNGOptimizer):
    binarynames = ["oxipng"]

    def __init__(self, olevel=2, threads=1):
        if olevel > 6:
            raise Exception("olevel should be between 0 and 6 inclusive")
        if threads < 1:
            raise Exception("threads needs to be at least 1")
        self.olevel = olevel
        self.threads = threads

    def optimize(self, img):
        Optimizer.fire_and_forget(self, [self.binaryname, "-o" +
                                         str(self.olevel), "-q", "-t" +
                                         str(self.threads), img])

    def is_crusher(self):
        return True


def optimize_image(imgpath, imgformat, optimizers):
        for opt in optimizers:
            if imgformat == 'png':
                if isinstance(opt, PNGOptimizer):
                    opt.optimize(imgpath)
            elif imgformat == 'jpg':
                if isinstance(opt, JPEGOptimizer):
                    opt.optimize(imgpath)
