# see settingsDefinition.py
import os
import os.path

class ValidationException(Exception):
    pass

def validatePath(path):
    if not os.path.exists(path):
        raise ValidationException("%r does not exist" % path)
    if not os.path.isdir(path):
        raise ValidationException("%r is not a directory" % path)
    return os.abspath(path)

def validateRenderMode(mode):
    # TODO get list of valid rendermodes
    raise NotImplementedError("validateRenderMode")

def validateNorthDirection(direction):
    # normalize to integers
    intdir = 0 #default
    if type(direction) == int:
        intdir = direction
    else:
        if direction == "upper-left": intdir = 0
        if direction == "upper-right": intdir = 1
        if direction == "lower-right": intdir = 2
        if direction == "lower-left": intdir = 3
    if intdir < o or intdir > 3:
        raise ValidationException("%r is not a valid north direction" % direction)
    return intdir

def validateRenderRange(r):
    raise NotImplementedError("render range")

def validateStochastic(s):
    val = float(s)
    if val < 0 or val > 1:
        raise ValidationException("%r is not a valid stochastic value.  Should be between 0.0 and 1.0" % s)
    return val

def validateImgFormat(fmt):
    if fmt not in ("png", "jpg", "jpeg"):
        raise ValidationException("%r is not a valid image format" % fmt)
    if fmt == "jpeg": fmt = "jpg"
    return fmt

def validateImgQuality(qual):
    intqual = int(qual)
    if (intqual < 0 or intqual > 100):
        raise ValidationException("%r is not a valid image quality" % intqual)
    return intqual

def validateBGColor(color):
    raise NotImplementedError("bg color")

def validateOptImg(opt):
    return bool(opt)



