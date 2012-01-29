# see settingsDefinition.py
import os
import os.path

import rendermodes
from world import UPPER_LEFT, UPPER_RIGHT, LOWER_LEFT, LOWER_RIGHT

class ValidationException(Exception):
    pass

def validateWorldPath(name, **kwargs):
    world = kwargs.get('world', dict())
    if name not in world.keys():
        raise ValidationException("bad world name")
    abs_path = os.path.abspath(world[name])
    if not os.path.exists(os.path.join(abs_path, "level.dat")):
        raise ValidationException("No level.dat file in %r.  Check the path for world %r" % (abs_path, name))
    return abs_path


def validateRenderMode(mode, **kwargs):
    # make sure that mode is a list of things that are all rendermode primative
    if type(mode) != list:
        raise ValidationException("%r is not a valid list of rendermodes.  It should be a list"% mode)
    for m in mode:
        if not isinstance(m, rendermodes.RenderPrimitive):
            raise ValidationException("%r is not a valid rendermode primitive." % m)


    return mode

def validateNorthDirection(direction, **kwargs):
    # normalize to integers
    intdir = 0 #default
    if type(direction) == int:
        intdir = direction
    else:
        if direction == "upper-left": intdir = UPPER_LEFT
        if direction == "upper-right": intdir = UPPER_RIGHT
        if direction == "lower-right": intdir = LOWER_RIGHT
        if direction == "lower-left": intdir = LOWER_LEFT
    if intdir < 0 or intdir > 3:
        raise ValidationException("%r is not a valid north direction" % direction)
    return intdir

def validateRenderRange(r, **kwargs):
    raise NotImplementedError("render range")

def validateStochastic(s, **kwargs):
    val = float(s)
    if val < 0 or val > 1:
        raise ValidationException("%r is not a valid stochastic value.  Should be between 0.0 and 1.0" % s)
    return val

def validateImgFormat(fmt, **kwargs):
    if fmt not in ("png", "jpg", "jpeg"):
        raise ValidationException("%r is not a valid image format" % fmt)
    if fmt == "jpeg": fmt = "jpg"
    return fmt

def validateImgQuality(qual, **kwargs):
    intqual = int(qual)
    if (intqual < 0 or intqual > 100):
        raise ValidationException("%r is not a valid image quality" % intqual)
    return intqual

def validateBGColor(color, **kwargs):
    """BG color must be an HTML color, with an option leading # (hash symbol)
    returns an (r,b,g) 3-tuple  
    """
    if type(color) == str:
        if color[0] != "#":
            color = "#" + color
        if len(color) != 7:
            raise ValidationException("%r is not a valid color.  Expected HTML color syntax (i.e. #RRGGBB)" % color)
        try:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            return (r,g,b,0)
        except ValueError:
            raise ValidationException("%r is not a valid color.  Expected HTML color syntax (i.e. #RRGGBB)" % color)
    elif type(color) == tuple:
        if len(color) != 4:
            raise ValidationException("%r is not a valid color.  Expected a 4-tuple" % (color,))
        return color


def validateOptImg(opt, **kwargs):
    return bool(opt)

def validateTexturePath(path, **kwargs):
    # Expand user dir in directories strings
    path = os.path.expanduser(path)
    # TODO assert this path exists?


def validateBool(b, **kwargs):
    return bool(b)

def validateFloat(f, **kwargs):
    return float(f)

def validateInt(i, **kwargs):
    return int(i)

def validateStr(s, **kwargs):
    return str(s)

def validateDimension(d, **kwargs):
    if d in ["nether", "overworld", "end", "default"]:
        return d
    raise ValidationException("%r is not a valid dimension" % d)

def validateOutputDir(d, **kwargs):
    return os.path.abs(d)
