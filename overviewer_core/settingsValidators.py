# see settingsDefinition.py
import os
import os.path
from collections import namedtuple

import rendermodes
from world import UPPER_LEFT, UPPER_RIGHT, LOWER_LEFT, LOWER_RIGHT

class ValidationException(Exception):
    pass

Setting = namedtuple("Setting", [
    'required',
    'validator',
    'default',
    ])

def validateWorldPath(worldpath):
    abs_path = os.path.abspath(worldpath)
    if not os.path.exists(os.path.join(abs_path, "level.dat")):
        raise ValidationException("No level.dat file in %r. Are you sure you have the right path?" % (abs_path,))
    return abs_path


def validateRenderMode(mode):
    # make sure that mode is a list of things that are all rendermode primative
    if isinstance(mode, str):
        # Try and find an item named "mode" in the rendermodes module
        try:
            mode = getattr(rendermodes, mode)
        except AttributeError:
            raise ValidationException("You must specify a valid rendermode, not '%s'. See the docs for valid rendermodes." % mode)

    if isinstance(mode, rendermodes.RenderPrimitive):
        mode = [mode]
    
    if not isinstance(mode, list):
        raise ValidationException("%r is not a valid list of rendermodes.  It should be a list"% mode)

    for m in mode:
        if not isinstance(m, rendermodes.RenderPrimitive):
            raise ValidationException("%r is not a valid rendermode primitive." % m)


    return mode

def validateNorthDirection(direction):
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


def validateOptImg(opt):
    return bool(opt)

def validateTexturePath(path):
    # Expand user dir in directories strings
    path = os.path.expanduser(path)
    # TODO assert this path exists?
    return path


def validateBool(b):
    return bool(b)

def validateFloat(f):
    return float(f)

def validateInt(i):
    return int(i)

def validateStr(s):
    return str(s)

def validateDimension(d):
    if d in ["nether", "overworld", "end", "default"]:
        return d
    raise ValidationException("%r is not a valid dimension" % d)

def validateOutputDir(d):
    if not d.strip():
        raise ValidationException("You must specify a valid output directory")
    return os.path.abspath(d)

def make_dictValidator(keyvalidator, valuevalidator):
    """Compose and return a dict validator -- a validator that validates each
    key and value in a dictionary.

    The arguments are the validator function to use for the keys, and the
    validator function to use for the values.
    
    """
    def v(d):
        newd = {}
        for key, value in d.iteritems():
            newd[keyvalidator(key)] = valuevalidator(value)
        return newd
    return v

def make_configDictValidator(config, ignore_undefined=False):
    """Okay, stay with me here, this may get confusing. This function returns a
    validator that validates a "configdict". This is a term I just made up to
    refer to a dict that holds config information: keys are strings and values
    are whatever that config value is. The argument to /this/ function is a
    dictionary defining the valid keys for the configdict. It therefore maps
    those key names to Setting objects. When the returned validator function
    validates, it calls all the appropriate validators for each configdict
    setting

    I hope that makes sense.

    ignore_undefined, if True, will ignore any items in the dict to be
    validated which don't have a corresponding definition in the config.
    Otherwise, undefined entries will raise an error.

    """
    def configDictValidator(d):
        newdict = {}
        for configkey, configsetting in config.iteritems():
            if configkey in d:
                # This key /was/ specified in the user's dict. Make sure it validates.
                newdict[configkey] = configsetting.validator(d[configkey])
            elif configsetting.default is not None:
                # There is a default, use that instead
                newdict[configkey] = configsetting.validator(configsetting.default)
            elif configsetting.required:
                # The user did not give us this key, there is no default, AND
                # it's required. This is an error.
                raise ValidationException("Required key '%s' was not specified. You must give a value for this setting" % configkey)

        # Now that all the defined keys have been accounted for, check to make
        # sure any unauthorized keys were not specified.
        if not ignore_undefined:
            for key in d.iterkeys():
                if key not in config:
                    raise ValidationException("'%s' is not a configuration item" % key)
        return newdict

    return configDictValidator
