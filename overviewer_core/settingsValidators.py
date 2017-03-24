# see settingsDefinition.py
import os
import os.path

import rendermodes
import util
from optimizeimages import Optimizer
from world import UPPER_LEFT, UPPER_RIGHT, LOWER_LEFT, LOWER_RIGHT
import logging

class ValidationException(Exception):
    pass

class Setting(object):
    __slots__ = ['required', 'validator', 'default']
    def __init__(self, required, validator, default):
        self.required = required
        self.validator = validator
        self.default = default

def expand_path(p):
    p = os.path.expanduser(p)
    p = os.path.expandvars(p)
    p = os.path.abspath(p)
    return p

def checkBadEscape(s):
    # If any of these weird characters are in the path, raise an exception
    # instead of fixing this should help us educate our users about pathslashes
    bad_escapes = ['\a', '\b', '\t', '\n', '\v', '\f', '\r']
    for b in bad_escapes:
        if b in s:
            raise ValueError("Invalid character %s in path.  Please use "
                             "forward slashes ('/').  Please see our docs for "
                             "more info." % repr(b))
    for c in range(10):
        if chr(c) in s:
            raise ValueError("Invalid character '\\%s' in path.  Please use forward slashes ('/').  Please see our docs for more info." % c)
    return s

def validateMarkers(filterlist):
    if type(filterlist) != list:
        raise ValidationException("Markers must specify a list of filters.  This has recently changed, so check the docs.")
    for x in filterlist:
        if type(x) != dict:
            raise ValidationException("Markers must specify a list of dictionaries.  This has recently changed, so check the docs.")
        if "name" not in x:
            raise ValidationException("Must define a name")
        if "filterFunction" not in x:
            raise ValidationException("Must define a filter function")
        if not callable(x['filterFunction']):
            raise ValidationException("%r must be a function"% x['filterFunction'])
    return filterlist

def validateOverlays(renderlist):
    if type(renderlist) != list:
        raise ValidationException("Overlay must specify a list of renders")
    for x in renderlist:
        if validateStr(x) == '':
            raise ValidationException("%r must be a string"% x)
    return renderlist

def validateWorldPath(worldpath):
    checkBadEscape(worldpath)
    abs_path = expand_path(worldpath)
    if not os.path.exists(os.path.join(abs_path, "level.dat")):
        raise ValidationException("No level.dat file in '%s'. Are you sure you have the right path?" % (abs_path,))
    return abs_path


def validateRenderMode(mode):
    # make sure that mode is a list of things that are all rendermode primative
    if isinstance(mode, str):
        # Try and find an item named "mode" in the rendermodes module
        mode = mode.lower().replace("-","_")
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
    elif isinstance(direction, str):
        direction = direction.lower().replace("-","").replace("_","")
        if direction == "upperleft": intdir = UPPER_LEFT
        elif direction == "upperright": intdir = UPPER_RIGHT
        elif direction == "lowerright": intdir = LOWER_RIGHT
        elif direction == "lowerleft": intdir = LOWER_LEFT
        else:
            raise ValidationException("'%s' is not a valid north direction" % direction)
    if intdir < 0 or intdir > 3:
        raise ValidationException("%r is not a valid north direction" % direction)
    return intdir

def validateRerenderprob(s):
    val = float(s)
    if val < 0 or val >= 1:
        raise ValidationException("%r is not a valid rerender probability value.  Should be between 0.0 and 1.0." % s)
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


def validateOptImg(optimizers):
    if isinstance(optimizers, (int, long)):
        from optimizeimages import pngcrush
        logging.warning("You're using a deprecated definition of optimizeimg. "\
                        "We'll do what you say for now, but please fix this as soon as possible.")
        optimizers = [pngcrush()]
    if not isinstance(optimizers, list):
        raise ValidationException("What you passed to optimizeimg is not a list. "\
                                  "Make sure you specify them like [foo()], with square brackets.")

    if optimizers:
        for opt, next_opt in zip(optimizers, optimizers[1:]) + [(optimizers[-1], None)]:
            if not isinstance(opt, Optimizer):
                raise ValidationException("Invalid Optimizer!")

            opt.check_availability()

            # Check whether the chaining is somewhat sane
            if next_opt:
                if opt.is_crusher() and not next_opt.is_crusher():
                    logging.warning("You're feeding a crushed output into an optimizer that does not crush. "\
                                    "This is most likely pointless, and wastes time.")

    return optimizers

def validateTexturePath(path):
    # Expand user dir in directories strings
    path = expand_path(path)
    if not os.path.exists(path):
        raise ValidationException("%r does not exist" % path)
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
    # returns (original, argument to get_type)
    
    # these are provided as arguments to RegionSet.get_type()
    pretty_names = {
        "nether": "DIM-1",
        "overworld": None,
        "end": "DIM1",
        "default": 0,
    }
    
    try:
        return (d, pretty_names[d])
    except KeyError:
        return (d, d)

def validateOutputDir(d):
    checkBadEscape(d)
    if not d.strip():
        raise ValidationException("You must specify a valid output directory")
    return expand_path(d)

def validateCrop(value):
    if not isinstance(value, list):
        value = [value]
        
    cropZones = []
    for zone in value:
        if not isinstance(zone, tuple) or len(zone) != 4:
            raise ValidationException("The value for the 'crop' setting must be an array of tuples of length 4")
        a, b, c, d = tuple(int(x) for x in zone)
    
        if a >= c:
            a, c = c, a
        if b >= d:
            b, d = d, b
        
        cropZones.append((a, b, c, d))
        
    return cropZones

def validateObserver(observer):
    if all(map(lambda m: hasattr(observer, m), ['start', 'add', 'update', 'finish'])):
        return observer
    else:
        raise ValidationException("%r does not look like an observer" % repr(observer))

def validateDefaultZoom(z):
    if z > 0:
        return int(z)
    else:
        raise ValidationException("The default zoom is set below 1")

def validateWebAssetsPath(p):
    try:
        validatePath(p)
    except ValidationException as e:
        raise ValidationException("Bad custom web assets path: %s" % e.message)

def validatePath(p):
    checkBadEscape(p)
    abs_path = expand_path(p)
    if not os.path.exists(abs_path):
        raise ValidationException("'%s' does not exist. Path initially given as '%s'" % (abs_path,p))

def validateManualPOIs(d):
    for poi in d:
        if not 'x' in poi or not 'y' in poi or not 'z' in poi or not 'id' in poi:
            raise ValidationException("Not all POIs have x/y/z coordinates or an id: %r" % poi)
    return d

def make_dictValidator(keyvalidator, valuevalidator):
    """Compose and return a dict validator -- a validator that validates each
    key and value in a dictionary.

    The arguments are the validator function to use for the keys, and the
    validator function to use for the values.

    """
    def v(d):
        newd = util.OrderedDict()
        for key, value in d.iteritems():
            newd[keyvalidator(key)] = valuevalidator(value)
        return newd
    # Put these objects as attributes of the function so they can be accessed
    # externally later if need be
    v.keyvalidator = keyvalidator
    v.valuevalidator = valuevalidator
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
        newdict = util.OrderedDict()

        # values are config keys that the user specified that don't match any
        # valid key
        # keys are the correct configuration key
        undefined_key_matches = {}

        # Go through the keys the user gave us and make sure they're all valid.
        for key in d.iterkeys():
            if key not in config:
                # Try to find a probable match
                match = _get_closest_match(key, config.iterkeys())
                if match and ignore_undefined:
                    # Save this for later. It only matters if this is a typo of
                    # some required key that wasn't specified. (If all required
                    # keys are specified, then this should be ignored)
                    undefined_key_matches[match] = key
                    newdict[key] = d[key]
                elif match:
                    raise ValidationException(
                            "'%s' is not a configuration item. Did you mean '%s'?"
                            % (key, match))
                elif not ignore_undefined:
                    raise ValidationException("'%s' is not a configuration item" % key)
                else:
                    # the key is to be ignored. Copy it as-is to the `newdict`
                    # to be returned. It shouldn't conflict, and may be used as
                    # a default value for a render configdict later on.
                    newdict[key] = d[key]

        # Iterate through the defined keys in the configuration (`config`),
        # checking each one to see if the user specified it (in `d`). Then
        # validate it and copy the result to `newdict`
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
                if configkey in undefined_key_matches:
                    raise ValidationException("Key '%s' is not a valid "
                    "configuration item. Did you mean '%s'?"
                            % (undefined_key_matches[configkey], configkey))
                else:
                    raise ValidationException("Required key '%s' was not "
                    "specified. You must give a value for this setting"
                    % configkey)

        return newdict
    # Put these objects as attributes of the function so they can be accessed
    # externally later if need be
    configDictValidator.config = config
    configDictValidator.ignore_undefined = ignore_undefined
    return configDictValidator

def error(errstr):
    def validator(_):
        raise ValidationException(errstr)
    return validator

# Activestate recipe 576874
def _levenshtein(s1, s2):
  l1 = len(s1)
  l2 = len(s2)

  matrix = [range(l1 + 1)] * (l2 + 1)
  for zz in range(l2 + 1):
    matrix[zz] = range(zz,zz + l1 + 1)
  for zz in range(0,l2):
    for sz in range(0,l1):
      if s1[sz] == s2[zz]:
        matrix[zz+1][sz+1] = min(matrix[zz+1][sz] + 1, matrix[zz][sz+1] + 1, matrix[zz][sz])
      else:
        matrix[zz+1][sz+1] = min(matrix[zz+1][sz] + 1, matrix[zz][sz+1] + 1, matrix[zz][sz] + 1)
  return matrix[l2][l1]

def _get_closest_match(s, keys):
    """Returns a probable match for the given key `s` out of the possible keys in
    `keys`. Returns None if no matches are very close.

    """
    # Adjust this. 3 is probably a good number, it's probably not a typo if the
    # distance is >3
    threshold = 3

    minmatch = None
    mindist = threshold+1

    for key in keys:
        d = _levenshtein(s, key)
        if d < mindist:
            minmatch = key
            mindist = d

    if mindist <= threshold:
        return minmatch
    return None
