# see settingsDefinition.py
import os
import os.path
from collections import namedtuple

import rendermodes
import util
from world import UPPER_LEFT, UPPER_RIGHT, LOWER_LEFT, LOWER_RIGHT

class ValidationException(Exception):
    pass

class Setting(object):
    __slots__ = ['required', 'validator', 'default']
    def __init__(self, required, validator, default):
        self.required = required
        self.validator = validator
        self.default = default

def checkBadEscape(s):
    fixed = False
    fixed_string = s
    if "\a" in fixed_string:
        fixed_string = s.replace("\a", r"\a")
        fixed = True
    if "\b" in fixed_string:
        fixed_string = s.replace("\b", r"\b")
        fixed = True
    if "\t" in fixed_string:
        fixed_string = s.replace("\t", r"\t")
        fixed = True
    if "\n" in fixed_string:
        fixed_string = s.replace("\n", r"\n")
        fixed = True
    if "\v" in fixed_string:
        fixed_string = s.replace("\v", r"\v")
        fixed = True
    if "\f" in fixed_string:
        fixed_string = s.replace("\f", r"\f")
        fixed = True
    if "\r" in fixed_string:
        fixed_string = s.replace("\r", r"\r")
        fixed = True
    return (fixed, fixed_string)

def validateMarkers(filterlist):
    if type(filterlist) != list:
        raise ValidationException("Markers must specify a list of filters")
    for x in filterlist:
        if not callable(x):
            raise ValidationException("%r must be a function"% x)
    return filterlist

def validateWorldPath(worldpath):
    _, worldpath = checkBadEscape(worldpath)
    abs_path = os.path.abspath(os.path.expanduser(worldpath))
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
    _, d = checkBadEscape(d)
    if not d.strip():
        raise ValidationException("You must specify a valid output directory")
    return os.path.abspath(d)

def validateCrop(value):
    if len(value) != 4:
        raise ValidationException("The value for the 'crop' setting must be a tuple of length 4")
    value = tuple(int(x) for x in value)
    if value[0] >= value[2]:
        value[0],value[2] = value[2],value[0]
    if value[1] >= value[3]:
        value[1],value[3] = value[3],value[1]
    return value

def validateObserver(observer):
    if all(map(lambda m: hasattr(observer, m), ['start', 'add', 'update', 'finish'])):
        return observer
    else:
        raise ValidationException("%r does not look like an observer" % repr(observer))

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
