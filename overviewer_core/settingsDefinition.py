# This file describes the format of the config file. Each item defined in this
# module is expected to appear in the described format in a valid config file.
# The only difference is, instead of actual values for the settings, values are
# Setting objects which define how to validate a value as correct, and whether
# the value is required or not.

# Settings objects have this signature:
# Setting(required, validator, default)

# required
#   a boolean indicating that this value is required. A required setting will
#   always exist in a validated config. This option only has effect in the
#   event that a user doesn't provide a value and the default is None. In this
#   case, a required setting will raise an error. Otherwise, the situation will
#   result in the setting being omitted from the config with no error.

#   (If it wasn't obvious: a required setting does NOT mean that the user is
#   required to specify it, just that the setting is required to be set for the
#   operation of the program, either by the user or by using the default)

# validator
#   a callable that takes the provided value and returns a cleaned/normalized
#   value to replace it with. It should raise a ValidationException if there is
#   a problem parsing or validating the value given.

# default
#   This is used in the event that the user does not provide a value.  In this
#   case, the default value is passed into the validator just the same. If
#   default is None, then depending on the value of required, it is either an
#   error to omit this setting or the setting is skipped entirely and will not
#   appear in the resulting parsed options.

# The signature for validator callables is validator(value_given). Remember
# that the default is passed in as value_given if the user did not provide a
# value.

# This file doesn't specify the format or even the type of the setting values,
# it is up to the validators to ensure the values passed in are the right type,
# either by coercion or by raising an error.

# Oh, one other thing: For top level values whose required attribute is True,
# the default value is set initially, before the config file is parsed, and is
# available during the execution of the config file. This way, container types
# can be initialized and then appended/added to when the config file is parsed.

from settingsValidators import *
import util
from observer import ProgressBarObserver, LoggingObserver, JSObserver
from optimizeimages import pngnq, optipng, pngcrush
import platform
import sys

# renders is a dictionary mapping strings to dicts. These dicts describe the
# configuration for that render. Therefore, the validator for 'renders' is set
# to a dict validator configured to validate keys as strings and values as...

# values are set to validate as a "configdict", which is a dict mapping a set
# of strings to some value. the make_configdictvalidator() function creates a
# validator to use here configured with the given set of keys and Setting
# objects with their respective validators.

# config file.
renders = Setting(required=True, default=util.OrderedDict(),
        validator=make_dictValidator(validateStr, make_configDictValidator(
        {
            "world": Setting(required=True, validator=validateStr, default=None),
            "dimension": Setting(required=True, validator=validateDimension, default="default"),
            "title": Setting(required=True, validator=validateStr, default=None),
            "rendermode": Setting(required=True, validator=validateRenderMode, default='normal'),
            "northdirection": Setting(required=True, validator=validateNorthDirection, default=0),
            "forcerender": Setting(required=False, validator=validateBool, default=None),
            "imgformat": Setting(required=True, validator=validateImgFormat, default="png"),
            "imgquality": Setting(required=False, validator=validateImgQuality, default=95),
            "bgcolor": Setting(required=True, validator=validateBGColor, default="1a1a1a"),
            "defaultzoom": Setting(required=True, validator=validateDefaultZoom, default=1),
            "optimizeimg": Setting(required=True, validator=validateOptImg, default=[]),
            "nomarkers": Setting(required=False, validator=validateBool, default=None),
            "texturepath": Setting(required=False, validator=validateTexturePath, default=None),
            "renderchecks": Setting(required=False, validator=validateInt, default=None),
            "rerenderprob": Setting(required=True, validator=validateRerenderprob, default=0),
            "crop": Setting(required=False, validator=validateCrop, default=None),
            "changelist": Setting(required=False, validator=validateStr, default=None),
            "markers": Setting(required=False, validator=validateMarkers, default=[]),
            "overlay": Setting(required=False, validator=validateOverlays, default=[]),
            "showspawn": Setting(required=False, validator=validateBool, default=True),
            "base": Setting(required=False, validator=validateStr, default=""),
            "poititle": Setting(required=False, validator=validateStr, default="Markers"),
            "customwebassets": Setting(required=False, validator=validateWebAssetsPath, default=None),
            "maxzoom": Setting(required=False, validator=validateInt, default=None),
            "minzoom": Setting(required=False, validator=validateInt, default=0),
            "manualpois": Setting(required=False, validator=validateManualPOIs, default=[]),
            "showlocationmarker": Setting(required=False, validator=validateBool, default=True),
            # Remove this eventually (once people update their configs)
            "worldname": Setting(required=False, default=None,
                validator=error("The option 'worldname' is now called 'world'. Please update your config files")),
        }
        )))

# The worlds dict, mapping world names to world paths
worlds = Setting(required=True, validator=make_dictValidator(validateStr, validateWorldPath), default=util.OrderedDict())

outputdir = Setting(required=True, validator=validateOutputDir, default=None)

processes = Setting(required=True, validator=int, default=-1)

# TODO clean up this ugly in sys.argv hack
if platform.system() == 'Windows' or not sys.stdout.isatty() or "--simple" in sys.argv:
    obs = LoggingObserver()
else:
    obs = ProgressBarObserver(fd=sys.stdout)

observer = Setting(required=True, validator=validateObserver, default=obs)
