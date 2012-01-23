# This file defines all of the things than can 
# appear in a settings.py file, as well as the
# function that can validate them

# the validator should raise an exception if there
# is a problem parsing or validating the value.
# it should return the value to use (which gives
# the validator an oppertunity to cleanup/normalize/
# whaterver)

# if a setting is not required, the validator is 
# expected to return the default option

from settingsValidators import *


# note that all defaults go thought the validator

render = {
    "worldname": dict(required=True, validator=validateWorldPath, save_orig=True),
    "dimension": dict(required=False, validator=validateDimension, default="default"),
    "title": dict(required=True, validator=validateStr),
    "rendermode": dict(required=False, validator=validateRenderMode),
    "northdirection": dict(required=False, validator=validateNorthDirection, default=0),
    "renderrange": dict(required=False, validator=validateRenderRange),
    "forcerender": dict(required=False, validator=validateBool),
    "stochasticrender": dict(required=False, validator=validateStochastic),
    "imgformat": dict(required=False, validator=validateImgFormat, default="png"),
    "imgquality": dict(required=False, validator=validateImgQuality),
    "bgcolor": dict(required=False, validator=validateBGColor, default="1a1a1a"),
    "optimizeimg": dict(required=False, validator=validateOptImg, default=0),
    "nomarkers": dict(required=False, validator=validateBool),
    "texturepath": dict(required=False, validator=validateTexturePath),
    "renderchecks": dict(required=False, validator=validateInt, default=0),
    "rerenderprob": dict(required=False, validator=validateFloat, default=0),
    }

