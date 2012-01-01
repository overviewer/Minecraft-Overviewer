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


render = {
    "world_path": dict(required=True, validator=validatePath),
    "rendermode": dict(required=False, validator=validateRenderMode),
    "north-direction": dict(required=False, validator=validateNorthDirection),
    "render-range": dict(required=False, validator=validateRenderRange),
    "force-render": dict(required=False, validator=bool),
    "stochastic-render": dict(required=False, validator=validateStochastic),
    "imgformat": dict(required=False, validator=validateImgFormat),
    "imgquality": dict(required=False, validator=validateImgQuality),
    "bg-color": dict(required=False, validator=validateBGColor),
    "optimize-img": dict(required=False, validator=validateOptImg),
    "no-markers": dict(required=False, validator=bool),
    "texture-path": dict(required=False, validator=validateTexturePath),
    }

