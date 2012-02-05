# This file describes the format of the config file. Each item defined in this
# module is expected to appear in the same format in a settings file. The only
# difference is, instead of actual values for the settings, one is to use a
# Setting object. Here is its signature:

# Setting(required, validator, default)

# required is a boolean indicating the user is required to provide this
# setting. In this case, default is unused and can be set to anything (None is
# a good choice).

# validator is a callable that takes the provided value and returns a
# cleaned/normalized value to use. It should raise an exception if there is a
# problem parsing or validating the value given.

# default is used instead of the user-provided value in the event that required
# is false. It is passed into the validator just the same. If default is None
# and required is False, then that value is skipped entirely and will not
# appear in the resulting parsed options.

# The signature for validators is validator(value_given). Remember that the
# default is passed in as value_given in the event that required is False and
# default is not None.

# This file doesn't specify the format or even the type of the setting values,
# that is up to the validators used.

from settingsValidators import *

# This is the export list for this module. It defines which items defined in
# this module are recognized by the config parser. Don't forget to update this
# if you add new items!
__all__ = ['render', 'world', 'outputdir']

# render is a dictionary mapping names to dicts describing the configuration
# for that render. It is therefore set to a settings object with a dict
# validator configured to validate keys as strings and values as...  values are
# set to validate as a "configdict", which is a dict mapping a set of strings
# to some value. the make_configdictvalidator function creates a validator to
# use here configured with the given set of keys and Setting objects with their
# respective validators.

# Perhaps unintuitively, this is set to required=False. Of course, if no
# renders are specified, this is an error. However, this is caught later on in
# the code, and it also lets an empty dict get defined beforehand for the
# config file.
render = Setting(required=False, default={},
        validator=dictValidator(validateStr, make_configdictvalidator(
        {
            "worldname": Setting(required=True, validator=validateStr, default=None),
            "dimension": Setting(required=False, validator=validateDimension, default="default"),
            "title": Setting(required=True, validator=validateStr, default=None),
            "rendermode": Setting(required=False, validator=validateRenderMode, default=None),
            "northdirection": Setting(required=False, validator=validateNorthDirection, default=0),
            "renderrange": Setting(required=False, validator=validateRenderRange, default=None),
            "forcerender": Setting(required=False, validator=validateBool, default=None),
            "stochasticrender": Setting(required=False, validator=validateStochastic, default=None),
            "imgformat": Setting(required=False, validator=validateImgFormat, default="png"),
            "imgquality": Setting(required=False, validator=validateImgQuality, default=None),
            "bgcolor": Setting(required=False, validator=validateBGColor, default="1a1a1a"),
            "optimizeimg": Setting(required=False, validator=validateOptImg, default=0),
            "nomarkers": Setting(required=False, validator=validateBool, default=None),
            "texturepath": Setting(required=False, validator=validateTexturePath, default=None),
            "renderchecks": Setting(required=False, validator=validateInt, default=0),
            "rerenderprob": Setting(required=False, validator=validateFloat, default=0),
        }
        )))

# The world dict, mapping world names to world paths
world = Setting(required=False, validator=dictValidator(validateStr, validateWorldPath), default={})

outputdir = Setting(required=True, validator=validateOutputDir, default=None)

