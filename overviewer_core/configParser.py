import optparse
import sys
import os.path
import logging
import traceback

import settingsDefinition
import settingsValidators

class MissingConfigException(Exception):
    "To be thrown when the config file can't be found"
    pass

class MultiWorldParser(object):
    """A class that is used to parse a settings.py file.
    
    This class's job is to compile and validate the configuration settings for
    a set of renders. It can read in configuration from the given file with the
    parse() method, and one can set configuration options directly with the
    set_config_item() method.

    get_validated_config() validates and returns the validated config

    """

    def __init__(self):
        """Initialize this parser object"""
        # This maps config names to their values
        self._config_state = {}

        # Scan the settings definition and build the config state heirarchy.
        # Also go ahead and set default values for non-required settings.
        # This maps setting names to their values as given in
        # settingsDefinition.py
        self._settings = {}
        for settingname in dir(settingsDefinition):
            setting = getattr(settingsDefinition, settingname)
            if not isinstance(setting, settingsValidators.Setting):
                continue

            self._settings[settingname] = setting
            
            # Set top level defaults. This is intended to be for container
            # types, so they can initialize a config file with an empty
            # container (like a dict)
            if setting.required and setting.default is not None:
                self._config_state[settingname] = setting.default

    def set_config_item(self, itemname, itemvalue):
        self._config_state[itemname] = itemvalue

    def set_renders_default(self, settingname, newdefault):
        """This method sets the default for one of the settings of the "renders"
        dictionary. This is hard-coded to expect a "renders" setting in the
        settings definition, and for its validator to be a dictValidator with
        its valuevalidator to be a configDictValidator

        """
        # If the structure of settingsDefinitions changes, you'll need to change
        # this to find the proper place to find the settings dictionary
        render_settings = self._settings['renders'].validator.valuevalidator.config
        render_settings[settingname].default = newdefault

    def parse(self, settings_file):
        """Reads in the named file and parses it, storing the results in an
        internal state awating to be validated and returned upon call to
        get_render_settings()

        Attributes defined in the file that do not match any setting are then
        matched against the renderdict setting, and if it does match, is used as
        the default for that setting.

        """
        if not os.path.exists(settings_file) and not os.path.isfile(settings_file):
            raise MissingConfigException("The settings file you specified (%r) does not exist, or is not a file" % settings_file)

        # The global environment should be the rendermode module, so the config
        # file has access to those resources.
        import rendermodes

        try:
            execfile(settings_file, rendermodes.__dict__, self._config_state)
        
        except Exception, ex:
            if isinstance(ex, SyntaxError):
                logging.error("Syntax error parsing %s" %  settings_file)
                logging.error("The traceback below will tell you which line triggered the syntax error\n")
            elif isinstance(ex, NameError):
                logging.error("NameError parsing %s" %  settings_file)
                logging.error("The traceback below will tell you which line referenced the non-existent variable\n")
            else:
                logging.error("Error parsing %s" %  settings_file)
                logging.error("The traceback below will tell you which line triggered the error\n")

            # skip the execfile part of the traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            formatted_lines = traceback.format_exc().splitlines()
            print_rest = False
            lines = []
            for l in formatted_lines:
                if print_rest: lines.append(l)
                else:
                    if "execfile" in l: print_rest = True
            # on windows, our traceback as no 'execfile'.  in this case, print everything
            if print_rest: logging.error("Partial traceback:\n" + "\n".join(lines))
            else: logging.error("Partial traceback:\n" + "\n".join(formatted_lines))
            sys.exit(1)

        # At this point, make a pass through the file to possibly set global
        # render defaults
        render_settings = self._settings['renders'].validator.valuevalidator.config
        for key in self._config_state.iterkeys():
            if key not in self._settings:
                if key in render_settings:
                    setting = render_settings[key]
                    setting.default = self._config_state[key]


    def get_validated_config(self):
        """Validate and return the configuration. Raises a ValidationException
        if there was a problem validating the config.

        Could also raise a ValueError
        
        """
        # Okay, this is okay, isn't it? We're going to create the validation
        # routine right here, right now. I hope this works!
        validator = settingsValidators.make_configDictValidator(self._settings, ignore_undefined=True)
        # Woah. What just happened? No. WAIT, WHAT ARE YOU...
        validated_config = validator(self._config_state)
        # WHAT HAVE YOU DONE?
        return validated_config
        # WHAT HAVE YOU DOOOOOOOOOOONE????
