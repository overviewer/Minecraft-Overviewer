import optparse
import sys
import os.path
import logging

import settingsDefinition
import settingsValidators

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
        for settingname in settingsDefinition.__all__:
            setting = getattr(settingsDefinition, settingname)
            assert isinstance(setting, settingsValidators.Setting)

            self._settings[settingname] = setting
            
            if not setting.required:
                self._config_state[settingname] = setting.default

    def set_config_item(self, itemname, itemvalue):
        self._config_state[itemname] = itemvalue

    def parse(self, settings_file):
        """Reads in the named file and parses it, storing the results in an
        internal state awating to be validated and returned upon call to
        get_render_settings()

        """
        if not os.path.exists(settings_file) and not os.path.isfile(settings_file):
            raise ValueError("bad settings file")

        # The global environment should be the rendermode module, so the config
        # file has access to those resources.
        import rendermodes

        try:
            execfile(settings_file, rendermodes.__dict__, self._config_state)
        
        except NameError, ex:
            logging.exception("Error parsing %s.  Please check the trackback for more info" % settings_file)
            sys.exit(1)
        except SyntaxError, ex:
            logging.exception("Error parsing %s.  Please check the trackback for more info" % self.settings_file)
            sys.exit(1)


    def get_validated_config(self):
        """Validate and return the configuration"""
        # Okay, this is okay, isn't it? We're going to create the validation
        # routine right here, right now. I hope this works!
        validator = settingsValidators.make_configdictvalidator(self._settings)
        # Woah. What just happened? No. WAIT, WHAT ARE YOU...
        validated_config = validator(self._config_state)
        # WHAT HAVE YOU DONE?
        return validated_config
        # WHAT HAVE YOU DOOOOOOOOOOONE????
