import unittest

from overviewer_core import configParser
from overviewer_core.settingsValidators import ValidationException

class SettingsTest(unittest.TestCase):
    
    def test_missing(self):
        "Validates that a non-existant settings.py causes an exception"
        self.assertRaises(ValueError, configParser.MultiWorldParser, "doesnotexist.py")
    def test_existing_file(self):
        s = configParser.MultiWorldParser("test/data/settings/settings_test_1.py")
        s.parse()
        s.validate()
        things = s.get_render_things()
        # no exceptions so far.  that's good
        self.assertEquals(things['world']['bgcolor'], (26,26,26,0))
        self.assertEquals(things['otherworld']['bgcolor'], (255,255,255,0))

    def test_rendermode_validation(self):
        s = configParser.MultiWorldParser("test/data/settings/settings_test_rendermode.py")
        s.parse()

        self.assertRaises(ValidationException,s.validate)

    def test_bgcolor_validation(self):
        s = configParser.MultiWorldParser("test/data/settings/settings_test_bgcolor.py")
        s.parse()

        self.assertRaises(ValidationException, s.validate)


if __name__ == "__main__":
    unittest.main()
