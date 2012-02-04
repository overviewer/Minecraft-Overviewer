import optparse
import sys
import os.path
import logging

import settingsDefinition

class MultiWorldParser(object):
    """A class that is used to parse a settings.py file.  It should replace
    ConfigOptionParser class above."""

    def __init__(self, settings):
        """Settings is a path to a settings.py file"""
        if not os.path.exists(settings) and not os.path.isfile(settings):
            raise ValueError("bad settings file")

        self.settings_file = settings

    def parse(self):
        # settingsDefinition.py defines what types of things you can put in a configfile
        # anything in settingsDefinitino that is a dict with a "type" key is a definiton
        defDicts = [x for x in dir(settingsDefinition) if type(getattr(settingsDefinition,x)) == dict and getattr(settingsDefinition,x).has_key("type") and x != "__builtins__"]

        glob = dict()
        for name in defDicts:
            d = getattr(settingsDefinition, name)
            glob[name] = d['type']()

        import rendermodes
        loc=dict()
        for thing in dir(rendermodes):
            thething = getattr(rendermodes, thing)
            if isinstance(thething, type) and issubclass(thething, rendermodes.RenderPrimitive):
                loc[thing] = thething

        try:
            execfile(self.settings_file, glob, loc)
            # delete the builtins, we don't need it
            del glob['__builtins__']
        
        except NameError, ex:
            import traceback
            traceback.print_exc()
            logging.error("Error parsing %s.  Please check the trackback above" % self.settings_file)
            sys.exit(1)
        except SyntaxError, ex:
            import traceback
            traceback.print_exc()
            tb = sys.exc_info()[2]
            #print tb.tb_frame.f_code.co_filename
            logging.error("Error parsing %s.  Please check the trackback above" % self.settings_file)
            sys.exit(1)


        for name in defDicts:
            setattr(self, name, glob[name])
            del glob[name]


        # seed with the Overviewer defaults, then update with the user defaults
        self.defaults = dict()
        for name in defDicts:
            d = getattr(settingsDefinition, name)
            if d['type'] == dict and d['valuetype'] == dict:
                for key in d['values']:
                    option = d['values'][key]
                    if option.has_key("default"):
                        self.defaults[key] = option["default"]

            
        self.defaults.update(glob)


    def validate(self):


        origs = dict()

        for worldname in self.render:
            world = dict()
            world.update(self.defaults)
            world.update(self.render[worldname])
        
            for key in world:
                if key not in settingsDefinition.render['values']:
                    logging.warning("%r is not a known setting", key)
                    continue
               
                definition = settingsDefinition.render['values'][key]
                try:
                    val = definition['validator'](world[key], world = self.world)
                    if definition.get('save_orig', False):
                        origs[key + "_orig"] = world[key]
                    world[key] = val
                except Exception as e:
                    logging.error("Error validating '%s' option in render definition for '%s':", key, worldname)
                    logging.error(e)
                    raise e
            world['name'] = worldname
            world.update(origs)
            self.render[worldname] = world

                

    def get_render_things(self):
        return self.render
