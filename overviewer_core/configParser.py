from optparse import OptionParser
import sys
import os.path
import logging

class OptionsResults(object):
    def get(self, *args):
        return self.__dict__.get(*args)

class ConfigOptionParser(object):
    def __init__(self, **kwargs):
        self.cmdParser = OptionParser(usage=kwargs.get("usage",""))
        self.configFile = kwargs.get("config","settings.py")
        self.configVars = []

        # these are arguments not understood by OptionParser, so they must be removed
        # in add_option before being passed to the OptionParser

        # note that default is a valid OptionParser argument, but we remove it
        # because we want to do our default value handling

        self.customArgs = ["required", "commandLineOnly", "default", "listify", "listdelim", "choices"]

        self.requiredArgs = []
        
        # add the *very* special config-file path option
        self.add_option("--settings", dest="config_file", help="Specifies a settings file to load, by name. This file's format is discussed in the README.", metavar="PATH", type="string", commandLineOnly=True)

    def display_config(self):
        logging.info("Using the following settings:")
        for x in self.configVars:
            n = x['dest']
            print  "%s: %r" % (n, self.configResults.__dict__[n])


    def add_option(self, *args, **kwargs):

        if kwargs.get("configFileOnly", False) and kwargs.get("commandLineOnly", False):
            raise Exception(args, "configFileOnly and commandLineOnly are mututally exclusive")

        self.configVars.append(kwargs.copy())

        if not kwargs.get("configFileOnly", False):
            for arg in self.customArgs:
                if arg in kwargs.keys(): del kwargs[arg]

            if kwargs.get("type", None):
                kwargs['type'] = 'string' # we'll do our own converting later
            self.cmdParser.add_option(*args, **kwargs)

    def print_help(self):
        self.cmdParser.print_help()

    def parse_args(self):

        # first, load the results from the command line:
        options, args = self.cmdParser.parse_args()

        # second, use these values to seed the locals dict
        l = dict()
        g = dict()
        for a in self.configVars:
            n = a['dest']
            if a.get('configFileOnly', False): continue
            if a.get('commandLineOnly', False): continue
            v = getattr(options, n)
            if v != None:
                #print "seeding %s with %s" % (n, v)
                l[n] = v
            else:
                # if this has a default, use that to seed the globals dict
                if a.get("default", None): g[n] = a['default']
        g['args'] = args
        
        try:
            if options.config_file:
                self.configFile = options.config_file
            elif os.path.exists(self.configFile):
                # warn about automatic loading
                logging.warning("Automatic settings.py loading is DEPRECATED, and may be removed in the future. Please use --settings instead.")
            
            if os.path.exists(self.configFile):
                execfile(self.configFile, g, l)
            elif options.config_file:
                # file does not exist, but *was* specified on the command line
                logging.error("Could not open %s." % self.configFile)
                sys.exit(1)
        except NameError, ex:
            import traceback
            traceback.print_exc()
            logging.error("Error parsing %s.  Please check the trackback above" % self.configFile)
            sys.exit(1)
        except SyntaxError, ex:
            import traceback
            traceback.print_exc()
            tb = sys.exc_info()[2]
            #print tb.tb_frame.f_code.co_filename
            logging.error("Error parsing %s.  Please check the trackback above" % self.configFile)
            sys.exit(1)

        #print l.keys()

        configResults = OptionsResults()
        # third, load the results from the config file:
        for a in self.configVars:
            n = a['dest']
            if a.get('commandLineOnly', False):
                if n in l.keys():
                    logging.error("Error: %s can only be specified on the command line.  It is not valid in the config file" % n)
                    sys.exit(1)

            configResults.__dict__[n] = l.get(n)


        
        # third, merge options into configReslts (with options overwriting anything in configResults)
        for a in self.configVars:
            n = a['dest']
            if a.get('configFileOnly', False): continue
            if getattr(options, n) != None:
                configResults.__dict__[n] = getattr(options, n)

        # forth, set defaults for any empty values
        for a in self.configVars:
            n = a['dest']
            if (n not in configResults.__dict__.keys() or configResults.__dict__[n] == None) and 'default' in a.keys():
                configResults.__dict__[n] = a['default']

        # fifth, check required args:
        for a in self.configVars:
            n = a['dest']
            if configResults.__dict__[n] == None and a.get('required',False):
                logging.error("%s is required" % n)
                sys.exit(1)

        # sixth, check types
        for a in self.configVars:
            n = a['dest']
            if 'listify' in a.keys():
                # this thing may be a list!
                if configResults.__dict__[n] != None and type(configResults.__dict__[n]) == str:
                    configResults.__dict__[n] = configResults.__dict__[n].split(a.get("listdelim",","))
                elif type(configResults.__dict__[n]) != list:
                    configResults.__dict__[n] = [configResults.__dict__[n]]
            if 'type' in a.keys() and configResults.__dict__[n] != None:
                try:
                    configResults.__dict__[n] = self.checkType(configResults.__dict__[n], a)
                except ValueError, ex:
                    logging.error("There was a problem converting the value '%s' to type %s for config parameter '%s'" % (configResults.__dict__[n], a['type'], n))
                    import traceback
                    #traceback.print_exc()
                    sys.exit(1)



        self.configResults = configResults

        return configResults, args

    def checkType(self, value, a):

        if type(value) == list:
            return map(lambda x: self.checkType(x, a), value)

        # switch on type.  there are only 7 types that can be used with optparse
        if a['type'] == "int":
            return int(value)
        elif a['type'] == "string":
            return str(value)
        elif a['type'] == "long":
            return long(value)
        elif a['type'] == "choice":
            if value not in a['choices']:
                logging.error("The value '%s' is not valid for config parameter '%s'" % (value, a['dest']))
                sys.exit(1)
            return value
        elif a['type'] == "float":
            return long(value)
        elif a['type'] == "complex":
            return complex(value)
        elif a['type'] == "function":
            if not callable(value):
                raise ValueError("Not callable")
        else:
            logging.error("Unknown type!")
            sys.exit(1)
