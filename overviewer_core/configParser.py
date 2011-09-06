import optparse
import sys
import os.path
import logging

class OptionsResults(object):
    def get(self, *args):
        return self.__dict__.get(*args)

class ConfigOptionParser(object):
    def __init__(self, **kwargs):
        self.cmdParser = optparse.OptionParser(usage=kwargs.get("usage",""))
        self.configFile = kwargs.get("config","settings.py")
        self.listifyDelimiters = kwargs.get("listdelim", ",:/")
        self.configVars = []
        self.advancedHelp = []
        # these are arguments not understood by OptionParser, so they must be removed
        # in add_option before being passed to the OptionParser

        # note that default is a valid OptionParser argument, but we remove it
        # because we want to do our default value handling

        self.customArgs = ["required", "commandLineOnly", "default", "listify", "listdelim", "choices", "helptext", "advanced"]

        self.requiredArgs = []
        
        # add the *very* special advanced help and config-file path options
        self.add_option("--advanced-help", dest="advanced_help", action="store_true", helptext="Display help - including advanced options", commandLineOnly=True)
        self.add_option("--settings", dest="config_file", helptext="Specifies a settings file to load, by name. This file's format is discussed in the README.", metavar="PATH", type="string", commandLineOnly=True)

    def display_config(self):
        logging.info("Using the following settings:")
        for x in self.configVars:
            n = x['dest']
            print  "%s: %r" % (n, self.configResults.__dict__[n])


    def add_option(self, *args, **kwargs):

        self.configVars.append(kwargs.copy())

        if kwargs.get("advanced"):
            kwargs['help'] = optparse.SUPPRESS_HELP
            self.advancedHelp.append((args, kwargs.copy()))
        else:
            kwargs["help"]=kwargs["helptext"]

        for arg in self.customArgs:
            if arg in kwargs.keys(): del kwargs[arg]
        if kwargs.get("type", None):
            kwargs['type'] = 'string' # we'll do our own converting later
        self.cmdParser.add_option(*args, **kwargs)


    def print_help(self):
        self.cmdParser.print_help()

    def advanced_help(self):
        self.cmdParser.set_conflict_handler('resolve') # Allows us to overwrite the previous definitions
        for opt in self.advancedHelp:
            opt[1]['help']="[!]" + opt[1]['helptext']
            for arg in self.customArgs:
                if arg in opt[1].keys():
                    del opt[1][arg]
            if opt[1].get("type", None):
                opt[1]['type'] = 'string' # we'll do our own converting later
            self.cmdParser.add_option(*opt[0], **opt[1])
            self.cmdParser.epilog = "Advanced options indicated by [!]. These options should not normally be required, and may have caveats regarding their use. See README file for more details"
        self.print_help()


    def parse_args(self):

        # first, load the results from the command line:
        options, args = self.cmdParser.parse_args()

        # second, use these values to seed the locals dict
        l = dict()
        g = dict()
        for a in self.configVars:
            n = a['dest']
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
                    delimiters = a.get("listdelim", self.listifyDelimiters)
                    # replace the rest of the delimiters with the first
                    for delim in delimiters[1:]:
                        configResults.__dict__[n] = configResults.__dict__[n].replace(delim, delimiters[0])
                    # split at each occurance of the first delimiter
                    configResults.__dict__[n] = configResults.__dict__[n].split(delimiters[0])
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
            if ('choices' in a) and (value not in a['choices']):
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
