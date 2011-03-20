from optparse import OptionParser
import sys
import os.path

class OptionsResults(object):
    pass

class ConfigOptionParser(object):
    def __init__(self, **kwargs):
        self.cmdParser = OptionParser(usage=kwargs.get("usage",""))
        self.configFile = kwargs.get("config","settings.py")
        self.configVars = []

        # these are arguments not understood by OptionParser, so they must be removed
        # in add_option before being passed to the OptionParser
        # note that default is a valid OptionParser argument, but we remove it
        # because we want to do our default value handling
        self.customArgs = ["required", "commandLineOnly", "default"]

        self.requiredArgs = []

    def display_config(self):
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
            if os.path.exists(self.configFile):
                execfile(self.configFile, g, l)
        except NameError, ex:
            import traceback
            traceback.print_exc()
            print "\nError parsing %s.  Please check the trackback above" % self.configFile
            sys.exit(1)
        except SyntaxError, ex:
            import traceback
            traceback.print_exc()
            tb = sys.exc_info()[2]
            #print tb.tb_frame.f_code.co_filename
            print "\nError parsing %s.  Please check the trackback above" % self.configFile
            sys.exit(1)

        #print l.keys()

        configResults = OptionsResults()
        # third, load the results from the config file:
        for a in self.configVars:
            n = a['dest']
            if a.get('commandLineOnly', False):
                if n in l.keys():
                    print "Error: %s can only be specified on the command line.  It is not valid in the config file" % n
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
                raise Exception("%s is required" % n)

        # sixth, check types
        for a in self.configVars:
            n = a['dest']
            if 'type' in a.keys() and configResults.__dict__[n] != None:
                try:
                    # switch on type.  there are only 6 types that can be used with optparse
                    if a['type'] == "int":
                        configResults.__dict__[n] = int(configResults.__dict__[n])
                    elif a['type'] == "string":
                        configResults.__dict__[n] = str(configResults.__dict__[n])
                    elif a['type'] == "long":
                        configResults.__dict__[n] = long(configResults.__dict__[n])
                    elif a['type'] == "choice":
                        if configResults.__dict__[n] not in a['choices']:
                            print "The value '%s' is not valid for config parameter '%s'" % (configResults.__dict__[n], n)
                            sys.exit(1)
                    elif a['type'] == "float":
                        configResults.__dict__[n] = long(configResults.__dict__[n])
                    elif a['type'] == "complex":
                        configResults.__dict__[n] = complex(configResults.__dict__[n])
                    else:
                        print "Unknown type!"
                        sys.exit(1)
                except ValueError, ex:
                    print "There was a problem converting the value '%s' to type %s for config parameter '%s'" % (configResults.__dict__[n], a['type'], n)
                    import traceback
                    #traceback.print_exc()
                    sys.exit(1)



        self.configResults = configResults

        return configResults, args

