#
# Code to check to make sure c_overviewer is built and working
#

import os.path
import os
import platform
import traceback
import sys

import util

def check_c_overviewer():
    """Check to make sure c_overviewer works and is up-to-date. Prints
    out a helpful error and returns 1 if something's wrong, returns 0
    otherwise.
    """
    root_dir = util.get_program_path()
    # make sure the c_overviewer extension is available
    try:
        import c_overviewer
    except ImportError:
        if os.environ.get("OVERVIEWER_DEBUG_IMPORT") == "1":
            traceback.print_exc()
        ## if this is a frozen windows package, the following error messages about
        ## building the c_overviewer extension are not appropriate
        if hasattr(sys, "frozen") and platform.system() == 'Windows':
            print "Something has gone wrong importing the c_overviewer extension.  Please"
            print "make sure the 2008 and 2010 redistributable packages from Microsoft"
            print "are installed."
            return 1

        ## try to find the build extension
        ext = os.path.join(root_dir, "overviewer_core", "c_overviewer.%s" % ("pyd" if platform.system() == "Windows" else "so"))
        if os.path.exists(ext):
            traceback.print_exc()
            print ""
            print "Something has gone wrong importing the c_overviewer extension.  Please"
            print "make sure it is up-to-date (clean and rebuild)"
            return 1

        print "You need to compile the c_overviewer module to run Minecraft Overviewer."
        print "Run `python setup.py build`, or see the README for details."
        return 1

    #
    # make sure it's up-to-date
    #

    if hasattr(sys, "frozen"):
        pass # we don't bother with a compat test since it should always be in sync
    elif "extension_version" in dir(c_overviewer):
        # check to make sure the binary matches the headers
        if os.path.exists(os.path.join(root_dir, "overviewer_core", "src", "overviewer.h")):
            with open(os.path.join(root_dir, "overviewer_core", "src", "overviewer.h")) as f:
                lines = f.readlines()
                lines = filter(lambda x: x.startswith("#define OVERVIEWER_EXTENSION_VERSION"), lines)
                if lines:
                    l = lines[0]
                    if int(l.split()[2].strip()) != c_overviewer.extension_version():
                        print "Please rebuild your c_overviewer module.  It is out of date!"
                        return 1
    else:
        print "Please rebuild your c_overviewer module.  It is out of date!"
        return 1
    
    # all good!
    return 0

# only check the module if we're not setup.py
if not sys.argv[0].endswith("setup.py"):
    ret = check_c_overviewer()
    if ret > 0:
        util.nice_exit(ret)
