#!/usr/bin/env python

# The contrib manager is used to help control the contribs script 
# that are shipped with overviewer in Windows packages

import sys
import os.path
sys.path.append("overviewer_core")
import nbt

scripts=dict( # keys are names, values are scripts
        benchmark="benchmark.py",
        findSigns="findSigns.py",
        validate="validateRegionFile.py"
        )

# you can symlink or hardlink contribManager.py to another name to have it
# automatically find the right script to run.  For example:
# > ln -s contribManager.py validate.exe
# > chmod +x validate.exe
# > ./validate.exe -h


# figure out what script to execute
argv=os.path.basename(sys.argv[0])

if argv[-4:] == ".exe":
    argv=argv[0:-4]
if argv[-3:] == ".py":
    argv=argv[0:-3]

if argv in scripts.keys():
    script = scripts[argv]
    sys.argv[0] = script
else:
    if "--list-contribs" in sys.argv:
        print scripts.keys()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] in scripts.keys():
        script = scripts[sys.argv[1]]
        sys.argv = [script] + sys.argv[2:]
    else:
        print "what do you want to run?"
        sys.exit(1)


torun = os.path.join("contrib", script)

if not os.path.exists(torun):
    print "Script '%s' is missing!" % script
    sys.exit(1)

execfile(torun)

