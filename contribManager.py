#!/usr/bin/env python

# The contrib manager is used to help control the contribs script 
# that are shipped with overviewer in Windows packages

import sys
import os.path

scripts=dict( # keys are names, values are scripts
        benchmark="benchmark.py",
        findSigns="findSigns.py",
        validate="validateRegionFile.py"
        )


# figure out what script to execute
argv=os.path.basename(sys.argv[0])

if argv[-4:] == ".exe":
    argv=argv[0:-4]
if argv[-3:] == ".py":
    argv=argv[0:-3]

print "argv is ", argv

if argv in scripts.keys():
    script = scripts[argv]
else:
    if sys.argv[1] in scripts.keys():
        script = scripts[sys.argv[1]]
    else:
        print "what do you want to run?"
        sys.exit(1)


print "running", script

execfile(os.path.join("contrib", script))
