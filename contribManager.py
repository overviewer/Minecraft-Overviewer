#!/usr/bin/env python

# The contrib manager is used to help control the contribs script 
# that are shipped with overviewer in Windows packages

import sys
import os.path
import ast

# incantation to be able to import overviewer_core
if not hasattr(sys, "frozen"):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0], '.')))

from overviewer_core import nbt

scripts=dict( # keys are names, values are scripts
        clearOldCache   = "clearOldCache.py",
        convertCyrillic = "cyrillic_convert.py",
        findSigns       = "findSigns.py",
        findAnimals     = "findAnimals.py",
        playerInspect   = "playerInspect.py",
        rerenderBlocks  = "rerenderBlocks.py",
        testRender      = "testRender.py",
        validate        = "validateRegionFile.py",
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


usage="""Usage:
%s --list-contribs | <script name> <arguments>

Executes a contrib script.  

Options:
  --list-contribs           Lists the supported contrib scripts

""" % os.path.basename(sys.argv[0])

if argv in scripts.keys():
    script = scripts[argv]
    sys.argv[0] = script
else:
    if "--list-contribs" in sys.argv:
        for contrib in scripts.keys():
            # use an AST to extract the docstring for this module
            script = scripts[contrib]
            with open(os.path.join("contrib",script)) as f:
                d = f.read()
            node=ast.parse(d, script);
            docstring = ast.get_docstring(node)
            if docstring:
                docstring = docstring.strip().splitlines()[0]
            else:
                docstring="(no description found.  add one by adding a docstring to %s)" % script
            print "%s : %s" % (contrib, docstring)
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] in scripts.keys():
        script = scripts[sys.argv[1]]
        sys.argv = [script] + sys.argv[2:]
    else:
        print usage
        sys.exit(1)


torun = os.path.join("contrib", script)

if not os.path.exists(torun):
    print "Script '%s' is missing!" % script
    sys.exit(1)

execfile(torun)

