#!/usr/bin/python
"""Dump the contents of overviewer.dat"""

import cPickle
import os
import sys
import pprint

if len(sys.argv)<1:
    print "Usage: %s <path to overviewer.dat>" % sys.argv[0]
    sys.exit(-1)

pickleFile = sys.argv[1]

with open(pickleFile,"rb") as p:
    persistentData = cPickle.load(p)
    out = pprint.PrettyPrinter()
    out.pprint(persistentData)
