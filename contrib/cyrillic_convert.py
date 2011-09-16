#!/usr/bin/python

"""Convert gibberish back into Cyrillic"""

import fileinput
import os
import sys

usage = """
If you have signs that should be Cyrillic, but are instead gibberish,
this script will convert it back to proper Cyrillic.

usage: python %(script)s <markers.js>
ex. python %(script)s C:\\Inetpub\\www\\map\\markers.js
 or %(script)s /srv/http/map/markers.js
""" % {'script': os.path.basename(sys.argv[0])}

if len(sys.argv) < 2:
    sys.exit(usage)

gibberish_to_cyrillic = {
    r"\u00c0": r"\u0410",
    r"\u00c1": r"\u0411",
    r"\u00c2": r"\u0412",
    r"\u00c3": r"\u0413",
    r"\u00c4": r"\u0414",
    r"\u00c5": r"\u0415",
    r"\u00c6": r"\u0416",
    r"\u00c7": r"\u0417",
    r"\u00c8": r"\u0418",
    r"\u00c9": r"\u0419",
    r"\u00ca": r"\u041a",
    r"\u00cb": r"\u041b",
    r"\u00cc": r"\u041c",
    r"\u00cd": r"\u041d",
    r"\u00ce": r"\u041e",
    r"\u00cf": r"\u041f",
    r"\u00d0": r"\u0420",
    r"\u00d1": r"\u0421",
    r"\u00d2": r"\u0422",
    r"\u00d3": r"\u0423",
    r"\u00d4": r"\u0424",
    r"\u00d5": r"\u0425",
    r"\u00d6": r"\u0426",
    r"\u00d7": r"\u0427",
    r"\u00d8": r"\u0428",
    r"\u00d9": r"\u0429",
    r"\u00da": r"\u042a",
    r"\u00db": r"\u042b",
    r"\u00dc": r"\u042c",
    r"\u00dd": r"\u042d",
    r"\u00de": r"\u042e",
    r"\u00df": r"\u042f",
    r"\u00e0": r"\u0430",
    r"\u00e1": r"\u0431",
    r"\u00e2": r"\u0432",
    r"\u00e3": r"\u0433",
    r"\u00e4": r"\u0434",
    r"\u00e5": r"\u0435",
    r"\u00e6": r"\u0436",
    r"\u00e7": r"\u0437",
    r"\u00e8": r"\u0438",
    r"\u00e9": r"\u0439",
    r"\u00ea": r"\u043a",
    r"\u00eb": r"\u043b",
    r"\u00ec": r"\u043c",
    r"\u00ed": r"\u043d",
    r"\u00ee": r"\u043e",
    r"\u00ef": r"\u043f",
    r"\u00f0": r"\u0440",
    r"\u00f1": r"\u0441",
    r"\u00f2": r"\u0442",
    r"\u00f3": r"\u0443",
    r"\u00f4": r"\u0444",
    r"\u00f5": r"\u0445",
    r"\u00f6": r"\u0446",
    r"\u00f7": r"\u0447",
    r"\u00f8": r"\u0448",
    r"\u00f9": r"\u0449",
    r"\u00fa": r"\u044a",
    r"\u00fb": r"\u044b",
    r"\u00fc": r"\u044c",
    r"\u00fd": r"\u044d",
    r"\u00fe": r"\u044e",
    r"\u00ff": r"\u044f"
}

for line in fileinput.FileInput(inplace=1):
    for i, j in gibberish_to_cyrillic.iteritems():
        line = line.replace(i, j)
    sys.stdout.write(line)

