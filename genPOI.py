#!/usr/bin/python2

'''
genPOI.py

Scans regionsets for TileEntities and Entities, filters them, and writes out
POI/marker info.

A markerSet is list of POIs to display on a tileset.  It has a display name,
and a group name.

markersDB.js holds a list of POIs in each group
markers.js holds a list of which markerSets are attached to each tileSet


'''
import os
import logging
import json
from optparse import OptionParser

from overviewer_core import logger
from overviewer_core import nbt
from overviewer_core import configParser, world

helptext = """
%prog --config=<config file>"""

logger.configure()

def handleSigns(rset, outputdir, render, rname):
    	
    if hasattr(rset, "_pois"):
        return

    logging.info("Looking for entities in %r", rset)

    filters = render['markers']
    rset._pois = dict(TileEntities=[], Entities=[])

    for (x,z,mtime) in rset.iterate_chunks():
        data = rset.get_chunk(x,z)
        rset._pois['TileEntities'] += data['TileEntities']
        rset._pois['Entities']     += data['Entities']


def main():
    parser = OptionParser(usage=helptext)
    parser.add_option("--config", dest="config", action="store", help="Specify the config file to use.")

    options, args = parser.parse_args()
    if not options.config:
        parser.print_help()
        return

    # Parse the config file
    mw_parser = configParser.MultiWorldParser()
    mw_parser.parse(options.config)
    try:
        config = mw_parser.get_validated_config()
    except Exception:
        logging.exception("An error was encountered with your configuration. See the info below.")
        return 1

    destdir = config['outputdir']
    # saves us from creating the same World object over and over again
    worldcache = {}

    markersets = set()
    markers = dict()

    for rname, render in config['renders'].iteritems():
        try:
            worldpath = config['worlds'][render['world']]
        except KeyError:
            logging.error("Render %s's world is '%s', but I could not find a corresponding entry in the worlds dictionary.",
                    rname, render['world'])
            return 1
        render['worldname_orig'] = render['world']
        render['world'] = worldpath
        
        # find or create the world object
        if (render['world'] not in worldcache):
            w = world.World(render['world'])
            worldcache[render['world']] = w
        else:
            w = worldcache[render['world']]
        
        rset = w.get_regionset(render['dimension'])
        if rset == None: # indicates no such dimension was found:
            logging.error("Sorry, you requested dimension '%s' for %s, but I couldn't find it", render['dimension'], render_name)
            return 1
      
        for f in render['markers']:
            markersets.add((f, rset))
            name = f.__name__ + hex(hash(f))[-4:] + "_" + hex(hash(rset))[-4:]
            try:
                l = markers[rname]
                l.append(dict(groupName=name, displayName = f.__doc__))
            except KeyError:
                markers[rname] = [dict(groupName=name, displayName=f.__doc__),]

        handleSigns(rset, os.path.join(destdir, rname), render, rname)

    logging.info("Done scanning regions")
    logging.info("Writing out javascript files")
    markerSetDict = dict()
    for (flter, rset) in markersets:
        # generate a unique name for this markerset.  it will not be user visible
        name = flter.__name__ + hex(hash(flter))[-4:] + "_" + hex(hash(rset))[-4:]
        markerSetDict[name] = dict(created=False, raw=[])
        for poi in rset._pois['TileEntities']:
            if flter(poi):
                markerSetDict[name]['raw'].append(poi)
    #print markerSetDict

    with open(os.path.join(destdir, "markersDB.js"), "w") as output:
        output.write("var markersDB=")
        json.dump(markerSetDict, output, indent=2)
        output.write(";\n");
    with open(os.path.join(destdir, "markers.js"), "w") as output:
        output.write("var markers=")
        json.dump(markers, output, indent=2)
        output.write(";\n");
    with open(os.path.join(destdir, "baseMarkers.js"), "w") as output:
        output.write("overviewer.util.injectMarkerScript('markersDB.js');\n")
        output.write("overviewer.util.injectMarkerScript('markers.js');\n")
        output.write("overviewer.collections.haveSigns=true;\n")
    logging.info("Done")

if __name__ == "__main__":
    main()
