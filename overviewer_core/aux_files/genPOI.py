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
import sys
from optparse import OptionParser

from overviewer_core import logger
from overviewer_core import nbt
from overviewer_core import configParser, world

def replaceBads(s):
    "Replaces bad characters with good characters!"
    bads = [" ", "(", ")"]
    x=s
    for bad in bads:
        x = x.replace(bad,"_")
    return x

def handleSigns(rset, outputdir, render, rname):

    # if we're already handled the POIs for this region regionset, do nothing
    if hasattr(rset, "_pois"):
        return

    logging.info("Looking for entities in %r", rset)

    filters = render['markers']
    rset._pois = dict(TileEntities=[], Entities=[])

    for (x,z,mtime) in rset.iterate_chunks():
        data = rset.get_chunk(x,z)
        rset._pois['TileEntities'] += data['TileEntities']
        rset._pois['Entities']     += data['Entities']

    logging.info("Done.")

def handlePlayers(rset, render, worldpath):
    if not hasattr(rset, "_pois"):
        rset._pois = dict(TileEntities=[], Entities=[])

    # only handle this region set once
    if 'Players' in rset._pois:
        return
    dimension = {'overworld': 0,
                 'nether': -1,
                 'end': 1,
                 'default': 0}[render['dimension']]
    playerdir = os.path.join(worldpath, "players")
    if os.path.isdir(playerdir):
        playerfiles = os.listdir(playerdir)
        playerfiles = [x for x in playerfiles if x.endswith(".dat")]
        isSinglePlayer = False

    else:
        playerfiles = [os.path.join(worldpath, "level.dat")]
        isSinglePlayer = True

    rset._pois['Players'] = []
    for playerfile in playerfiles:
        try:
            data = nbt.load(os.path.join(playerdir, playerfile))[1]
            if isSinglePlayer:
                data = data['Data']['Player']
        except IOError:
            logging.warning("Skipping bad player dat file %r", playerfile)
            continue
        playername = playerfile.split(".")[0]
        if isSinglePlayer:
            playername = 'Player'
        if data['Dimension'] == dimension:
            # Position at last logout
            data['id'] = "Player"
            data['EntityId'] = playername
            data['x'] = int(data['Pos'][0])
            data['y'] = int(data['Pos'][1])
            data['z'] = int(data['Pos'][2])
            rset._pois['Players'].append(data)
        if "SpawnX" in data and dimension == 0:
            # Spawn position (bed or main spawn)
            spawn = {"id": "PlayerSpawn",
                     "EntityId": playername,
                     "x": data['SpawnX'],
                     "y": data['SpawnY'],
                     "z": data['SpawnZ']}
            rset._pois['Players'].append(spawn)

def main():

    if os.path.basename(sys.argv[0]) == """genPOI.py""":
        helptext = """genPOI.py
            %prog --config=<config file> [--quiet]"""
    else:
        helptext = """genPOI
            %prog --genpoi --config=<config file> [--quiet]"""

    logger.configure()

    parser = OptionParser(usage=helptext)
    parser.add_option("--config", dest="config", action="store", help="Specify the config file to use.")
    parser.add_option("--quiet", dest="quiet", action="count", help="Reduce logging output")

    options, args = parser.parse_args()
    if not options.config:
        parser.print_help()
        return

    if options.quiet > 0:
        logger.configure(logging.WARN, False)

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
            markersets.add(((f['name'], f['filterFunction']), rset))
            name = replaceBads(f['name']) + hex(hash(f['filterFunction']))[-4:] + "_" + hex(hash(rset))[-4:]
            to_append = dict(groupName=name, 
                    displayName = f['name'], 
                    icon=f.get('icon', 'signpost_icon.png'), 
                    createInfoWindow=f.get('createInfoWindow',True),
                    checked = f.get('checked', False))
            try:
                l = markers[rname]
                l.append(to_append)
            except KeyError:
                markers[rname] = [to_append]

        handleSigns(rset, os.path.join(destdir, rname), render, rname)
        handlePlayers(rset, render, worldpath)

    logging.info("Done scanning regions")
    logging.info("Writing out javascript files")
    markerSetDict = dict()
    for (flter, rset) in markersets:
        # generate a unique name for this markerset.  it will not be user visible
        filter_name =     flter[0]
        filter_function = flter[1]

        name = replaceBads(filter_name) + hex(hash(filter_function))[-4:] + "_" + hex(hash(rset))[-4:]
        markerSetDict[name] = dict(created=False, raw=[], name=filter_name)
        for poi in rset._pois['TileEntities']:
            result = filter_function(poi)
            if result:
                d = dict(x=poi['x'], y=poi['y'], z=poi['z'], text=result)
                if "icon" in poi:
                    d.update({"icon": poi['icon']})
                if "createInfoWindow" in poi:
                    d.update({"createInfoWindow": poi['createInfoWindow']})
                markerSetDict[name]['raw'].append(d)
        for poi in rset._pois['Players']:
            result = filter_function(poi)
            if result:
                d = dict(x=poi['x'], y=poi['y'], z=poi['z'], text=result)
                if "icon" in poi:
                    d.update({"icon": poi['icon']})
                if "createInfoWindow" in poi:
                    d.update({"createInfoWindow": poi['createInfoWindow']})
                markerSetDict[name]['raw'].append(d)
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
