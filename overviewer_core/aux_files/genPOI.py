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
import re
import urllib2
import Queue
import multiprocessing

from multiprocessing import Process
from multiprocessing import Pool
from optparse import OptionParser

from overviewer_core import logger
from overviewer_core import nbt
from overviewer_core import configParser, world

UUID_LOOKUP_URL = 'https://sessionserver.mojang.com/session/minecraft/profile/'

def replaceBads(s):
    "Replaces bad characters with good characters!"
    bads = [" ", "(", ")"]
    x=s
    for bad in bads:
        x = x.replace(bad,"_")
    return x

# yes there's a double parenthesis here
# see below for when this is called, and why we do this
# a smarter way would be functools.partial, but that's broken on python 2.6
# when used with multiprocessing
def parseBucketChunks((bucket, rset)):
    pid = multiprocessing.current_process().pid
    pois = dict(TileEntities=[], Entities=[]);
  
    i = 0
    cnt = 0
    l = len(bucket)
    for b in bucket:
        try:
            data = rset.get_chunk(b[0],b[1])
            pois['TileEntities'] += data['TileEntities']
            pois['Entities']     += data['Entities']
        except nbt.CorruptChunkError:
            logging.warning("Ignoring POIs in corrupt chunk %d,%d", b[0], b[1])

        # Perhaps only on verbose ?
        i = i + 1
        if i == 250:
            i = 0
            cnt = 250 + cnt
            logging.info("Found %d entities and %d tile entities in thread %d so far at %d chunks", len(pois['Entities']), len(pois['TileEntities']), pid, cnt);

    return pois

def handleEntities(rset, outputdir, render, rname, config):

    # if we're already handled the POIs for this region regionset, do nothing
    if hasattr(rset, "_pois"):
        return

    logging.info("Looking for entities in %r", rset)

    filters = render['markers']
    rset._pois = dict(TileEntities=[], Entities=[])

    numbuckets = config['processes'];
    if numbuckets < 0:
        numbuckets = multiprocessing.cpu_count()

    if numbuckets == 1:
        for (x,z,mtime) in rset.iterate_chunks():
            try:
                data = rset.get_chunk(x,z) 
                rset._pois['TileEntities'] += data['TileEntities']
                rset._pois['Entities']     += data['Entities']
            except nbt.CorruptChunkError:
                logging.warning("Ignoring POIs in corrupt chunk %d,%d", x,z)
  
    else:
        buckets = [[] for i in range(numbuckets)];
  
        for (x,z,mtime) in rset.iterate_chunks():
            i = x / 32 + z / 32
            i = i % numbuckets 
            buckets[i].append([x,z])
  
        for b in buckets:
            logging.info("Buckets has %d entries", len(b));
  
        # Create a pool of processes and run all the functions
        pool = Pool(processes=numbuckets)
        results = pool.map(parseBucketChunks, ((buck, rset) for buck in buckets))
  
        logging.info("All the threads completed")
  
        # Fix up all the quests in the reset
        for data in results:
            rset._pois['TileEntities'] += data['TileEntities']
            rset._pois['Entities']     += data['Entities']

    logging.info("Done.")

class PlayerDict(dict):
    use_uuid = False
    _name = ''
    def __getitem__(self, item):
        if item == "EntityId":
            if not super(PlayerDict, self).has_key("EntityId"):
                if self.use_uuid:
                    super(PlayerDict, self).__setitem__("EntityId", self.get_name_from_uuid())
                else:
                    super(PlayerDict, self).__setitem__("EntityId", self._name)
        
        return super(PlayerDict, self).__getitem__(item)

    def get_name_from_uuid(self):
        try:
            profile = json.loads(urllib2.urlopen(UUID_LOOKUP_URL + self._name.replace('-','')).read())
            if 'name' in profile:
                return profile['name']
        except (ValueError, urllib2.URLError):
            logging.warning("Unable to get player name for UUID %s", self._name)

def handlePlayers(rset, render, worldpath):
    if not hasattr(rset, "_pois"):
        rset._pois = dict(TileEntities=[], Entities=[])

    # only handle this region set once
    if 'Players' in rset._pois:
        return

    if rset.get_type():
        dimension = int(re.match(r"^DIM(_MYST)?(-?\d+)$", rset.get_type()).group(2))
    else:
        dimension = 0

    playerdir = os.path.join(worldpath, "playerdata")
    useUUIDs = True
    if not os.path.isdir(playerdir):
        playerdir = os.path.join(worldpath, "players")
        useUUIDs = False

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
            data = PlayerDict(nbt.load(os.path.join(playerdir, playerfile))[1])
            data.use_uuid = useUUIDs
            if isSinglePlayer:
                data = data['Data']['Player']
        except IOError:
            logging.warning("Skipping bad player dat file %r", playerfile)
            continue
        playername = playerfile.split(".")[0]

        if isSinglePlayer:
            playername = 'Player'

        data._name = playername

        if data['Dimension'] == dimension:
            # Position at last logout
            data['id'] = "Player"
            data['x'] = int(data['Pos'][0])
            data['y'] = int(data['Pos'][1])
            data['z'] = int(data['Pos'][2])
            rset._pois['Players'].append(data)
        if "SpawnX" in data and dimension == 0:
            # Spawn position (bed or main spawn)
            spawn = PlayerDict()
            spawn._name = playername
            spawn["id"] = "PlayerSpawn"
            spawn["x"] = data['SpawnX']
            spawn["y"] = data['SpawnY']
            spawn["z"] = data['SpawnZ']
            rset._pois['Players'].append(spawn)

def handleManual(rset, manualpois):
    if not hasattr(rset, "_pois"):
        rset._pois = dict(TileEntities=[], Entities=[])
    
    rset._pois['Manual'] = []

    if manualpois:
        rset._pois['Manual'].extend(manualpois)

def main():

    if os.path.basename(sys.argv[0]) == """genPOI.py""":
        helptext = """genPOI.py
            %prog --config=<config file> [--quiet]"""
    else:
        helptext = """genPOI
            %prog --genpoi --config=<config file> [--quiet]"""

    logger.configure()

    parser = OptionParser(usage=helptext)
    parser.add_option("-c", "--config", dest="config", action="store", help="Specify the config file to use.")
    parser.add_option("--quiet", dest="quiet", action="count", help="Reduce logging output")
    parser.add_option("--skip-scan", dest="skipscan", action="store_true", help="Skip scanning for entities when using GenPOI")

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
        
        rset = w.get_regionset(render['dimension'][1])
        if rset == None: # indicates no such dimension was found:
            logging.error("Sorry, you requested dimension '%s' for the render '%s', but I couldn't find it", render['dimension'][0], rname)
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
        
        if not options.skipscan:
            handleEntities(rset, os.path.join(destdir, rname), render, rname, config)

        handlePlayers(rset, render, worldpath)
        handleManual(rset, render['manualpois'])

    logging.info("Done handling POIs")
    logging.info("Writing out javascript files")
    markerSetDict = dict()
    for (flter, rset) in markersets:
        # generate a unique name for this markerset.  it will not be user visible
        filter_name =     flter[0]
        filter_function = flter[1]

        name = replaceBads(filter_name) + hex(hash(filter_function))[-4:] + "_" + hex(hash(rset))[-4:]
        markerSetDict[name] = dict(created=False, raw=[], name=filter_name)
        for poi in rset._pois['Entities']:
            result = filter_function(poi)
            if result:
                if isinstance(result, basestring):
                    d = dict(x=poi['Pos'][0], y=poi['Pos'][1], z=poi['Pos'][2], text=result, hovertext=result)
                elif type(result) == tuple:
                    d = dict(x=poi['Pos'][0], y=poi['Pos'][1], z=poi['Pos'][2], text=result[1], hovertext=result[0])
                if "icon" in poi:
                    d.update({"icon": poi['icon']})
                if "createInfoWindow" in poi:
                    d.update({"createInfoWindow": poi['createInfoWindow']})
                markerSetDict[name]['raw'].append(d)
        for poi in rset._pois['TileEntities']:
            result = filter_function(poi)
            if result:
                if isinstance(result, basestring):
                    d = dict(x=poi['x'], y=poi['y'], z=poi['z'], text=result, hovertext=result)
                elif type(result) == tuple:
                    d = dict(x=poi['x'], y=poi['y'], z=poi['z'], text=result[1], hovertext=result[0])
                # Dict support to allow more flexible things in the future as well as polylines on the map.
                elif type(result) == dict:
                    d = dict(x=poi['x'], y=poi['y'], z=poi['z'], text=result['text'])
                    # Use custom hovertext if provided...
                    if 'hovertext' in result and isinstance(result['hovertext'], basestring):
                        d['hovertext'] = result['hovertext']
                    else: # ...otherwise default to display text.
                        d['hovertext'] = result['text']
                    if 'polyline' in result and type(result['polyline']) == tuple:  #if type(result.get('polyline', '')) == tuple:
                        d['polyline'] = []
                        for point in result['polyline']:
                            # This poor man's validation code almost definately needs improving.
                            if type(point) == dict:
                                d['polyline'].append(dict(x=point['x'],y=point['y'],z=point['z']))
                        if isinstance(result['color'], basestring):
                            d['strokeColor'] = result['color']
                if "icon" in poi:
                    d.update({"icon": poi['icon']})
                if "createInfoWindow" in poi:
                    d.update({"createInfoWindow": poi['createInfoWindow']})
                markerSetDict[name]['raw'].append(d)
        for poi in rset._pois['Players']:
            result = filter_function(poi)
            if result:
                if isinstance(result, basestring):
                    d = dict(x=poi['x'], y=poi['y'], z=poi['z'], text=result, hovertext=result)
                elif type(result) == tuple:
                    d = dict(x=poi['x'], y=poi['y'], z=poi['z'], text=result[1], hovertext=result[0])
                # Dict support to allow more flexible things in the future as well as polylines on the map.
                elif type(result) == dict:
                    d = dict(x=poi['x'], y=poi['y'], z=poi['z'], text=result['text'])
                    # Use custom hovertext if provided...
                    if 'hovertext' in result and isinstance(result['hovertext'], basestring):
                        d['hovertext'] = result['hovertext']
                    else: # ...otherwise default to display text.
                        d['hovertext'] = result['text']
                    if 'polyline' in result and type(result['polyline']) == tuple:  #if type(result.get('polyline', '')) == tuple:
                        d['polyline'] = []
                        for point in result['polyline']:
                            # This poor man's validation code almost definately needs improving.
                            if type(point) == dict:
                                d['polyline'].append(dict(x=point['x'],y=point['y'],z=point['z']))
                        if isinstance(result['color'], basestring):
                            d['strokeColor'] = result['color']
                if "icon" in poi:
                    d.update({"icon": poi['icon']})
                if "createInfoWindow" in poi:
                    d.update({"createInfoWindow": poi['createInfoWindow']})
                markerSetDict[name]['raw'].append(d)
        for poi in rset._pois['Manual']:
            result = filter_function(poi)
            if result:
                if isinstance(result, basestring):
                    d = dict(x=poi['x'], y=poi['y'], z=poi['z'], text=result, hovertext=result)
                elif type(result) == tuple:
                    d = dict(x=poi['x'], y=poi['y'], z=poi['z'], text=result[1], hovertext=result[0])
                # Dict support to allow more flexible things in the future as well as polylines on the map.
                elif type(result) == dict:
                    d = dict(x=poi['x'], y=poi['y'], z=poi['z'], text=result['text'])
                    # Use custom hovertext if provided...
                    if 'hovertext' in result and isinstance(result['hovertext'], basestring):
                        d['hovertext'] = result['hovertext']
                    else: # ...otherwise default to display text.
                        d['hovertext'] = result['text']
                    if 'polyline' in result and type(result['polyline']) == tuple:  #if type(result.get('polyline', '')) == tuple:
                        d['polyline'] = []
                        for point in result['polyline']:
                            # This poor man's validation code almost definately needs improving.
                            if type(point) == dict:
                                d['polyline'].append(dict(x=point['x'],y=point['y'],z=point['z']))
                        if isinstance(result['color'], basestring):
                            d['strokeColor'] = result['color']
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
