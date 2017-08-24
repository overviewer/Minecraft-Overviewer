#!/usr/bin/env python2

'''
genPOI.py

Scans regionsets for TileEntities and Entities, filters them, and writes out
POI/marker info.

A markerSet is list of POIs to display on a tileset.  It has a display name,
and a group name.

markersDB.js holds a list of POIs in each group
markers.js holds a list of which markerSets are attached to each tileSet


'''
import gzip
import itertools
import json
import logging
import multiprocessing
import os
import re
import sys
import time
import urllib2
import datetime

from collections import defaultdict
from contextlib import closing
from multiprocessing import Pool
from optparse import OptionParser

from overviewer_core import logger
from overviewer_core import nbt
from overviewer_core import configParser, world
from overviewer_core.files import FileReplacer, get_fs_caps

UUID_LOOKUP_URL = 'https://sessionserver.mojang.com/session/minecraft/profile/'

def replaceBads(s):
    "Replaces bad characters with good characters!"
    bads = [" ", "(", ")"]
    x=s
    for bad in bads:
        x = x.replace(bad,"_")
    return x

# If you want to keep your stomach contents do not, under any circumstance,
# read the body of the following function. You have been warned.
# All of this could be replaced by a simple json.loads if Mojang had
# introduced a TAG_JSON, but they didn't.
#
# So here are a few curiosities how 1.7 signs get seen in 1.8 in Minecraft:
# - null        ->
# - "null"      -> null
# - ["Hello"]   -> Hello
# - [Hello]     -> Hello
# - [1,2,3]     -> 123
# Mojang just broke signs for everyone who ever used [, { and ". GG.
def jsonText(s):
    if s is None or s == "null":
        return ""
    if (s.startswith('"') and s.endswith('"')) or \
        (s.startswith('{') and s.endswith('}')):
        try:
            js = json.loads(s)
        except ValueError:
            return s

        def parseLevel(foo):
            bar = ""
            if isinstance(foo, list):
                for extra in foo:
                    bar += parseLevel(extra)
            elif isinstance(foo, dict):
                if "text" in foo:
                    bar += foo["text"]
                if "extra" in foo:
                    bar += parseLevel(foo["extra"])
            elif isinstance(foo, basestring):
                bar = foo
            return bar

        return parseLevel(js)

    else:
        return s

# Since functions are not pickleable, we send their names instead.
# Here, set up worker processes to have a name -> function map
bucketChunkFuncs = {}
def initBucketChunks(config_path):
    global bucketChunkFuncs
    
    mw_parser = configParser.MultiWorldParser()
    mw_parser.parse(config_path)
    # ought not to fail since we already did it once
    config = mw_parser.get_validated_config()
    for name, render in config['renders'].iteritems():
        for f in render['markers']:
            ff = f['filterFunction']
            bucketChunkFuncs[ff.__name__] = ff

# yes there's a double parenthesis here
# see below for when this is called, and why we do this
# a smarter way would be functools.partial, but that's broken on python 2.6
# when used with multiprocessing
def parseBucketChunks((bucket, rset, filters)):
    global bucketChunkFuncs
    pid = multiprocessing.current_process().pid
    markers = defaultdict(list)

    i = 0
    cnt = 0
    for b in bucket:
        try:
            data = rset.get_chunk(b[0],b[1])
            for poi in itertools.chain(data['TileEntities'], data['Entities']):
                if poi['id'] == 'Sign' or poi['id'] == 'minecraft:sign':
                    poi = signWrangler(poi)
                for name, filter_function in filters:
                    ff = bucketChunkFuncs[filter_function]
                    result = ff(poi)
                    if result:
                        d = create_marker_from_filter_result(poi, result)
                        markers[name].append(d)
        except nbt.CorruptChunkError:
            logging.warning("Ignoring POIs in corrupt chunk %d,%d", b[0], b[1])

        # Perhaps only on verbose ?
        i = i + 1
        if i == 250:
            i = 0
            cnt = 250 + cnt
            logging.info("Found %d markers in thread %d so far at %d chunks", sum(len(v) for v in markers.itervalues()), pid, cnt);

    return markers

def signWrangler(poi):
    """
    Just does the JSON things for signs
    """
    for field in ["Text1", "Text2", "Text3", "Text4"]:
        poi[field] = jsonText(poi[field])
    return poi


def handleEntities(rset, config, config_path, filters, markers):
    """
    Add markers for Entities or TileEntities.

    For this every chunk of the regionset is parsed and filtered using multiple
    processes, if so configured.
    This function will not return anything, but it will update the parameter
    `markers`.
    """
    logging.info("Looking for entities in %r", rset)

    numbuckets = config['processes'];
    if numbuckets < 0:
        numbuckets = multiprocessing.cpu_count()

    if numbuckets == 1:
        for (x, z, mtime) in rset.iterate_chunks():
            try:
                data = rset.get_chunk(x, z)
                for poi in itertools.chain(data['TileEntities'], data['Entities']):
                    if poi['id'] == 'Sign' or poi['id'] == 'minecraft:sign': # kill me
                        poi = signWrangler(poi)
                    for name, __, filter_function, __, __, __ in filters:
                        result = filter_function(poi)
                        if result:
                            d = create_marker_from_filter_result(poi, result)
                            markers[name]['raw'].append(d)
            except nbt.CorruptChunkError:
                logging.warning("Ignoring POIs in corrupt chunk %d,%d", x,z)
  
    else:
        buckets = [[] for i in range(numbuckets)];
  
        for (x, z, mtime) in rset.iterate_chunks():
            i = x / 32 + z / 32
            i = i % numbuckets 
            buckets[i].append([x, z])
  
        for b in buckets:
            logging.info("Buckets has %d entries", len(b));
  
        # Create a pool of processes and run all the functions
        pool = Pool(processes=numbuckets, initializer=initBucketChunks, initargs=(config_path,))

        # simplify the filters dict, so pickle doesn't have to do so much
        filters = [(name, filter_function.__name__) for name, __, filter_function, __, __, __ in filters]

        results = pool.map(parseBucketChunks, ((buck, rset, filters) for buck in buckets))
  
        logging.info("All the threads completed")
  
        for marker_dict in results:
            for name, marker_list in marker_dict.iteritems():
                markers[name]['raw'].extend(marker_list)

    logging.info("Done.")


class PlayerDict(dict):
    use_uuid = False
    _name = ''
    uuid_cache = None # A cache for the UUID->profile lookups
    
    @classmethod
    def load_cache(cls, outputdir):
        cache_file = os.path.join(outputdir, "uuidcache.dat")
        if os.path.exists(cache_file):
            try:
                with closing(gzip.GzipFile(cache_file)) as gz:
                    cls.uuid_cache = json.load(gz)
                    logging.info("Loaded UUID cache from %r with %d entries",
                                 cache_file, len(cls.uuid_cache.keys()))
            except (ValueError, IOError):
                logging.warning("Failed to load UUID cache -- it might be corrupt")
                cls.uuid_cache = {}
                corrupted_cache = cache_file + ".corrupted." + datetime.datetime.now().isoformat()
                try:
                    os.rename(cache_file, corrupted_cache)
                    logging.warning("If %s does not appear to contain meaningful data, you may safely delete it", corrupted_cache)
                except OSError:
                    logging.warning("Failed to backup corrupted UUID cache")

                logging.info("Initialized an empty UUID cache")
        else:
            cls.uuid_cache = {}
            logging.info("Initialized an empty UUID cache")

    @classmethod
    def save_cache(cls, outputdir):
        cache_file = os.path.join(outputdir, "uuidcache.dat")
        caps = get_fs_caps(outputdir)

        with FileReplacer(cache_file, caps) as cache_file_name:
            with closing(gzip.GzipFile(cache_file_name, "wb")) as gz:
                json.dump(cls.uuid_cache, gz)
                logging.info("Wrote UUID cache with %d entries",
                             len(cls.uuid_cache.keys()))

    def __getitem__(self, item):
        if item == "EntityId":
            if not super(PlayerDict, self).has_key("EntityId"):
                if self.use_uuid:
                    super(PlayerDict, self).__setitem__("EntityId", self.get_name_from_uuid())
                else:
                    super(PlayerDict, self).__setitem__("EntityId", self._name)
        
        return super(PlayerDict, self).__getitem__(item)

    def get_name_from_uuid(self):
        sname = self._name.replace('-','')
        try:
            profile = PlayerDict.uuid_cache[sname]
            if profile['retrievedAt'] > time.mktime(self['time']):
                return profile['name']
        except (KeyError,):
            pass

        try:
            profile = json.loads(urllib2.urlopen(UUID_LOOKUP_URL + sname).read())
            if 'name' in profile:
                profile['retrievedAt'] = time.mktime(time.localtime())
                PlayerDict.uuid_cache[sname] = profile
                return profile['name']
        except (ValueError, urllib2.URLError):
            logging.warning("Unable to get player name for UUID %s", self._name)


def handlePlayers(worldpath, filters, markers):
    """
    Add markers for players to the list of markers.

    For this the player files under the given `worldpath` are parsed and
    filtered.
    This function will not return anything, but it will update the parameter
    `markers`.
    """
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

    for playerfile in playerfiles:
        try:
            data = PlayerDict(nbt.load(os.path.join(playerdir, playerfile))[1])
            data.use_uuid = useUUIDs
            if isSinglePlayer:
                data = data['Data']['Player']
        except (IOError, TypeError):
            logging.warning("Skipping bad player dat file %r", playerfile)
            continue

        playername = playerfile.split(".")[0]
        if isSinglePlayer:
            playername = 'Player'
        data._name = playername
        if useUUIDs:
            data['uuid'] = playername

        # Position at last logout
        data['id'] = "Player"
        data['x'] = int(data['Pos'][0])
        data['y'] = int(data['Pos'][1])
        data['z'] = int(data['Pos'][2])
        # Time at last logout, calculated from last time the player's file was modified
        data['time'] = time.localtime(os.path.getmtime(os.path.join(playerdir, playerfile)))

        # Spawn position (bed or main spawn)
        if "SpawnX" in data:
            # Spawn position (bed or main spawn)
            spawn = PlayerDict()
            spawn.use_uuid = useUUIDs
            spawn._name = playername
            spawn["id"] = "PlayerSpawn"
            spawn["x"] = data['SpawnX']
            spawn["y"] = data['SpawnY']
            spawn["z"] = data['SpawnZ']

        for name, __, filter_function, rset, __, __ in filters:
            # get the dimension for the filter
            # This has do be done every time, because we have filters for
            # different regionsets.

            if rset.get_type():
                dimension = int(re.match(r"^DIM(_MYST)?(-?\d+)$", rset.get_type()).group(2))
            else:
                dimension = 0

            if data['Dimension'] == dimension:
                result = filter_function(data)
                if result:
                    d = create_marker_from_filter_result(data, result)
                    markers[name]['raw'].append(d)

            if dimension == 0 and "SpawnX" in data:
                result = filter_function(spawn)
                if result:
                    d = create_marker_from_filter_result(spawn, result)
                    markers[name]['raw'].append(d)


def handleManual(manualpois, filters, markers):
    """
    Add markers for manually defined POIs to the list of markers.

    This function will not return anything, but it will update the parameter
    `markers`.
    """
    for poi in manualpois:
        for name, __, filter_function, __, __, __ in filters:
            result = filter_function(poi)
            if result:
                d = create_marker_from_filter_result(poi, result)
                markers[name]['raw'].append(d)


def create_marker_from_filter_result(poi, result):
    """
    Takes a POI and the return value of a filter function for it and creates a
    marker dict depending on the type of the returned value.
    """
    # every marker has a position either directly via attributes x, y, z or
    # via tuple attribute Pos
    if 'Pos' in poi:
        d = dict((v, poi['Pos'][i]) for i, v in enumerate('xyz'))
    else:
        d = dict((v, poi[v]) for v in 'xyz')

    # read some Defaults from POI
    if "icon" in poi:
        d["icon"] = poi['icon']
    if "createInfoWindow" in poi:
        d["createInfoWindow"] = poi['createInfoWindow']

    # Fill in the rest from result
    if isinstance(result, basestring):
        d.update(dict(text=result, hovertext=result))
    elif isinstance(result, tuple):
        d.update(dict(text=result[1], hovertext=result[0]))
    # Dict support to allow more flexible things in the future as well as polylines on the map.
    elif isinstance(result, dict):
        d['text'] = result['text']

        # Use custom hovertext if provided...
        if 'hovertext' in result:
            d['hovertext'] = unicode(result['hovertext'])
        else: # ...otherwise default to display text.
            d['hovertext'] = result['text']

        if 'polyline' in result and hasattr(result['polyline'], '__iter__'):
            d['polyline'] = []
            for point in result['polyline']:
                d['polyline'].append(dict(x=point['x'], y=point['y'], z=point['z'])) # point.copy() would work, but this validates better
            if isinstance(result['color'], basestring):
                d['strokeColor'] = result['color']

            if "icon" in result:
                d["icon"] = result['icon']
            if "createInfoWindow" in result:
                d["createInfoWindow"] = result['createInfoWindow']
    else:
        raise ValueError("got an %s as result for POI with id %s" % (type(result).__name__, poi['id']))

    return d


def main():

    if os.path.basename(sys.argv[0]) == """genPOI.py""":
        helptext = """genPOI.py
            %prog --config=<config file> [options]"""
    else:
        helptext = """genPOI
            %prog --genpoi --config=<config file> [options]"""

    logger.configure()

    parser = OptionParser(usage=helptext)
    parser.add_option("-c", "--config", dest="config", action="store",
                      help="Specify the config file to use.")
    parser.add_option("-q", "--quiet", dest="quiet", action="count",
                      help="Reduce logging output")
    parser.add_option("--skip-scan", dest="skipscan", action="store_true",
                      help="Skip scanning for entities when using GenPOI")
    parser.add_option("--skip-players", dest="skipplayers", action="store_true",
                      help="Skip getting player data when using GenPOI")

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

    filters = set()
    marker_groups = defaultdict(list)

    # collect all filters and get regionsets
    for rname, render in config['renders'].iteritems():
        # Convert render['world'] to the world path, and store the original
        # in render['worldname_orig']
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

        # get the regionset for this dimension
        rset = w.get_regionset(render['dimension'][1])
        if rset == None: # indicates no such dimension was found:
            logging.warn("Sorry, you requested dimension '%s' for the render '%s', but I couldn't find it", render['dimension'][0], rname)
            continue

        # find filters for this render
        for f in render['markers']:
            # internal identifier for this filter
            name = replaceBads(f['name']) + hex(hash(f['filterFunction']))[-4:] + "_" + hex(hash(rname))[-4:]

            # add it to the list of filters
            filters.add((name, f['name'], f['filterFunction'], rset, worldpath, rname))

            # add an entry in the menu to show markers found by this filter
            group = dict(groupName=name,
                    displayName = f['name'], 
                    icon=f.get('icon', 'signpost_icon.png'), 
                    createInfoWindow=f.get('createInfoWindow', True),
                    checked = f.get('checked', False))
            marker_groups[rname].append(group)

    # initialize the structure for the markers
    markers = dict((name, dict(created=False, raw=[], name=filter_name))
                    for name, filter_name, __, __, __, __ in filters)

    # apply filters to regionsets
    if not options.skipscan:
        # group filters by rset
        keyfunc = lambda x: x[3]
        sfilters = sorted(filters, key=keyfunc)
        for rset, rset_filters in itertools.groupby(sfilters, keyfunc):
            handleEntities(rset, config, options.config, list(rset_filters), markers)

    # apply filters to players
    if not options.skipplayers:
        PlayerDict.load_cache(destdir)
        # group filters by worldpath, so we only search for players once per
        # world
        keyfunc = lambda x: x[4]
        sfilters = sorted(filters, key=keyfunc)
        for worldpath, worldpath_filters in itertools.groupby(sfilters, keyfunc):
            handlePlayers(worldpath, list(worldpath_filters), markers)

    # add manual POIs
    # group filters by name of the render, because only filter functions for
    # the current render should be used on the current render's manualpois
    keyfunc = lambda x: x[5]
    sfilters = sorted(filters, key=keyfunc)
    for rname, rname_filters in itertools.groupby(sfilters, keyfunc):
        manualpois = config['renders'][rname]['manualpois']
        handleManual(manualpois, list(rname_filters), markers)

    logging.info("Done handling POIs")
    logging.info("Writing out javascript files")

    if not options.skipplayers:
        PlayerDict.save_cache(destdir)

    with open(os.path.join(destdir, "markersDB.js"), "w") as output:
        output.write("var markersDB=")
        json.dump(markers, output, indent=2)
        output.write(";\n");
    with open(os.path.join(destdir, "markers.js"), "w") as output:
        output.write("var markers=")
        json.dump(marker_groups, output, indent=2)
        output.write(";\n");
    with open(os.path.join(destdir, "baseMarkers.js"), "w") as output:
        output.write("overviewer.util.injectMarkerScript('markersDB.js');\n")
        output.write("overviewer.util.injectMarkerScript('markers.js');\n")
        output.write("overviewer.util.injectMarkerScript('regions.js');\n")
        output.write("overviewer.collections.haveSigns=true;\n")
    logging.info("Done")

if __name__ == "__main__":
    main()
