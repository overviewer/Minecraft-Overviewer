#    This file is part of the Minecraft Overviewer.
#
#    Minecraft Overviewer is free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or (at
#    your option) any later version.
#
#    Minecraft Overviewer is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#    Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with the Overviewer.  If not, see <http://www.gnu.org/licenses/>.

import functools
import os
import os.path
import logging
import time
import random
import re
import locale

import numpy

from . import nbt
from . import cache

"""
This module has routines for extracting information about available worlds

"""

class ChunkDoesntExist(Exception):
    pass


class UnsupportedVersion(Exception):
    pass


def log_other_exceptions(func):
    """A decorator that prints out any errors that are not
    ChunkDoesntExist errors. This should decorate any functions or
    methods called by the C code, such as get_chunk(), because the C
    code is likely to swallow exceptions. This will at least make them
    visible.
    """
    functools.wraps(func)
    def newfunc(*args):
        try:
            return func(*args)
        except ChunkDoesntExist:
            raise
        except Exception, e:
            logging.exception("%s raised this exception", func.func_name)
            raise
    return newfunc


class World(object):
    """Encapsulates the concept of a Minecraft "world". A Minecraft world is a
    level.dat file, a players directory with info about each player, a data
    directory with info about that world's maps, and one or more "dimension"
    directories containing a set of region files with the actual world data.

    This class deals with reading all the metadata about the world.  Reading
    the actual world data for each dimension from the region files is handled
    by a RegionSet object.

    Note that vanilla Minecraft servers and single player games have a single
    world with multiple dimensions: one for the overworld, the nether, etc.

    On Bukkit enabled servers, to support "multiworld," the server creates
    multiple Worlds, each with a single dimension.

    In this file, the World objects act as an interface for RegionSet objects.
    The RegionSet objects are what's really important and are used for reading
    block data for rendering.  A RegionSet object will always correspond to a
    set of region files, or what is colloquially referred to as a "world," or
    more accurately as a dimension.

    The only thing this class actually stores is a list of RegionSet objects
    and the parsed level.dat data

    """
    
    def __init__(self, worlddir):
        self.worlddir = worlddir

        # This list, populated below, will hold RegionSet files that are in
        # this world
        self.regionsets = []
       
        # The level.dat file defines a minecraft world, so assert that this
        # object corresponds to a world on disk
        if not os.path.exists(os.path.join(self.worlddir, "level.dat")):
            raise ValueError("level.dat not found in %s" % self.worlddir)

        data = nbt.load(os.path.join(self.worlddir, "level.dat"))[1]['Data']
        # it seems that reading a level.dat file is unstable, particularly with respect
        # to the spawnX,Y,Z variables.  So we'll try a few times to get a good reading
        # empirically, it seems that 0,50,0 is a "bad" reading
        # update: 0,50,0 is the default spawn, and may be valid is some cases
        # more info is needed
        data = nbt.load(os.path.join(self.worlddir, "level.dat"))[1]['Data']
            

        # Hard-code this to only work with format version 19133, "Anvil"
        if not ('version' in data and data['version'] == 19133):
            if 'version' in data and data['version'] == 0:
                logging.debug("Note: Allowing a version of zero in level.dat!")
                ## XXX temporary fix for #1194
            else:
                raise UnsupportedVersion(
                    ("Sorry, This version of Minecraft-Overviewer only works "
                     "with the 'Anvil' chunk format\n"
                     "World at %s is not compatible with Overviewer")
                    % self.worlddir)

        # This isn't much data, around 15 keys and values for vanilla worlds.
        self.leveldat = data


        # Scan worlddir to try to identify all region sets. Since different
        # server mods like to arrange regions differently and there does not
        # seem to be any set standard on what dimensions are in each world,
        # just scan the directory heirarchy to find a directory with .mca
        # files.
        for root, dirs, files in os.walk(self.worlddir, followlinks=True):
            # any .mcr files in this directory?
            mcas = [x for x in files if x.endswith(".mca")]
            if mcas:
                # construct a regionset object for this
                rel = os.path.relpath(root, self.worlddir)
                rset = RegionSet(root, rel)
                if root == os.path.join(self.worlddir, "region"):
                    self.regionsets.insert(0, rset)
                else:
                    self.regionsets.append(rset)
        
        # TODO move a lot of the following code into the RegionSet


        try:
            # level.dat should have the LevelName attribute so we'll use that
            self.name = data['LevelName']
        except KeyError:
            # but very old ones might not? so we'll just go with the world dir name if they don't
            self.name = os.path.basename(os.path.realpath(self.worlddir))
        
        try:
            # level.dat also has a RandomSeed attribute
            self.seed = data['RandomSeed']
        except KeyError:
            self.seed = 0 # oh well
       
        # TODO figure out where to handle regionlists

    def get_regionsets(self):
        return self.regionsets
    def get_regionset(self, index):
        if type(index) == int:
            return self.regionsets[index]
        else: # assume a get_type() value
            candids = [x for x in self.regionsets if x.get_type() == index]
            logging.debug("You asked for %r, and I found the following candids: %r", index, candids)
            if len(candids) > 0:
                return candids[0]
            else: 
                return None


    def get_level_dat_data(self):
        # Return a copy
        return dict(self.data)
      
    def find_true_spawn(self):
        """Returns the spawn point for this world. Since there is one spawn
        point for a world across all dimensions (RegionSets), this method makes
        sense as a member of the World class.
        
        Returns (x, y, z)
        
        """
        # The spawn Y coordinate is almost always the default of 64.  Find the
        # first air block above the stored spawn location for the true spawn
        # location

        ## read spawn info from level.dat
        data = self.leveldat
        disp_spawnX = spawnX = data['SpawnX']
        spawnY = data['SpawnY']
        disp_spawnZ = spawnZ = data['SpawnZ']
   
        ## The chunk that holds the spawn location 
        chunkX = spawnX//16
        chunkZ = spawnZ//16
        
        ## clamp spawnY to a sane value, in-chunk value
        if spawnY < 0:
            spawnY = 0
        if spawnY > 255:
            spawnY = 255
        
        # Open up the chunk that the spawn is in
        regionset = self.get_regionset(None)
        if not regionset:
            return None
        try:
            chunk = regionset.get_chunk(chunkX, chunkZ)
        except ChunkDoesntExist:
            return (spawnX, spawnY, spawnZ)
    
        def getBlock(y):
            "This is stupid and slow but I don't care"
            targetSection = spawnY//16
            for section in chunk['Sections']:
                if section['Y'] == targetSection:
                    blockArray = section['Blocks']
                    return blockArray[inChunkX, inChunkZ, y % 16]
            return 0



        ## The block for spawn *within* the chunk
        inChunkX = spawnX - (chunkX*16)
        inChunkZ = spawnZ - (chunkZ*16)

        ## find the first air block
        while (getBlock(spawnY) != 0) and spawnY < 256:
            spawnY += 1

        return spawnX, spawnY, spawnZ

class RegionSet(object):
    """This object is the gateway to a particular Minecraft dimension within a
    world. It corresponds to a set of region files containing the actual
    world data. This object has methods for parsing and returning data from the
    chunks from its regions.

    See the docs for the World object for more information on the difference
    between Worlds and RegionSets.


    """

    def __init__(self, regiondir, rel):
        """Initialize a new RegionSet to access the region files in the given
        directory.

        regiondir is a path to a directory containing region files.
        
        rel is the relative path of this directory, with respect to the
        world directory.

        cachesize, if specified, is the number of chunks to keep parsed and
        in-memory.

        """
        self.regiondir = os.path.normpath(regiondir)
        self.rel = os.path.normpath(rel)
        logging.debug("regiondir is %r" % self.regiondir)
        logging.debug("rel is %r" % self.rel)
        
        # we want to get rid of /regions, if it exists
        if self.rel.endswith(os.path.normpath("/region")):
            self.type = self.rel[0:-len(os.path.normpath("/region"))]
        elif self.rel == "region":
            # this is the main world
            self.type = None
        else:
            logging.warning("Unkown region type in %r", regiondir)
            self.type = "__unknown"

        logging.debug("Scanning regions.  Type is %r" % self.type)
        
        # This is populated below. It is a mapping from (x,y) region coords to filename
        self.regionfiles = {}

        # This holds a cache of open regionfile objects
        self.regioncache = cache.LRUCache(size=16, destructor=lambda regionobj: regionobj.close())
        
        for x, y, regionfile in self._iterate_regionfiles():
            # regionfile is a pathname
            self.regionfiles[(x,y)] = (regionfile, os.path.getmtime(regionfile))

        self.empty_chunk = [None,None]
        logging.debug("Done scanning regions")

        self._blockmap = {
            'minecraft:air': (0, 0),
            'minecraft:cave_air': (0, 0),
            'minecraft:stone': (1, 0),
            'minecraft:granite': (1, 1),
            'minecraft:polished_granite': (1, 2),
            'minecraft:diorite': (1, 3),
            'minecraft:polished_diorite': (1, 4),
            'minecraft:andesite': (1, 5),
            'minecraft:polished_andesite': (1, 6),
            'minecraft:grass_block': (2, 0),
            'minecraft:dirt': (3, 0),
            'minecraft:coarse_dirt': (3, 1),
            'minecraft:podzol': (3, 2),
            'minecraft:cobblestone': (4, 0),
            'minecraft:oak_planks': (5, 0),
            'minecraft:spruce_planks': (5, 1),
            'minecraft:birch_planks': (5, 2),
            'minecraft:jungle_planks': (5, 3),
            'minecraft:acacia_planks': (5, 4),
            'minecraft:dark_oak_planks': (5, 5),
            'minecraft:oak_sapling': (6, 0),
            'minecraft:spruce_sapling': (6, 1),
            'minecraft:birch_sapling': (6, 2),
            'minecraft:jungle_sapling': (6, 3),
            'minecraft:acacia_sapling': (6, 4),
            'minecraft:dark_oak_sapling': (6, 5),
            'minecraft:bedrock': (7, 0),
            'minecraft:water': (8, 0),
            'minecraft:lava': (10, 0),
            'minecraft:sand': (12, 0),
            'minecraft:red_sand': (12, 1),
            'minecraft:gravel': (13, 0),
            'minecraft:gold_ore': (14, 0),
            'minecraft:iron_ore': (15, 0),
            'minecraft:coal_ore': (16, 0),
            'minecraft:oak_log': (17, 0),
            'minecraft:spruce_log': (17, 1),
            'minecraft:birch_log': (17, 2),
            'minecraft:jungle_log': (17, 3),
            'minecraft:oak_leaves': (18, 0),
            'minecraft:spruce_leaves': (18, 1),
            'minecraft:birch_leaves': (18, 2),
            'minecraft:jungle_leaves': (18, 3),
            'minecraft:acacia_leaves': (18, 4),
            'minecraft:dark_oak_leaves': (18, 5),
            'minecraft:sponge': (19, 0),
            'minecraft:wet_sponge': (19, 1),
            'minecraft:glass': (20, 0),
            'minecraft:lapis_ore': (21, 0),
            'minecraft:lapis_block': (22, 0),
            'minecraft:dispenser': (23, 0),
            'minecraft:sandstone': (24, 0),
            'minecraft:cut_sandstone': (24, 2),
            'minecraft:chiseled_sandstone': (24, 3),

            'minecraft:white_bed': (26, 0),
            'minecraft:orange_bed': (26, 0),
            'minecraft:magenta_bed': (26, 0),
            'minecraft:light_blue_bed': (26, 0),
            'minecraft:yellow_bed': (26, 0),
            'minecraft:lime_bed': (26, 0),
            'minecraft:pink_bed': (26, 0),
            'minecraft:gray_bed': (26, 0),
            'minecraft:light_gray_bed': (26, 0),
            'minecraft:cyan_bed': (26, 0),
            'minecraft:purple_bed': (26, 0),
            'minecraft:blue_bed': (26, 0),
            'minecraft:brown_bed': (26, 0),
            'minecraft:green_bed': (26, 0),
            'minecraft:red_bed': (26, 0),
            'minecraft:black_bed': (26, 0),
            'minecraft:detector_rail': (28, 0),
            'minecraft:sticky_piston': (29, 0),
            'minecraft:cobweb': (30, 0),
            'minecraft:dead_bush': (31, 0),
            'minecraft:grass': (31, 1),
            'minecraft:fern': (31, 2),
            'minecraft:piston': (33, 0),
            'minecraft:piston_head': (34, 0),
            'minecraft:white_wool': (35, 0),
            'minecraft:orange_wool': (35, 1),
            'minecraft:magenta_wool': (35, 2),
            'minecraft:light_blue_wool': (35, 3),
            'minecraft:yellow_wool': (35, 4),
            'minecraft:lime_wool': (35, 5),
            'minecraft:pink_wool': (35, 6),
            'minecraft:gray_wool': (35, 7),
            'minecraft:light_gray_wool': (35, 8),
            'minecraft:cyan_wool': (35, 9),
            'minecraft:purple_wool': (35, 10),
            'minecraft:blue_wool': (35, 11),
            'minecraft:brown_wool': (35, 12),
            'minecraft:green_wool': (35, 13),
            'minecraft:red_wool': (35, 14),
            'minecraft:black_wool': (35, 15),

            'minecraft:poppy': (38, 0),
            'minecraft:blue_orchid': (38, 1),
            'minecraft:allium': (38, 2),
            'minecraft:azure_bluet': (38, 3),
            'minecraft:red_tulip': (38, 4),
            'minecraft:orange_tulip': (38, 5),
            'minecraft:white_tulip': (38, 6),
            'minecraft:pink_tulip': (38, 7),
            'minecraft:oxeye_daisy': (38, 8),
            'minecraft:dandelion': (38, 9),
            'minecraft:brown_mushroom': (39, 0),	
            'minecraft:red_mushroom': (40, 0),
            'minecraft:gold_block': (41, 0),
            'minecraft:iron_block': (42, 0),
            'minecraft:stone_slab': (44, 0),
            'minecraft:sandstone_slab': (44, 1),
            'minecraft:wood_slab': (44, 2),
            'minecraft:cobblestone_slab': (44, 3),
            'minecraft:brick_slab': (44, 4),
            'minecraft:stone_brick_slab': (44, 5),
            'minecraft:nether_brick_slab': (44, 6),
            'minecraft:quartz_slab': (44, 7),
            'minecraft:bricks': (45, 0),
            'minecraft:tnt': (46, 0),
            'minecraft:bookshelf': (47, 0),
            'minecraft:obsidian': (49, 0),
            'minecraft:wall_torch': (50, 0),
            'minecraft:torch': (50, 5),
            'minecraft:fire': (51, 0),
            'minecraft:spawner': (52, 0),
            'minecraft:oak_stairs': (53, 0),
            'minecraft:chest': (54, 0),
            'minecraft:redstone_wire': (55, 0),
            'minecraft:diamond_ore': (56, 0),
            'minecraft:diamond_block': (57, 0),
            'minecraft:crafting_table': (58, 0),
            'minecraft:wheat': (59, 0),
            'minecraft:farmland': (60, 0),
            'minecraft:furnace': (61, 0),
            'minecraft:standing_sign': (63, 0),
            'minecraft:oak_door': (64, 0),
            'minecraft:ladder': (65, 3), # todo: incorporate facing
            'minecraft:rail': (66, 0),
            'minecraft:stone_stairs': (67, 0),
            'minecraft:cobblestone_stairs': (67, 0),
            'minecraft:wall_sign': (68, 0),
            'minecraft:lever': (69, 0),
            'minecraft:stone_pressure_plate': (70, 0),
            'minecraft:iron_door': (71, 0),
            'minecraft:oak_pressure_plate': (72, 0),
            'minecraft:redstone_ore': (73, 0),
            'minecraft:redstone_torch': (76, 0),
            'minecraft:stone_button': (77, 0),
            'minecraft:snow': (78, 0),
            'minecraft:ice': (79, 0),
            'minecraft:snow_block': (80, 0),
            'minecraft:cactus': (81, 0),
            'minecraft:clay': (82, 0),
            'minecraft:sugar_cane': (83, 0),
            'minecraft:jukebox': (84, 0),
            'minecraft:oak_fence': (85, 0),
            'minecraft:pumpkin': (86, 0),
            'minecraft:netherrack': (87, 0),
            'minecraft:soul_sand': (88, 0),
            'minecraft:glowstone': (89, 0),
            'minecraft:nether_portal': (90, 0),
            'minecraft:jack_o_lantern': (91, 0),
            'minecraft:cake': (92, 0),
            'minecraft:repeater': (93,0), # temporarily map to inactive repeater
            'minecraft:oak_trapdoor': (96, 0),
            'minecraft:spruce_trapdoor': (96, 0), #wrong
            'minecraft:birch_trapdoor': (96, 0),
            'minecraft:jungle_trapdoor': (96, 0),
            'minecraft:acacia_trapdoor': (96, 0),
            'minecraft:dark_oak_trapdoor': (96, 0),
            'minecraft:infested_stone': (97, 0),
            'minecraft:stone_bricks': (98, 0),
            'minecraft:infested_stone_bricks': (98, 0),
            'minecraft:mossy_stone_bricks': (98, 1),
            'minecraft:cracked_stone_bricks': (98, 2),
            'minecraft:chiseled_stone_bricks': (98, 3),
            'minecraft:infested_chiseled_stone_bricks': (98, 3),
            'minecraft:brown_mushroom_block': (99, 0),
            'minecraft:red_mushroom_block': (100, 0),
            'minecraft:iron_bars': (101, 0),
            'minecraft:glass_pane': (102, 0),
            'minecraft:attached_melon_stem': (104, 0),
            'minecraft:pumpkin_stem': (104, 0),
            'minecraft:melon_stem': (105, 0),
            'minecraft:vine': (106, 0),
            'minecraft:brick_stairs': (108, 0),
            'minecraft:stone_brick_stairs': (109, 0),
            'minecraft:mycelium': (110, 0),
            'minecraft:lily_pad': (111, 0),
            'minecraft:nether_bricks': (112, 0),
            'minecraft:nether_brick_fence': (113, 0),
            'minecraft:nether_brick_stairs': (114, 0),
            'minecraft:nether_wart': (115, 0),
            'minecraft:enchanting_table': (116, 0),
            'minecraft:brewing_stand': (117, 0),
            'minecraft:end_portal': (119, 0),
            'minecraft:end_portal_frame': (120, 0),
            'minecraft:end_stone': (121, 0),
            'minecraft:redstone_lamp': (123, 0),
            'minecraft:oak_slab': (126, 0),
            'minecraft:spruce_slab': (126, 1),
            'minecraft:birch_slab': (126, 2),
            'minecraft:jungle_slab': (126, 3),
            'minecraft:acacia_slab': (126, 4),
            'minecraft:dark_oak_slab': (126, 5),
            'minecraft:cocoa': (127, 0),
            'minecraft:sandstone_stairs': (128, 0),
            'minecraft:emerald_ore': (129, 0),
            'minecraft:ender_chest': (130, 0),
            'minecraft:tripwire': (131, 0),
            'minecraft:tripwire_hook': (132, 0),
            'minecraft:emerald_block': (133, 0),
            'minecraft:spruce_stairs': (134, 0),
            'minecraft:birch_stairs': (135, 0),
            'minecraft:jungle_stairs': (136, 0),
            'minecraft:command_block': (137, 0),
            'minecraft:beacon': (138, 0),
            'minecraft:mossy_cobblestone': (139, 16),
            'minecraft:cobblestone_wall': (139, 0),
            'minecraft:flower_pot': (140, 0),
            'minecraft:carrots': (141, 0),
            'minecraft:potatoes': (142, 0),
            'minecraft:anvil': (145, 0),
            'minecraft:trapped_chest': (146, 0),
            'minecraft:light_weighted_pressure_plate': (147, 0),
            'minecraft:heavy_weighted_pressure_plate': (148, 0),
            'minecraft:daylight_detector': (151, 0),
            'minecraft:redstone_block': (152, 0),
            'minecraft:nether_quartz_ore': (153, 0),
            'minecraft:hopper': (154, 0),
            'minecraft:smooth_quartz': (155, 0),
            'minecraft:quartz_block': (155, 0),
            'minecraft:quartz_stairs': (156, 0),
            'minecraft:activator_rail': (157, 0),
            'minecraft:dropper': (158, 0),
            'minecraft:white_terracotta': (159, 0),
            'minecraft:orange_terracotta': (159, 1),
            'minecraft:magenta_terracotta': (159, 2),
            'minecraft:light_blue_terracotta': (159, 3),
            'minecraft:yellow_terracotta': (159, 4),
            'minecraft:lime_terracotta': (159, 5),
            'minecraft:pink_terracotta': (159, 6),
            'minecraft:gray_terracotta': (159, 7),
            'minecraft:silver_terracotta': (159, 8),
            'minecraft:cyan_terracotta': (159, 9),
            'minecraft:purple_terracotta': (159, 10),
            'minecraft:blue_terracotta': (159, 11),
            'minecraft:brown_terracotta': (159, 12),
            'minecraft:green_terracotta': (159, 13),
            'minecraft:red_terracotta': (159, 14),
            'minecraft:black_terracotta': (159, 15),
            'minecraft:acacia_log': (162, 0),
            'minecraft:dark_oak_log': (162, 1),
            'minecraft:acacia_stairs': (163, 0),
            'minecraft:dark_oak_stairs': (164, 0),
            'minecraft:slime_block': (165,0),
            'minecraft:iron_trapdoor': (167, 0),
            'minecraft:prismarine': (168, 0),
            'minecraft:dark_prismarine': (168, 2),
            'minecraft:prismarine_bricks': (168, 1),
            'minecraft:sea_lantern': (169, 0),
            'minecraft:hay_block': (170, 0),
            'minecraft:white_carpet': (171, 0),
            'minecraft:orange_carpet': (171, 1),
            'minecraft:magenta_carpet': (171, 2),
            'minecraft:light_blue_carpet': (171, 3),
            'minecraft:yellow_carpet': (171, 4),
            'minecraft:lime_carpet': (171, 5),
            'minecraft:pink_carpet': (171, 6),
            'minecraft:gray_carpet': (171, 7),
            'minecraft:light_gray_carpet': (171, 8),
            'minecraft:cyan_carpet': (171, 9),
            'minecraft:purple_carpet': (171, 10),
            'minecraft:blue_carpet': (171, 11),
            'minecraft:brown_carpet': (171, 12),
            'minecraft:green_carpet': (171, 13),
            'minecraft:red_carpet': (171, 14),
            'minecraft:black_carpet': (171, 15),
            'minecraft:terracotta': (172, 0),
            'minecraft:coal_block': (173, 0),
            'minecraft:packed_ice': (174, 0),
            'minecraft:blue_ice': (174, 0), # close enough
            'minecraft:sunflower': (175, 0),
            'minecraft:lilac': (175, 1),
            'minecraft:tall_grass': (175, 2),
            'minecraft:large_fern': (175, 3),
            'minecraft:rose_bush': (175, 4),
            'minecraft:peony': (175, 5),
            'minecraft:red_sandstone': (179, 0),
            'minecraft:red_sandstone_stairs': (180, 0),
            'minecraft:spruce_fence_gate': (183, 0),
            'minecraft:birch_fence_gate': (184, 0),
            'minecraft:jungle_fence_gate': (185, 0),
            'minecraft:dark_oak_fence_gate': (186, 0),
            'minecraft:acacia_fence_gate': (187, 0),
            'minecraft:spruce_fence': (188, 0),
            'minecraft:birch_fence': (189, 0),
            'minecraft:jungle_fence': (190, 0),
            'minecraft:dark_oak_fence': (191, 0),
            'minecraft:acacia_fence': (192, 0),
            'minecraft:spruce_door': (193, 0),
            'minecraft:birch_door': (194, 0),
            'minecraft:jungle_door': (195, 0),
            'minecraft:acacia_door': (196, 0),
            'minecraft:dark_oak_door': (197, 0),
            'minecraft:chorus_plant': (199, 0),
            'minecraft:chorus_flower': (200, 0),
            'minecraft:purpur_block': (201, 0),
            'minecraft:purpur_stairs': (203, 0),
            'minecraft:grass_path': (208, 0),
            'minecraft:end_rod': (208, 0),
            'minecraft:frosted_ice': (212, 0),
            'minecraft:magma_block': (213, 0),
            'minecraft:nether_wart_block': (214, 0),
            'minecraft:red_nether_bricks': (215, 0),
            'minecraft:bone_block': (216, 0),
            'minecraft:white_shulker_box': (218, 0),
            'minecraft:orange_shulker_box': (218, 1),
            'minecraft:magenta_shulker_box': (218, 2),
            'minecraft:light_blue_shulker_box': (218, 3),
            'minecraft:yellow_shulker_box': (218, 4),
            'minecraft:lime_shulker_box': (218, 5),
            'minecraft:pink_shulker_box': (218, 6),
            'minecraft:gray_shulker_box': (218, 7),
            'minecraft:light_gray_shulker_box': (218, 8),
            'minecraft:cyan_shulker_box': (218, 9),
            'minecraft:purple_shulker_box': (218, 10),
            'minecraft:blue_shulker_box': (218, 11),
            'minecraft:brown_shulker_box': (218, 12),
            'minecraft:green_shulker_box': (218, 13),
            'minecraft:red_shulker_box': (218, 14),
            'minecraft:black_shulker_box': (218, 15),
            'minecraft:white_glazed_terracotta': (235, 0),
            'minecraft:orange_glazed_terracotta': (236, 0),
            'minecraft:magenta_glazed_terracotta': (237, 0),
            'minecraft:light_blue_glazed_terracotta': (238, 0),
            'minecraft:yellow_glazed_terracotta': (239, 0),
            'minecraft:lime_glazed_terracotta': (240, 0),
            'minecraft:pink_glazed_terracotta': (241, 0),
            'minecraft:gray_glazed_terracotta': (242, 0),
            'minecraft:silver_glazed_terracotta': (243, 0),
            'minecraft:cyan_glazed_terracotta': (244, 0),
            'minecraft:purple_glazed_terracotta': (245, 0),
            'minecraft:blue_glazed_terracotta': (246, 0),
            'minecraft:brown_glazed_terracotta': (247, 0),
            'minecraft:green_glazed_terracotta': (248, 0),
            'minecraft:red_glazed_terracotta': (249, 0),
            'minecraft:black_glazed_terracotta': (250, 0),

            'minecraft:beetroots': (244, 0),
            
            'minecraft:structure_block': (255, 0),
            'minecraft:sign': (323, 0),
            'minecraft:comparator': (149, 0), # temporary map all comparators to inactive
            'minecraft:prismarine_shard': (409, 0),
            'minecraft:prismarine_crystals': (410, 0),
            'minecraft:rabbit_stew': (413, 0),
            'minecraft:rabbit_foot': (414, 0),
            'minecraft:rabbit_hide': (415, 0),
            'minecraft:armor_stand': (416, 0),
            'minecraft:mutton': (423, 0),
            # The following blocks are underwater and are not yet rendered.
            # To avoid spurious warnings, we'll treat them as air for now.
            'minecraft:brain_coral': (0, 0),
            'minecraft:brain_coral_block': (0, 0),
            'minecraft:brain_coral_fan': (0, 0),
            'minecraft:brain_coral_wall_fan': (0, 0),
            'minecraft:bubble_column': (0, 0),
            'minecraft:bubble_coral': (0, 0),
            'minecraft:bubble_coral_block': (0, 0),
            'minecraft:bubble_coral_fan': (0, 0),
            'minecraft:bubble_coral_wall_fan': (0, 0),
            'minecraft:fire_coral': (0, 0),
            'minecraft:fire_coral_block': (0, 0),
            'minecraft:fire_coral_fan': (0, 0),
            'minecraft:fire_coral_wall_fan': (0, 0),
            'minecraft:horn_coral': (0, 0),
            'minecraft:horn_coral_block': (0, 0),
            'minecraft:horn_coral_fan': (0, 0),
            'minecraft:horn_coral_wall_fan': (0, 0),
            'minecraft:kelp': (0, 0),
            'minecraft:kelp_plant': (0, 0),
            'minecraft:sea_pickle': (0, 0),
            'minecraft:seagrass': (0, 0),
            'minecraft:tall_seagrass': (0, 0),
            'minecraft:tube_coral': (0, 0),
            'minecraft:tube_coral_block': (0, 0),
            'minecraft:tube_coral_fan': (0, 0),
            'minecraft:tube_coral_wall_fan': (0, 0),
        }


        # The following blocks are underwater and are not yet rendered.
        # To avoid spurious warnings, we'll treat them as water for now.
        treat_as_water = [
             'brain_coral',  'brain_coral_block',  'brain_coral_fan',  'brain_coral_wall_fan',
            'bubble_coral', 'bubble_coral_block', 'bubble_coral_fan', 'bubble_coral_wall_fan',
              'fire_coral',   'fire_coral_block',   'fire_coral_fan',   'fire_coral_wall_fan',
              'horn_coral',   'horn_coral_block',   'horn_coral_fan',   'horn_coral_wall_fan',
              'tube_coral',   'tube_coral_block',   'tube_coral_fan',   'tube_coral_wall_fan',
            'kelp', 'kelp_plant', 'sea_pickle', 'seagrass', 'tall_seagrass',
            'bubble_column',
        ]
        for t in treat_as_water:
            self._blockmap['minecraft:%s' % t] = (8, 0)

        colors = [   'white', 'orange', 'magenta', 'light_blue',
                    'yellow',   'lime',    'pink',       'gray',
                'light_gray',   'cyan',  'purple',       'blue',
                     'brown',  'green',     'red',      'black']
        for i in range(len(colors)):
            self._blockmap['minecraft:%s_stained_glass_pane' % colors[i]] = (160, i)
            self._blockmap['minecraft:%s_concrete_powder'    % colors[i]] = (252, i)
            self._blockmap['minecraft:%s_stained_glass'      % colors[i]] = (95, i)
            self._blockmap['minecraft:%s_concrete'           % colors[i]] = (251, i)
            

    # Re-initialize upon unpickling
    def __getstate__(self):
        return (self.regiondir, self.rel)
    def __setstate__(self, state):
        return self.__init__(*state)

    def __repr__(self):
        return "<RegionSet regiondir=%r>" % self.regiondir

    def _get_block(self, palette_entry):
        key = palette_entry['Name']
        (block, data) = self._blockmap[key]
        if key == 'minecraft:redstone_ore':
            if palette_entry['Properties']['lit']:
                block += 1
        elif key == 'minecraft:grass_block':
            if palette_entry['Properties']['snowy']:
                data = 0x10
        elif key in ('minecraft:sunflower', 'minecraft:lilac', 'minecraft:tall_grass', 'minecraft:large_fern', 'minecraft:rose_bush', 'minecraft:peony'):
            if palette_entry['Properties']['half'] == 'upper':
                data |= 0x08
        elif key == 'minecraft_wheat':
            data = int(palette_entry['Properties']['age'])
        elif key == 'minecraft:wall_torch':
            facing = palette_entry['Properties']['facing']
            data = {'south': 1, 'north': 2, 'west': 3, 'east': 4}[facing]
        elif key == 'minecraft:nether_wart':
            data = int(palette_entry['Properties']['age'])
        elif key.endswith('_log'):
            axis = palette_entry['Properties']['axis']
            if axis == 'x':
                data |= 4
            elif axis == 'z':
                data |= 8
        elif key == 'minecraft:vine':
            p = palette_entry['Properties']
            if p['south'] == 'true': data |= 1
            if p['west']  == 'true': data |= 2
            if p['north'] == 'true': data |= 4
            if p['east']  == 'true': data |= 8
        elif key.endswith('_fence'):
            p = palette_entry['Properties']
            if p['north'] == 'true': data |= 1
            if p['west']  == 'true': data |= 2
            if p['south'] == 'true': data |= 4
            if p['east']  == 'true': data |= 8
        elif key.endswith('_stairs'):
            facing = palette_entry['Properties']['facing']
            if   facing == 'south': data |= 0x60
            elif facing == 'east':  data |= 0x30
            elif facing == 'north': data |= 0x18
            elif facing == 'west':  data |= 0x48
        elif key.endswith('_door'):
            p = palette_entry['Properties']
            if p['hinge'] == 'left': data |= 0x10
            if p['half'] == 'upper': data |= 0x08
            data |= {
                'north': 0x01,
                'west':  0x04,
                'south': 0x03,
                'east':  0x02,
               }[p['facing']]

        return (block, data)

    def get_type(self):
        """Attempts to return a string describing the dimension
        represented by this regionset.  Usually this is the relative
        path of the regionset within the world, minus the suffix
        /region, but for the main world it's None.
        """
        # path will be normalized in __init__
        return self.type

    def _get_regionobj(self, regionfilename):
        # Check the cache first. If it's not there, create the
        # nbt.MCRFileReader object, cache it, and return it
        # May raise an nbt.CorruptRegionError
        try:
            return self.regioncache[regionfilename]
        except KeyError:
            region = nbt.load_region(regionfilename)
            self.regioncache[regionfilename] = region
            return region

    def _packed_longarray_to_shorts(self, long_array, n):
        bits_per_value = (len(long_array) * 64) / n
        if bits_per_value < 4 or 12 < bits_per_value:
            raise nbt.CorruptChunkError()
        b = numpy.frombuffer(numpy.asarray(long_array, dtype=numpy.uint64), dtype=numpy.uint8)
        if bits_per_value == 8:
            result = b.astype(numpy.uint16)
        else:
            result = []
            i = 0
            # We will consume the byte array in chunks equal to bits_per_value.
            while i < len(b):
                if bits_per_value == 4:
                    for k in range(0, 4):
                        result.extend([
                             b[i + k] & 0x0f,
                            (b[i + k] & 0xf0) >> 4,
                            ])
                    i += 4
                elif bits_per_value == 5:
                    result.extend([
                          b[i] & 0x1f,
                        ((b[i+1] & 0x03) << 3) | ((b[i] & 0xe0) >> 5),
                         (b[i+1] & 0x7c) >> 2,
                        ((b[i+2] & 0x0f) << 1) | ((b[i+1] & 0x80) >> 7),
                        ((b[i+3] & 0x01) << 4) | ((b[i+2] & 0xf0) >> 4),
                         (b[i+3] & 0x3e) >> 1,
                        ((b[i+4]   & 0x07) << 2) | ((b[i+3] & 0xc0) >> 6),
                         (b[i+4]   & 0xf8) >> 3,
                        ])
                    i += 5
                elif bits_per_value == 6:
                    result.extend([
                          b[i] & 0x3f,
                        ((b[i+1] & 0x0f) << 2) | ((b[i]   & 0xc0) >> 6),
                        ((b[i+2] & 0x03) << 4) | ((b[i+1] & 0xf0) >> 4),
                         (b[i+2] & 0xfc) >> 2,
                         ])
                    i += 3
                elif bits_per_value == 7:
                    result.extend([
                          b[i] & 0x7f,
                        ((b[i+1] & 0x3f) << 1) | ((b[i]   & 0x80) >> 7),
                        ((b[i+2] & 0x1f) << 2) | ((b[i+1] & 0xc0) >> 6),
                        ((b[i+3] & 0x0f) << 3) | ((b[i+2] & 0xe0) >> 5),
                        ((b[i+4] & 0x07) << 4) | ((b[i+3] & 0xf0) >> 4),
                        ((b[i+5] & 0x03) << 5) | ((b[i+4] & 0xf8) >> 3),
                        ((b[i+6] & 0x01) << 6) | ((b[i+5] & 0xfc) >> 2),
                         (b[i+6] & 0xfc) >> 1,
                         ])
                    i += 7
                elif bits_per_value == 9:
                    result.extend([
                        ((b[i+1] & 0x01) << 8) | b[0],
                        ((b[i+2] & 0x03) << 7) | ((b[i+1] & 0xfe) >> 1),
                        ((b[i+3] & 0x07) << 6) | ((b[i+2] & 0xfc) >> 2),
                        ((b[i+4] & 0x0f) << 5) | ((b[i+3] & 0xf8) >> 3),
                        ((b[i+5] & 0x1f) << 4) | ((b[i+4] & 0xf0) >> 4),
                        ((b[i+6] & 0x3f) << 3) | ((b[i+5] & 0xe0) >> 5),
                        ((b[i+7] & 0x7f) << 2) | ((b[i+6] & 0xc0) >> 6),
                         (b[i+8] << 1) | ((b[i+7] & 0x80) >> 7),
                        ])
                    i += 9
                elif bits_per_value == 10:
                    result.extend([
                        ((b[i+1] & 0x03) << 8) |   b[0],
                        ((b[i+2] & 0x0f) << 6) | ((b[i+1] & 0xfc) >> 2),
                        ((b[i+3] & 0x3f) << 4) | ((b[i+2] & 0xf0) >> 4),
                         (b[i+4] << 2)         | ((b[i+3] & 0xc0) >> 6),
                        ])
                    i += 5
                elif bits_per_value == 11:
                    result.extend([
                        ((b[i+1] & 0x07) << 8) |   b[0],
                        ((b[i+2] & 0x3f) << 5) | ((b[i+1] & 0xf8) >> 3),
                        ((b[i+4] & 0x01) << 10)| (b[i+3] << 2) | ((b[i+2] & 0xc0) >> 6),
                        ((b[i+5] & 0x0f) << 7) | ((b[i+4] & 0xfe) >> 1),
                        ((b[i+6] & 0x7f) << 4) | ((b[i+5] & 0xf0) >> 4),
                        ((b[i+8] & 0x03) << 9) | (b[i+7] << 1) | ((b[i+6] & 0x80) >> 7),
                        ((b[i+9] & 0x1f) << 2) | ((b[i+8] & 0xfc) >> 2),
                         (b[i+10]        << 3) | ((b[i+9] & 0xe0) >> 5),
                       ])
                    i += 11
                elif bits_per_value == 12:
                    result.extend([
                        ((b[i+1] & 0x0f) << 8) |   b[0],
                         (b[i+2]         << 4) | ((b[i+1] & 0xf0) >> 4),
                        ])
                    i += 3
            result = numpy.asarray(result, numpy.uint16)
        return result

    def _get_blockdata_v113(self, section, unrecognized_block_types):
        # Translate each entry in the palette to a 1.2-era (block, data) int pair.
        num_palette_entries = len(section['Palette'])
        palette_translated = [] # (block, data) pairs, num_palette_entries in length
        palette_block_counts = [] # ints, num_palette_entries in length
        unrecognized_palette_entries = []
        for i in range(num_palette_entries):
            key = section['Palette'][i]
            palette_block_counts.append(0)
            try:
                palette_translated.append(self._get_block(key))
            except KeyError as e:
                # Unknown block type? Track it, treat it as air, and move on.
                unrecognized_palette_entries.append(i)
                palette_translated.append(self._blockmap['minecraft:air'])

        # Turn the BlockStates array into a 16x16x16 numpy matrix of shorts.
        blocks = numpy.zeros((4096,), dtype=numpy.uint16)
        data = numpy.zeros((4096,), dtype=numpy.uint8)
        block_states = self._packed_longarray_to_shorts(section['BlockStates'], 4096)
        for i in range(4096):
            palette_index = block_states[i]
            (blocks[i], data[i]) = palette_translated[palette_index]
            palette_block_counts[palette_index] += 1

        # Turn the Data array into a 16x16x16 matrix, same as SkyLight
        blocks  = blocks.reshape((16, 16, 16))
        data = data.reshape((16, 16, 16))

        for i in unrecognized_palette_entries:
            if palette_block_counts[i] > 0:
                palette_entry = section['Palette'][i]
                k = palette_entry['Name']
                if 'Properties' in palette_entry:
                    k += " %s" % str(palette_entry['Properties'])
                unrecognized_block_types[k] = unrecognized_block_types.get(k, 0) + palette_block_counts[i]

        return (blocks, data)

    def _get_blockdata_v112(self, section):
        # Turn the Data array into a 16x16x16 matrix, same as SkyLight
        data = numpy.frombuffer(section['Data'], dtype=numpy.uint8)
        data = data.reshape((16,16,8))
        data_expanded = numpy.empty((16,16,16), dtype=numpy.uint8)
        data_expanded[:,:,::2] = data & 0x0F
        data_expanded[:,:,1::2] = (data & 0xF0) >> 4

        # Turn the Blocks array into a 16x16x16 numpy matrix of shorts,
        # adding in the additional block array if included.
        blocks = numpy.frombuffer(section['Blocks'], dtype=numpy.uint8)
        # Cast up to uint16, blocks can have up to 12 bits of data
        blocks = blocks.astype(numpy.uint16)
        blocks = blocks.reshape((16,16,16))
        if "Add" in section:
            # This section has additional bits to tack on to the blocks
            # array. Add is a packed array with 4 bits per slot, so
            # it needs expanding
            additional = numpy.frombuffer(section['Add'], dtype=numpy.uint8)
            additional = additional.astype(numpy.uint16).reshape((16,16,8))
            additional_expanded = numpy.empty((16,16,16), dtype=numpy.uint16)
            additional_expanded[:,:,::2] = (additional & 0x0F) << 8
            additional_expanded[:,:,1::2] = (additional & 0xF0) << 4
            blocks += additional_expanded
            del additional
            del additional_expanded
            del section['Add'] # Save some memory

        return (blocks, data_expanded)

    #@log_other_exceptions
    def get_chunk(self, x, z):
        """Returns a dictionary object representing the "Level" NBT Compound
        structure for a chunk given its x, z coordinates. The coordinates given
        are chunk coordinates. Raises ChunkDoesntExist exception if the given
        chunk does not exist.

        The returned dictionary corresponds to the "Level" structure in the
        chunk file, with a few changes:

        * The Biomes array is transformed into a 16x16 numpy array

        * For each chunk section:

          * The "Blocks" byte string is transformed into a 16x16x16 numpy array
          * The Add array, if it exists, is bitshifted left 8 bits and
            added into the Blocks array
          * The "SkyLight" byte string is transformed into a 16x16x128 numpy
            array
          * The "BlockLight" byte string is transformed into a 16x16x128 numpy
            array
          * The "Data" byte string is transformed into a 16x16x128 numpy array

        Warning: the returned data may be cached and thus should not be
        modified, lest it affect the return values of future calls for the same
        chunk.
        """
        regionfile = self._get_region_path(x, z)
        if regionfile is None:
            raise ChunkDoesntExist("Chunk %s,%s doesn't exist (and neither does its region)" % (x,z))

        # Try a few times to load and parse this chunk before giving up and
        # raising an error
        tries = 5
        while True:
            try:
                region = self._get_regionobj(regionfile)
                data = region.load_chunk(x, z)
            except nbt.CorruptionError, e:
                tries -= 1
                if tries > 0:
                    # Flush the region cache to possibly read a new region file
                    # header
                    logging.debug("Encountered a corrupt chunk at %s,%s. Flushing cache and retrying", x, z)
                    #logging.debug("Error was:", exc_info=1)
                    del self.regioncache[regionfile]
                    time.sleep(0.5)
                    continue
                else:
                    logging.warning("The following was encountered while reading from %s:", self.regiondir)
                    if isinstance(e, nbt.CorruptRegionError):
                        logging.warning("Tried several times to read chunk %d,%d. Its region (%d,%d) may be corrupt. Giving up.",
                                x, z,x//32,z//32)
                    elif isinstance(e, nbt.CorruptChunkError):
                        logging.warning("Tried several times to read chunk %d,%d. It may be corrupt. Giving up.",
                                x, z)
                    else:
                        logging.warning("Tried several times to read chunk %d,%d. Unknown error. Giving up.",
                                x, z)
                    logging.debug("Full traceback:", exc_info=1)
                    # Let this exception propagate out through the C code into
                    # tileset.py, where it is caught and gracefully continues
                    # with the next chunk
                    raise
            else:
                # no exception raised: break out of the loop
                break

        if data is None:
            raise ChunkDoesntExist("Chunk %s,%s doesn't exist" % (x,z))

        level = data[1]['Level']
        chunk_data = level

        # From the interior of a map to the edge, a chunk's status may be one of:
        # - postprocessed (interior, or next to fullchunk)
        # - fullchunk (next to decorated)
        # - decorated (next to liquid_carved)
        # - liquid_carved (next to carved)
        # - carved (edge of world)
        # - empty
        # Empty is self-explanatory, and liquid_carved and carved seem to correspond
        # to SkyLight not being calculated, which results in mostly-black chunks,
        # so we'll just pretend they aren't there.
        if chunk_data.get("Status", "") in ("empty", "carved", "liquid_carved", "decorated"):
            raise ChunkDoesntExist("Chunk %s,%s doesn't exist" % (x,z))

        # Turn the Biomes array into a 16x16 numpy arra
        if 'Biomes' in chunk_data:
            biomes = chunk_data['Biomes']
            if isinstance(biomes, str):
                biomes = numpy.frombuffer(biomes, dtype=numpy.uint8)
            else:
                biomes = numpy.asarray(biomes)
            biomes = biomes.reshape((16,16))
        else:
            # worlds converted by Jeb's program may be missing the Biomes key
            biomes = numpy.zeros((16, 16), dtype=numpy.uint8)
        chunk_data['Biomes'] = biomes

        unrecognized_block_types = {}
        for section in chunk_data['Sections']:

            # Turn the skylight array into a 16x16x16 matrix. The array comes
            # packed 2 elements per byte, so we need to expand it.
            try:
                skylight = numpy.frombuffer(section['SkyLight'], dtype=numpy.uint8)
                skylight = skylight.reshape((16,16,8))
                skylight_expanded = numpy.empty((16,16,16), dtype=numpy.uint8)
                skylight_expanded[:,:,::2] = skylight & 0x0F
                skylight_expanded[:,:,1::2] = (skylight & 0xF0) >> 4
                del skylight
                section['SkyLight'] = skylight_expanded

                # Turn the BlockLight array into a 16x16x16 matrix, same as SkyLight
                blocklight = numpy.frombuffer(section['BlockLight'], dtype=numpy.uint8)
                blocklight = blocklight.reshape((16,16,8))
                blocklight_expanded = numpy.empty((16,16,16), dtype=numpy.uint8)
                blocklight_expanded[:,:,::2] = blocklight & 0x0F
                blocklight_expanded[:,:,1::2] = (blocklight & 0xF0) >> 4
                del blocklight
                section['BlockLight'] = blocklight_expanded

                if 'Palette' in section:
                    (blocks, data) = self._get_blockdata_v113(section, unrecognized_block_types)
                else:
                    (blocks, data) = self._get_blockdata_v112(section)
                (section['Blocks'], section['Data']) = (blocks, data)

            except ValueError:
                # iv'e seen at least 1 case where numpy raises a value error during the reshapes.  i'm not
                # sure what's going on here, but let's treat this as a corrupt chunk error
                logging.warning("There was a problem reading chunk %d,%d.  It might be corrupt.  I am giving up and will not render this particular chunk.", x, z)

                logging.debug("Full traceback:", exc_info=1)
                raise nbt.CorruptChunkError()

        for k in unrecognized_block_types:
            logging.debug("Found %d blocks of unknown type %s" % (unrecognized_block_types[k], k))

        return chunk_data      
    

    def iterate_chunks(self):
        """Returns an iterator over all chunk metadata in this world. Iterates
        over tuples of integers (x,z,mtime) for each chunk.  Other chunk data
        is not returned here.
        
        """

        for (regionx, regiony), (regionfile, filemtime) in self.regionfiles.iteritems():
            try:
                mcr = self._get_regionobj(regionfile)
            except nbt.CorruptRegionError:
                logging.warning("Found a corrupt region file at %s,%s in %s, Skipping it.", regionx, regiony, self.regiondir)
                continue
            for chunkx, chunky in mcr.get_chunks():
                yield chunkx+32*regionx, chunky+32*regiony, mcr.get_chunk_timestamp(chunkx, chunky)

    def iterate_newer_chunks(self, mtime):
        """Returns an iterator over all chunk metadata in this world. Iterates
        over tuples of integers (x,z,mtime) for each chunk.  Other chunk data
        is not returned here.
        
        """

        for (regionx, regiony), (regionfile, filemtime) in self.regionfiles.iteritems():
            """ SKIP LOADING A REGION WHICH HAS NOT BEEN MODIFIED! """
            if (filemtime < mtime):
                continue

            try:
                mcr = self._get_regionobj(regionfile)
            except nbt.CorruptRegionError:
                logging.warning("Found a corrupt region file at %s,%s in %s, Skipping it.", regionx, regiony, self.regiondir)
                continue

            for chunkx, chunky in mcr.get_chunks():
                yield chunkx+32*regionx, chunky+32*regiony, mcr.get_chunk_timestamp(chunkx, chunky)

    def get_chunk_mtime(self, x, z):
        """Returns a chunk's mtime, or False if the chunk does not exist.  This
        is therefore a dual purpose method. It corrects for the given north
        direction as described in the docs for get_chunk()
        
        """

        regionfile = self._get_region_path(x,z)
        if regionfile is None:
            return None
        try:
            data = self._get_regionobj(regionfile)
        except nbt.CorruptRegionError:
            logging.warning("Ignoring request for chunk %s,%s; region %s,%s seems to be corrupt",
                    x,z, x//32,z//32)
            return None
        if data.chunk_exists(x,z):
            return data.get_chunk_timestamp(x,z)
        return None

    def _get_region_path(self, chunkX, chunkY):
        """Returns the path to the region that contains chunk (chunkX, chunkY)
        Coords can be either be global chunk coords, or local to a region

        """
        (regionfile,filemtime) = self.regionfiles.get((chunkX//32, chunkY//32),(None, None))
        return regionfile
            
    def _iterate_regionfiles(self):
        """Returns an iterator of all of the region files, along with their 
        coordinates

        Returns (regionx, regiony, filename)"""

        logging.debug("regiondir is %s, has type %r", self.regiondir, self.type)

        for f in os.listdir(self.regiondir):
            if re.match(r"^r\.-?\d+\.-?\d+\.mca$", f):
                p = f.split(".")
                x = int(p[1])
                y = int(p[2])
                if abs(x) > 500000 or abs(y) > 500000:
                    logging.warning("Holy shit what is up with region file %s !?" % f)
                yield (x, y, os.path.join(self.regiondir, f))

class RegionSetWrapper(object):
    """This is the base class for all "wrappers" of RegionSet objects. A
    wrapper is an object that acts similarly to a subclass: some methods are
    overridden and functionality is changed, others may not be. The difference
    here is that these wrappers may wrap each other, forming chains.

    In fact, subclasses of this object may act exactly as if they've subclassed
    the original RegionSet object, except the first parameter of the
    constructor is a regionset object, not a regiondir.

    This class must implement the full public interface of RegionSet objects

    """
    def __init__(self, rsetobj):
        self._r = rsetobj

    def get_type(self):
        return self._r.get_type()
    def get_biome_data(self, x, z):
        return self._r.get_biome_data(x,z)
    def get_chunk(self, x, z):
        return self._r.get_chunk(x,z)
    def iterate_chunks(self):
        return self._r.iterate_chunks()
    def iterate_newer_chunks(self,filemtime):
        return self._r.iterate_newer_chunks(filemtime)
    def get_chunk_mtime(self, x, z):
        return self._r.get_chunk_mtime(x,z)
    
# see RegionSet.rotate.  These values are chosen so that they can be
# passed directly to rot90; this means that they're the number of
# times to rotate by 90 degrees CCW
UPPER_LEFT  = 0 ## - Return the world such that north is down the -Z axis (no rotation)
UPPER_RIGHT = 1 ## - Return the world such that north is down the +X axis (rotate 90 degrees counterclockwise)
LOWER_RIGHT = 2 ## - Return the world such that north is down the +Z axis (rotate 180 degrees)
LOWER_LEFT  = 3 ## - Return the world such that north is down the -X axis (rotate 90 degrees clockwise)

class RotatedRegionSet(RegionSetWrapper):
    """A regionset, only rotated such that north points in the given direction

    """
    
    # some class-level rotation constants
    _NO_ROTATION =               lambda x,z: (x,z)
    _ROTATE_CLOCKWISE =          lambda x,z: (-z,x)
    _ROTATE_COUNTERCLOCKWISE =   lambda x,z: (z,-x)
    _ROTATE_180 =                lambda x,z: (-x,-z)
    
    # These take rotated coords and translate into un-rotated coords
    _unrotation_funcs = [
        _NO_ROTATION,
        _ROTATE_COUNTERCLOCKWISE,
        _ROTATE_180,
        _ROTATE_CLOCKWISE,
    ]
    
    # These translate un-rotated coordinates into rotated coordinates
    _rotation_funcs = [
        _NO_ROTATION,
        _ROTATE_CLOCKWISE,
        _ROTATE_180,
        _ROTATE_COUNTERCLOCKWISE,
    ]
    
    def __init__(self, rsetobj, north_dir):
        self.north_dir = north_dir
        self.unrotate = self._unrotation_funcs[north_dir]
        self.rotate = self._rotation_funcs[north_dir]

        super(RotatedRegionSet, self).__init__(rsetobj)

    
    # Re-initialize upon unpickling. This is needed because we store a couple
    # lambda functions as instance variables
    def __getstate__(self):
        return (self._r, self.north_dir)
    def __setstate__(self, args):
        self.__init__(args[0], args[1])
    
    def get_chunk(self, x, z):
        x,z = self.unrotate(x,z)
        chunk_data = dict(super(RotatedRegionSet, self).get_chunk(x,z))
        newsections = []
        for section in chunk_data['Sections']:
            section = dict(section)
            newsections.append(section)
            for arrayname in ['Blocks', 'Data', 'SkyLight', 'BlockLight']:
                array = section[arrayname]
                # Since the anvil change, arrays are arranged with axes Y,Z,X
                # numpy.rot90 always rotates the first two axes, so for it to
                # work, we need to temporarily move the X axis to the 0th axis.
                array = numpy.swapaxes(array, 0,2)
                array = numpy.rot90(array, self.north_dir)
                array = numpy.swapaxes(array, 0,2)
                section[arrayname] = array
        chunk_data['Sections'] = newsections
        
        # same as above, for biomes (Z/X indexed)
        biomes = numpy.swapaxes(chunk_data['Biomes'], 0, 1)
        biomes = numpy.rot90(biomes, self.north_dir)
        chunk_data['Biomes'] = numpy.swapaxes(biomes, 0, 1)
        return chunk_data

    def get_chunk_mtime(self, x, z):
        x,z = self.unrotate(x,z)
        return super(RotatedRegionSet, self).get_chunk_mtime(x, z)

    def iterate_chunks(self):
        for x,z,mtime in super(RotatedRegionSet, self).iterate_chunks():
            x,z = self.rotate(x,z)
            yield x,z,mtime

    def iterate_newer_chunks(self, filemtime):
        for x,z,mtime in super(RotatedRegionSet, self).iterate_newer_chunks(filemtime):
            x,z = self.rotate(x,z)
            yield x,z,mtime

class CroppedRegionSet(RegionSetWrapper):
    def __init__(self, rsetobj, xmin, zmin, xmax, zmax):
        super(CroppedRegionSet, self).__init__(rsetobj)
        self.xmin = xmin//16
        self.xmax = xmax//16
        self.zmin = zmin//16
        self.zmax = zmax//16

    def get_chunk(self,x,z):
        if (
                self.xmin <= x <= self.xmax and
                self.zmin <= z <= self.zmax
                ):
            return super(CroppedRegionSet, self).get_chunk(x,z)
        else:
            raise ChunkDoesntExist("This chunk is out of the requested bounds")

    def iterate_chunks(self):
        return ((x,z,mtime) for (x,z,mtime) in super(CroppedRegionSet,self).iterate_chunks()
                if
                    self.xmin <= x <= self.xmax and
                    self.zmin <= z <= self.zmax
                )

    def iterate_newer_chunks(self, filemtime):
        return ((x,z,mtime) for (x,z,mtime) in super(CroppedRegionSet,self).iterate_newer_chunks(filemtime)
                if
                    self.xmin <= x <= self.xmax and
                    self.zmin <= z <= self.zmax
                )

    def get_chunk_mtime(self,x,z):
        if (
                self.xmin <= x <= self.xmax and
                self.zmin <= z <= self.zmax
                ):
            return super(CroppedRegionSet, self).get_chunk_mtime(x,z)
        else:
            return None

class CachedRegionSet(RegionSetWrapper):
    """A regionset wrapper that implements caching of the results from
    get_chunk()

    """
    def __init__(self, rsetobj, cacheobjects):
        """Initialize this wrapper around the given regionset object and with
        the given list of cache objects. The cache objects may be shared among
        other CachedRegionSet objects.

        """
        super(CachedRegionSet, self).__init__(rsetobj)
        self.caches = cacheobjects

        # Construct a key from the sequence of transformations and the real
        # RegionSet object, so that items we place in the cache don't conflict
        # with other worlds/transformation combinations.
        obj = self._r
        s = ""
        while isinstance(obj, RegionSetWrapper):
            s += obj.__class__.__name__ + "."
            obj = obj._r
        # obj should now be the actual RegionSet object
        try:
            s += obj.regiondir
        except AttributeError:
            s += repr(obj)

        logging.debug("Initializing a cache with key '%s'", s)

        self.key = s

    def get_chunk(self, x, z):
        key = (self.key, x, z)
        for i, cache in enumerate(self.caches):
            try:
                retval = cache[key]
                # This did have it, no need to re-add it to this cache, just
                # the ones before it
                i -= 1
                break
            except KeyError:
                pass
        else:
            retval = super(CachedRegionSet, self).get_chunk(x,z)

        # Now add retval to all the caches that didn't have it, all the caches
        # up to and including index i
        for cache in self.caches[:i+1]:
            cache[key] = retval

        return retval
        

def get_save_dir():
    """Returns the path to the local saves directory
      * On Windows, at %APPDATA%/.minecraft/saves/
      * On Darwin, at $HOME/Library/Application Support/minecraft/saves/
      * at $HOME/.minecraft/saves/

    """
    
    savepaths = []
    if "APPDATA" in os.environ:
        savepaths += [os.path.join(os.environ['APPDATA'], ".minecraft", "saves")]
    if "HOME" in os.environ:
        savepaths += [os.path.join(os.environ['HOME'], "Library",
                "Application Support", "minecraft", "saves")]
        savepaths += [os.path.join(os.environ['HOME'], ".minecraft", "saves")]

    for path in savepaths:
        if os.path.exists(path):
            return path

def get_worlds():
    "Returns {world # or name : level.dat information}"
    ret = {}
    save_dir = get_save_dir()
    loc = locale.getpreferredencoding()

    # No dirs found - most likely not running from inside minecraft-dir
    if not save_dir is None:
        for dir in os.listdir(save_dir):
            world_path = os.path.join(save_dir, dir)
            world_dat = os.path.join(world_path, "level.dat")
            if not os.path.exists(world_dat): continue
            try:
                info = nbt.load(world_dat)[1]
                info['Data']['path'] = os.path.join(save_dir, dir).decode(loc)
                if 'LevelName' in info['Data'].keys():
                    ret[info['Data']['LevelName']] = info['Data']
            except nbt.CorruptNBTError:
                ret[os.path.basename(world_path).decode(loc) + " (corrupt)"] = {'path': world_path.decode(loc),
                        'LastPlayed': 0,
                        'Time': 0,
                        'IsCorrupt': True}

    
    for dir in os.listdir("."):
        world_dat = os.path.join(dir, "level.dat")
        if not os.path.exists(world_dat): continue
        world_path = os.path.join(".", dir)
        try:
            info = nbt.load(world_dat)[1]
            info['Data']['path'] = world_path.decode(loc)
            if 'LevelName' in info['Data'].keys():
                ret[info['Data']['LevelName']] = info['Data']
        except nbt.CorruptNBTError:
            ret[os.path.basename(world_path).decode(loc) + " (corrupt)"] = {'path': world_path.decode(loc),
                    'LastPlayed': 0,
                    'Time': 0,
                    'IsCorrupt': True}

    return ret
