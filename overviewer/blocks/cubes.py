"""
This module defines all simple cube blocks
"""

from collections import namedtuple

from overviewer import chunkrenderer
from overviewer.blockdefinitions import BlockModel, BlockDefinition

CubeTextures = namedtuple("CubeTextures", ["nx","nz","px","pz","ny","py"])

color_map = ["white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray",
        "silver", "cyan", "purple", "blue", "brown", "green", "red", "black"]

# the Positive-Z face faces south
# the Negative-Z faces north (and is normally hidden from view)
# the Positive-X faces east (and is normally hidden from view)
# the Negative-X faces west 


# Height must be between 0 and 1.0
def make_box(tex, nx=(0, 0, 1, 1), px=(0, 0, 1, 1), ny=(0, 0, 1, 1), py=(0, 0, 1, 1), nz=(0, 0, 1, 1), pz=(0, 0, 1, 1), color=(255, 255, 255, 255), topcolor=None, bottom=0, height=1):
    if topcolor is None:
        topcolor = color
    xcolor = tuple(int(c * 0.8) for c in color[:3]) + (color[3],)
    zcolor = tuple(int(c * 0.9) for c in color[:3]) + (color[3],)

    if not isinstance(tex, CubeTextures):
        if isinstance(tex, tuple):
            tex = CubeTextures._make(tex)
        else:
            tex = CubeTextures._make((tex,)*6)

    model = BlockModel()
    model.vertices = [
        # NX face
        ((0, bottom, 0), (nx[0], nx[1]), xcolor),
        ((0, bottom, 1), (nx[2], nx[1]), xcolor),
        ((0, height, 1), (nx[2], nx[3]), xcolor),
        ((0, height, 0), (nx[0], nx[3]), xcolor),
        
        # NZ face
        ((1, bottom, 0), (nz[0], nz[1]), zcolor),
        ((0, bottom, 0), (nz[2], nz[1]), zcolor),
        ((0, height, 0), (nz[2], nz[3]), zcolor),
        ((1, height, 0), (nz[0], nz[3]), zcolor),
        
        # PX face
        ((1, bottom, 1), (px[0], px[1]), xcolor),
        ((1, bottom, 0), (px[2], px[1]), xcolor),
        ((1, height, 0), (px[2], px[3]), xcolor),
        ((1, height, 1), (px[0], px[3]), xcolor),
        
        # PZ face
        ((0, bottom, 1), (pz[0], pz[1]), zcolor),
        ((1, bottom, 1), (pz[2], pz[1]), zcolor),
        ((1, height, 1), (pz[2], pz[3]), zcolor),
        ((0, height, 1), (pz[0], pz[3]), zcolor),
        
        # NY face
        ((0, bottom, 1), (ny[0], ny[1]), color),
        ((0, bottom, 0), (ny[2], ny[1]), color),
        ((1, bottom, 0), (ny[2], ny[3]), color),
        ((1, bottom, 1), (ny[0], ny[3]), color),
        
        # PY face
        ((0, height, 1), (py[0], py[1]), topcolor),
        ((1, height, 1), (py[2], py[1]), topcolor),
        ((1, height, 0), (py[2], py[3]), topcolor),
        ((0, height, 0), (py[0], py[3]), topcolor),
    ]
    model.faces = [
        ([0, 1, 2, 3], chunkrenderer.FACE_TYPE_NX, tex.nx),
        ([4, 5, 6, 7], chunkrenderer.FACE_TYPE_NZ, tex.nz),
        ([8, 9, 10, 11], chunkrenderer.FACE_TYPE_PX, tex.px),
        ([12, 13, 14, 15], chunkrenderer.FACE_TYPE_PZ, tex.pz),
        ([16, 17, 18, 19], chunkrenderer.FACE_TYPE_NY, tex.ny),
        ([20, 21, 22, 23], chunkrenderer.FACE_TYPE_PY, tex.py),
    ]
    return model

def make_simple(terrain, **kwargs):
    t = (0, 0, 1, 1)
    return make_box(terrain, nx=t, px=t, ny=t, py=t, nz=t, pz=t, **kwargs)

def make_custom(terrain, **kwargs):
    return make_box(terrain, **kwargs)

def add(bd):

    # stone
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/stone.png")), 1)
    
    # grass
    side = "assets/minecraft/textures/blocks/grass_side.png"
    top = "assets/minecraft/textures/blocks/grass_top.png"
    bottom = "assets/minecraft/textures/blocks/dirt.png"
    bd.add(BlockDefinition(make_simple((side,side,side,side,bottom,top), topcolor=(0,255,0,255))), 2)
    
    # dirt
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/dirt.png")), 3)
    # cobblestone
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/cobblestone.png")), 4)
    
    # wood planks
    wood_planks = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    wood_planks.add(make_simple("assets/minecraft/textures/blocks/planks_oak.png"), 0)
    wood_planks.add(make_simple("assets/minecraft/textures/blocks/planks_spruce.png"), 1)
    wood_planks.add(make_simple("assets/minecraft/textures/blocks/planks_birch.png"), 2)
    wood_planks.add(make_simple("assets/minecraft/textures/blocks/planks_jungle.png"), 3)
    bd.add(wood_planks, 5)

    # bedrock
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/bedrock.png")), 7)
    # sand
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/sand.png")), 12)
    # gravel
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/gravel.png")), 13)
    # gold ore
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/gold_ore.png")), 14)
    # iron ore
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/iron_ore.png")), 15)
    # coal
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/coal_ore.png")), 16)

    # logs
    side = "assets/minecraft/textures/blocks/log_{type}.png"
    top = "assets/minecraft/textures/blocks/log_{type}_top.png"
    logs = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    def make_log(logtype, direction):
        if direction == 0:
            # up/down
            tex = CubeTextures(ny=top,py=top,nx=side,px=side,nz=side,pz=side)
        elif direction == 1:
            # east/west
            tex = CubeTextures(ny=side,py=side,nx=top,px=top,nz=side,pz=side)
        elif direction == 2:
            # north/south
            tex = CubeTextures(ny=side,py=side,nx=side,px=side,nz=top,pz=top)
        elif direction == 3:
            # all bark
            tex = CubeTextures(ny=side,py=side,nx=side,px=side,nz=side,pz=side)
        tex = tuple(x.format(type=logtype) for x in tex)
        return make_simple(tex)
    logs.add(make_log("oak", 0),    0)
    logs.add(make_log("spruce", 0), 1)
    logs.add(make_log("birch", 0),  2)
    logs.add(make_log("jungle", 0), 3)
    logs.add(make_log("oak", 1),    4)
    logs.add(make_log("spruce", 1), 5)
    logs.add(make_log("birch", 1),  6)
    logs.add(make_log("jungle", 1), 7)
    logs.add(make_log("oak", 2),    8)
    logs.add(make_log("spruce", 2), 9)
    logs.add(make_log("birch", 2),  10)
    logs.add(make_log("jungle", 2), 11)
    logs.add(make_log("oak", 3),    12)
    logs.add(make_log("spruce", 3), 13)
    logs.add(make_log("birch", 3),  14)
    logs.add(make_log("jungle", 3), 15)
    bd.add(logs, 17)
    # 1.7 added 2 more log types under block id 162
    logs2 = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    logs2.add(make_log("acacia", 0),    0)
    logs2.add(make_log("big_oak", 0),   1)

    logs2.add(make_log("acacia", 1),    4)
    logs2.add(make_log("big_oak", 1),   5)

    logs2.add(make_log("acacia", 2),    8)
    logs2.add(make_log("big_oak", 2),   9)

    logs2.add(make_log("acacia", 3),    12)
    logs2.add(make_log("big_oak", 3),   13)
    bd.add(logs2, 162)
    
    # leaves
    leaves = BlockDefinition(transparent=True, datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    leaves.add(make_simple("assets/minecraft/textures/blocks/leaves_oak.png", color=(0, 150, 0, 255)), 0)
    leaves.add(make_simple("assets/minecraft/textures/blocks/leaves_spruce.png", color=(0, 150, 0, 255)), 1)
    leaves.add(make_simple("assets/minecraft/textures/blocks/leaves_birch.png", color=(0, 150, 0, 255)), 2)
    leaves.add(make_simple("assets/minecraft/textures/blocks/leaves_jungle.png", color=(0, 150, 0, 255)), 3)
    bd.add(leaves, 18)

    # sponge
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/sponge.png")), 19)
    
    # glass
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/glass.png"), transparent=True), 20)

    # lapis core
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/lapis_ore.png")), 21)

    # lapis block
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/lapis_block.png")), 22)

    # dispenser - 23
    # data values encode oritentation
    dispenser = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    def make_dispenser(data, jack=False):
        top = "assets/minecraft/textures/blocks/furnace_top.png"
        side = "assets/minecraft/textures/blocks/furnace_side.png"
        front = "assets/minecraft/textures/blocks/dispenser_front_horizontal.png"
        vert = "assets/minecraft/textures/blocks/dispenser_front_vertical.png"
        if data & 0b111 == 0: # pointing down:
            tex = CubeTextures(ny=vert, py=top, nx=side, px=side, nz=side, pz=side)
        elif data & 0b111 == 1: # pointing up:
            tex = CubeTextures(ny=top, py=vert, nx=side, px=side, nz=side, pz=side)
        elif data & 0b111 == 3: # pointing south
            tex = CubeTextures(ny=side, py=top, nx=side, px=side, nz=side, pz=front)
        elif data & 0b111 == 4: # pointing west
            tex = CubeTextures(ny=side, py=top, nx=front, px=side, nz=side, pz=side)
        elif data & 0b111 == 2: # pointing north
            tex = CubeTextures(ny=side, py=top, nx=side, px=side, nz=front, pz=side)
        elif data & 0b111 == 5: # pointing east
            tex = CubeTextures(ny=side, py=top, nx=side, px=front, nz=side, pz=side)

        return make_simple(tex)
    # note: i can't tell if bit 4 is a flagbit or not, since the masking in make_dispenser
    for x in range(6):
        dispenser.add(make_dispenser(x), x)
    bd.add(dispenser, 23)

    # sandstone - 24
    sandstone = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    def make_sandstone(data):
        top = "assets/minecraft/textures/blocks/sandstone_top.png"
        side = "assets/minecraft/textures/blocks/sandstone_{type}.png"
        bottom = "assets/minecraft/textures/blocks/sandstone_bottom.png"
        tex = CubeTextures(ny=bottom, py=top, nx=side, px=side, nz=side, pz=side)
        typemap = ["normal", "carved", "smooth"]
        tex = tuple(x.format(type=typemap[data]) for x in tex)
        return make_simple(tex)
    sandstone.add(make_sandstone(0), 0) # normal
    sandstone.add(make_sandstone(1), 1) # carved/chiseled
    sandstone.add(make_sandstone(2), 2) # smooth
    bd.add(sandstone, 24)

    #  noteblock - 25
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/noteblock.png")), 25)

    # Wool - 35
    wool = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    for color in range(16):
        wool.add(make_simple("assets/minecraft/textures/blocks/wool_colored_%s.png" % color_map[color]), color)

    bd.add(wool, 35)

    # Gold Block - 41
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/gold_block.png")), 41)

    # Iron Block - 42
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/iron_block.png")), 42)
    
    # Double stone slab - 43
    double_slab = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    def make_double_slab(data):
        if data == 0:
            top = "assets/minecraft/textures/blocks/stone_slab_top.png"
            side = "assets/minecraft/textures/blocks/stone_slab_side.png"
        elif data == 1:
            top = "assets/minecraft/textures/blocks/sandstone_top.png"
            side = "assets/minecraft/textures/blocks/sandstone_normal.png"
        elif data == 2:
            top = "assets/minecraft/textures/blocks/planks_oak.png"
            side = "assets/minecraft/textures/blocks/planks_oak.png"
        elif data == 3:
            top = "assets/minecraft/textures/blocks/cobblestone.png"
            side = "assets/minecraft/textures/blocks/cobblestone.png"
        elif data == 4:
            top = "assets/minecraft/textures/blocks/brick.png"
            side = "assets/minecraft/textures/blocks/brick.png"
        elif data == 5:
            top = "assets/minecraft/textures/blocks/stonebrick.png"
            side = "assets/minecraft/textures/blocks/stonebrick.png"
        elif data == 6:
            top = "assets/minecraft/textures/blocks/nether_brick.png"
            side = "assets/minecraft/textures/blocks/nether_brick.png"
        elif data == 7:
            top = "assets/minecraft/textures/blocks/quartz_block_top.png"
            side = "assets/minecraft/textures/blocks/quartz_block_side.png"
        elif data == 8:
            top = "assets/minecraft/textures/blocks/stone.png"
            side = "assets/minecraft/textures/blocks/stone.png"
        elif data == 9:
            top = "assets/minecraft/textures/blocks/sandstone_top.png"
            side = "assets/minecraft/textures/blocks/sandstone_smooth.png"
        tex = CubeTextures(ny=top, py=top, nx=side, px=side, nz=side, pz=side)
        return make_simple(tex)
    for x in range(10):
        double_slab.add(make_double_slab(x), x)
    bd.add(double_slab, 43)


    # signle stone slab - 44
    stone_slab = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH, transparent=True)
    def make_stone_slab(data):
        is_top = (data & 0x8) == 0x8
        texdata = (data & 0b111)
        if texdata == 0:
            top = "assets/minecraft/textures/blocks/stone_slab_top.png"
            side = "assets/minecraft/textures/blocks/stone_slab_side.png"
        elif texdata == 1:
            top = "assets/minecraft/textures/blocks/sandstone_top.png"
            side = "assets/minecraft/textures/blocks/sandstone_normal.png"
        elif texdata == 2:
            top = "assets/minecraft/textures/blocks/planks_oak.png"
            side = "assets/minecraft/textures/blocks/planks_oak.png"
        elif texdata == 3:
            top = "assets/minecraft/textures/blocks/cobblestone.png"
            side = "assets/minecraft/textures/blocks/cobblestone.png"
        elif texdata == 4:
            top = "assets/minecraft/textures/blocks/brick.png"
            side = "assets/minecraft/textures/blocks/brick.png"
        elif texdata == 5:
            top = "assets/minecraft/textures/blocks/stonebrick.png"
            side = "assets/minecraft/textures/blocks/stonebrick.png"
        elif texdata == 6:
            top = "assets/minecraft/textures/blocks/nether_brick.png"
            side = "assets/minecraft/textures/blocks/nether_brick.png"
        elif texdata == 7:
            top = "assets/minecraft/textures/blocks/quartz_block_top.png"
            side = "assets/minecraft/textures/blocks/quartz_block_side.png"
        tex = CubeTextures(ny=top, py=top, nx=side, px=side, nz=side, pz=side)
        if not is_top:
            t = (0, 0, 1, 0.5)
            return make_custom(tex, nx=t, px=t, nz=t, pz=t, bottom=0, height=0.5)
        else:
            t = (0, 0.5, 1, 1)
            return make_custom(tex, nx=t, px=t, nz=t, pz=t, bottom=0.5, height=1)
    for x in range(8):
        stone_slab.add(make_stone_slab(x | 0x8), x | 0x8)
        stone_slab.add(make_stone_slab(x), x)
    bd.add(stone_slab, 44)

    # Bricks - 45
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/brick.png")), 45)

    # TNT - 46
    top = "assets/minecraft/textures/blocks/tnt_top.png"
    bottom = "assets/minecraft/textures/blocks/tnt_bottom.png"
    side = "assets/minecraft/textures/blocks/tnt_side.png"
    tex = CubeTextures(ny=bottom, py=top, nx=side, px=side, nz=side, pz=side)
    bd.add(BlockDefinition(make_simple(tex)), 46)

    # Bookshel - 47
    top = "assets/minecraft/textures/blocks/planks_oak.png"
    side = "assets/minecraft/textures/blocks/bookshelf.png"
    tex = CubeTextures(ny=top, py=top, nx=side, px=side, nz=side, pz=side)
    bd.add(BlockDefinition(make_simple(tex)), 47)

    # mossy cobblestone - 48
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/cobblestone_mossy.png")), 48)

    # obsidian - 49
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/obsidian.png")), 49)

    # mob spawner - 52
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/mob_spawner.png")), 52)

    # diamond ore - 56
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/diamond_ore.png")), 56)
    
    # diamond block - 57
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/diamond_block.png")), 57)

    # crafting table - 58
    # the front texture is fixed -- does not depend on block oritentation
    side = "assets/minecraft/textures/blocks/crafting_table_side.png"
    front = "assets/minecraft/textures/blocks/crafting_table_front.png"
    top = "assets/minecraft/textures/blocks/crafting_table_top.png"
    tex = CubeTextures(ny=top, py=top, nx=front, px=front, nz=side, pz=side)
    bd.add(BlockDefinition(make_simple(tex)), 58)

    # farmland - 60
    side = "assets/minecraft/textures/blocks/dirt.png"
    dry = "assets/minecraft/textures/blocks/farmland_dry.png"
    wet = "assets/minecraft/textures/blocks/farmland_wet.png"
    farmland = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    drytex = CubeTextures(ny=side, py=dry, nx=side, px=side, nz=side, pz=side)
    wettex = CubeTextures(ny=side, py=wet, nx=side, px=side, nz=side, pz=side)
    # Im not sure what's up with all these data values.  the wiki doesn't any anything about this, i just copied from masterbranch
    for x in range(9):
        if x == 0:
            farmland.add(make_simple(drytex), x)
        else:
            farmland.add(make_simple(wettex), x)
    bd.add(farmland, 60)

    # furnace - 61
    # lit furnace - 62
    def make_furnace(data, lit=False):
        top = "assets/minecraft/textures/blocks/furnace_top.png"
        side = "assets/minecraft/textures/blocks/furnace_side.png"
        if lit:
            front = "assets/minecraft/textures/blocks/furnace_front_on.png"
        else:
            front = "assets/minecraft/textures/blocks/furnace_front_off.png"
        if data == 3: # pointing south
            tex = CubeTextures(ny=top, py=top, nx=side, px=side, nz=side, pz=front)
        elif data == 4: # pointing west
            tex = CubeTextures(ny=top, py=top, nx=front, px=side, nz=side, pz=side)
        elif data == 2: # pointing north
            tex = CubeTextures(ny=top, py=top, nx=side, px=side, nz=front, pz=side)
        elif data == 5: # pointing east
            tex = CubeTextures(ny=top, py=top, nx=side, px=front, nz=side, pz=side)
        return make_simple(tex)
    furnace = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    lit_furnace = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    for x in range(2,6):
        furnace.add(make_furnace(x, False), x)
        lit_furnace.add(make_furnace(x, True), x)
    bd.add(furnace, 61)
    bd.add(lit_furnace, 62)

    # stone pressure plate - 70
    t = (0, 0, 1, 1.0/16)
    bd.add(BlockDefinition(make_custom("assets/minecraft/textures/blocks/stone.png", nx=t, px=t, nz=t, pz=t, height=(1.0/16)), transparent=True), 70)
    
    # wooden pressure plate - 72
    t = (0, 0, 1, 1.0/16)
    bd.add(BlockDefinition(make_custom("assets/minecraft/textures/blocks/planks_oak.png", nx=t, px=t, nz=t, pz=t, height=(1.0/16)), transparent=True), 72)

    # redstone ore  - 73
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/redstone_ore.png")), 73)
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/redstone_ore.png")), 74) # glowing

    # ice - 79
    # TODO do face culling like we do for water?
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/ice.png"), transparent=True), 79)

    # snow - 80
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/snow.png")), 80)

    # clay - 82
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/clay.png")), 82)

    # jukebox - 84
    side = "assets/minecraft/textures/blocks/jukebox_side.png"
    top = "assets/minecraft/textures/blocks/jukebox_top.png"
    tex = CubeTextures(ny=side, py=top, nx=side, px=side, nz=side, pz=side)
    bd.add(BlockDefinition(make_simple(tex)), 84)

    # pumpkins (86) and jackolantern (91)
    # data values encode oritentation
    pumpkin = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    jacko = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    def make_pumpkin(data, jack=False):
        top = "assets/minecraft/textures/blocks/pumpkin_top.png"
        side = "assets/minecraft/textures/blocks/pumpkin_side.png"
        if jack:
            front = "assets/minecraft/textures/blocks/pumpkin_face_on.png"
        else:
            front = "assets/minecraft/textures/blocks/pumpkin_face_off.png"
        if data == 0: # pointing south
            tex = CubeTextures(ny=top, py=top, nx=side, px=side, nz=side, pz=front)
        elif data == 1: # pointing west
            tex = CubeTextures(ny=top, py=top, nx=front, px=side, nz=side, pz=side)
        elif data == 2: # pointing north
            tex = CubeTextures(ny=top, py=top, nx=side, px=side, nz=front, pz=side)
        elif data == 3: # pointing east
            tex = CubeTextures(ny=top, py=top, nx=side, px=front, nz=side, pz=side)

        return make_simple(tex)
    for x in range(4):
        pumpkin.add(make_pumpkin(x), x)
        jacko.add(make_pumpkin(x, True), x)
    bd.add(pumpkin, 86)
    bd.add(jacko, 91)



    # netherrack - 87
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/netherrack.png")), 87)
    
    # soul sand - 88
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/soul_sand.png")), 88)
    
    # glowstone - 89
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/glowstone.png")), 89)

    # stained glass - 95
    # TODO face culling?
    stained_glass = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH, transparent=True)
    for color in range(16):
        stained_glass.add(make_simple("assets/minecraft/textures/blocks/glass_%s.png" % color_map[color]), color)

    bd.add(stained_glass, 95)


    # monster egg - 97
    monster_egg = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    def make_egg(data):
        typemap = ["stone", "cobblestone", "stonebrick", "stonebrick_mossy", "stonebrick_cracked", "stonebrick_carved"]
        return make_simple("assets/minecraft/textures/blocks/%s.png" % typemap[data])
    for x in range(6):
        monster_egg.add(make_egg(x), x)
    bd.add(monster_egg, 97)
    
    # stone brick - 98
    stone_brick = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    def make_brick(data):
        typemap = ["stonebrick", "stonebrick_mossy", "stonebrick_cracked", "stonebrick_carved"]
        return make_simple("assets/minecraft/textures/blocks/%s.png" % typemap[data])
    for x in range(4):
        stone_brick.add(make_brick(x), x)
    bd.add(stone_brick, 98)

    # melon block - 103
    top = "assets/minecraft/textures/blocks/melon_top.png"
    side = "assets/minecraft/textures/blocks/melon_side.png"
    tex = CubeTextures(ny=top, py=top, nx=side, px=side, nz=side, pz=side)
    bd.add(BlockDefinition(make_simple(tex)), 103)

    # mycelium - 110
    top = "assets/minecraft/textures/blocks/mycelium_top.png"
    side = "assets/minecraft/textures/blocks/mycelium_side.png"
    tex = CubeTextures(ny=top, py=top, nx=side, px=side, nz=side, pz=side)
    bd.add(BlockDefinition(make_simple(tex)), 110)

    # nether brick - 112
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/nether_brick.png")), 112)

    # end stone - 121
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/end_stone.png")), 121)

    # redstone lamp (off) - 123
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/redstone_lamp_off.png")), 123)
    # redstoen lamp (on) - 124
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/redstone_lamp_on.png")), 124)

    # double wooden slabs - 125
    double_wooden_slab = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    def make_double_wooden_slab(data):
        if data == 0:
            side = "assets/minecraft/textures/blocks/planks_oak.png"
        elif data == 1:
            side = "assets/minecraft/textures/blocks/planks_spruce.png"
        elif data == 2:
            side = "assets/minecraft/textures/blocks/planks_birch.png"
        elif data == 3:
            side = "assets/minecraft/textures/blocks/planks_jungle.png"
        elif data == 4:
            side = "assets/minecraft/textures/blocks/planks_acacia.png"
        elif data == 5:
            side = "assets/minecraft/textures/blocks/planks_big_oak.png"
        tex = CubeTextures(ny=side, py=side, nx=side, px=side, nz=side, pz=side)
        return make_simple(tex)
    for x in range(6):
        double_wooden_slab.add(make_double_wooden_slab(x), x)
    bd.add(double_wooden_slab, 125)

    # wooden slabs (half) - 126
    wooden_slab = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH, transparent=True)
    def make_wooden_slab(data):
        is_top = (data & 0x8) == 0x8
        texdata = (data & 0b111)
        if texdata == 0:
            side = "assets/minecraft/textures/blocks/planks_oak.png"
        elif texdata == 1:
            side = "assets/minecraft/textures/blocks/planks_spruce.png"
        elif texdata == 2:
            side = "assets/minecraft/textures/blocks/planks_birch.png"
        elif texdata == 3:
            side = "assets/minecraft/textures/blocks/planks_jungle.png"
        elif texdata == 4:
            side = "assets/minecraft/textures/blocks/planks_acacia.png"
        elif texdata == 5:
            side = "assets/minecraft/textures/blocks/planks_big_oak.png"
        if not is_top:
            t = (0, 0, 1, 0.5)
            return make_custom(side, nx=t, px=t, nz=t, pz=t, bottom=0, height=0.5)
        else:
            t = (0, 0.5, 1, 1)
            return make_custom(side, nx=t, px=t, nz=t, pz=t, bottom=0.5, height=1)
    for x in range(6):
        wooden_slab.add(make_wooden_slab(x | 0x8), x | 0x8)
        wooden_slab.add(make_wooden_slab(x), x)
    bd.add(wooden_slab, 126)

    # emerald ore - 129
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/emerald_ore.png")), 129)

    # emerald block - 133
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/emerald_block.png")), 133)
    
    # light weighted pressure plate - 147
    t = (0, 0, 1, 1.0/16)
    bd.add(BlockDefinition(make_custom("assets/minecraft/textures/blocks/gold_block.png", nx=t, px=t, nz=t, pz=t, height=(1.0/16)), transparent=True), 147)
    
    # heavy weighted pressure plate - 148
    t = (0, 0, 1, 1.0/16)
    bd.add(BlockDefinition(make_custom("assets/minecraft/textures/blocks/iron_block.png", nx=t, px=t, nz=t, pz=t, height=(1.0/16)), transparent=True), 148)
    
    # daylight sensor - 151
    top = "assets/minecraft/textures/blocks/daylight_detector_top.png"
    side = "assets/minecraft/textures/blocks/daylight_detector_side.png"
    tex = CubeTextures(ny=side, py=top, nx=side, px=side, nz=side, pz=side)
    t = (0, 0, 1, 1.0/3)
    bd.add(BlockDefinition(make_custom(tex, nx=t, px=t, nz=t, pz=t, height=1.0/3), transparent=True), 151)

    # redstone block - 152
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/redstone_block.png")), 152)
    
    # quartz ore - 153
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/quartz_ore.png")), 153)
    
    # dropper - 158
    # data values encode oritentation
    dropper = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    def make_dropper(data, jack=False):
        top = "assets/minecraft/textures/blocks/furnace_top.png"
        side = "assets/minecraft/textures/blocks/furnace_side.png"
        front = "assets/minecraft/textures/blocks/dropper_front_horizontal.png"
        vert = "assets/minecraft/textures/blocks/dropper_front_vertical.png"
        if data & 0b111 == 0: # pointing down:
            tex = CubeTextures(ny=vert, py=top, nx=side, px=side, nz=side, pz=side)
        elif data & 0b111 == 1: # pointing up:
            tex = CubeTextures(ny=top, py=vert, nx=side, px=side, nz=side, pz=side)
        elif data & 0b111 == 3: # pointing south
            tex = CubeTextures(ny=side, py=top, nx=side, px=side, nz=side, pz=front)
        elif data & 0b111 == 4: # pointing west
            tex = CubeTextures(ny=side, py=top, nx=front, px=side, nz=side, pz=side)
        elif data & 0b111 == 2: # pointing north
            tex = CubeTextures(ny=side, py=top, nx=side, px=side, nz=front, pz=side)
        elif data & 0b111 == 5: # pointing east
            tex = CubeTextures(ny=side, py=top, nx=side, px=front, nz=side, pz=side)

        return make_simple(tex)
    # note: i can't tell if bit 4 is a flagbit or not, since the masking in make_dropper
    for x in range(6):
        dropper.add(make_dropper(x), x)
    bd.add(dropper, 158)
    
    # stained hardened clay - 159
    stained_clay = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    for color in range(16):
        stained_clay.add(make_simple("assets/minecraft/textures/blocks/hardened_clay_stained_%s.png" % color_map[color]), color)
    bd.add(stained_clay, 159)
    
    # carpet - 171
    carpet = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH, transparent=True)
    for color in range(16):
        carpet.add(make_simple("assets/minecraft/textures/blocks/wool_colored_%s.png" % color_map[color], height=1.0/16), color)
    bd.add(carpet, 171)
    
    # hardened clay (uncolored) - 172
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/hardened_clay.png")), 172)
    
    # coal block- 173
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/coal_block.png")), 173)

    

