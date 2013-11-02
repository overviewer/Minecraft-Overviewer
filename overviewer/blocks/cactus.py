from overviewer.blockdefinitions import BlockModel, BlockDefinition
from overviewer import chunkrenderer

def add(bd):

    # a cactus is almost a cube, but with each side face moved in one pixel.
    # This code is mostly copied and pasted from the cube code but with the
    # vertices adjusted
    side = "assets/minecraft/textures/blocks/cactus_side.png"
    top = "assets/minecraft/textures/blocks/cactus_top.png"
    bottom = "assets/minecraft/textures/blocks/cactus_bottom.png"
    model = BlockModel()
    nx=px=nz=pz=ny=py=(0,0,1,1)
    model.vertices = [
        # NX face
        ((0+0.0625, 0, 0), (nx[0], nx[1]), (204,204,204,255)),
        ((0+0.0625, 0, 1), (nx[2], nx[1]), (204,204,204,255)),
        ((0+0.0625, 1, 1), (nx[2], nx[3]), (204,204,204,255)),
        ((0+0.0625, 1, 0), (nx[0], nx[3]), (204,204,204,255)),
        
        # NZ face
        ((1, 0, 0+0.0625), (nz[0], nz[1]), (229,229,229,255)),
        ((0, 0, 0+0.0625), (nz[2], nz[1]), (229,229,229,255)),
        ((0, 1, 0+0.0625), (nz[2], nz[3]), (229,229,229,255)),
        ((1, 1, 0+0.0625), (nz[0], nz[3]), (229,229,229,255)),
        
        # PX face
        ((1-0.0625, 0, 1), (px[0], px[1]), (204,204,204,255)),
        ((1-0.0625, 0, 0), (px[2], px[1]), (204,204,204,255)),
        ((1-0.0625, 1, 0), (px[2], px[3]), (204,204,204,255)),
        ((1-0.0625, 1, 1), (px[0], px[3]), (204,204,204,255)),
        
        # PZ face
        ((0, 0, 1-0.0625), (pz[0], pz[1]), (229,229,229,255)),
        ((1, 0, 1-0.0625), (pz[2], pz[1]), (229,229,229,255)),
        ((1, 1, 1-0.0625), (pz[2], pz[3]), (229,229,229,255)),
        ((0, 1, 1-0.0625), (pz[0], pz[3]), (229,229,229,255)),
        
        # NY face
        ((0, 0, 1), (ny[0], ny[1]), (255,255,255,255)),
        ((0, 0, 0), (ny[2], ny[1]), (255,255,255,255)),
        ((1, 0, 0), (ny[2], ny[3]), (255,255,255,255)),
        ((1, 0, 1), (ny[0], ny[3]), (255,255,255,255)),
        
        # PY face
        ((1, 1, 1), (py[0], py[1]), (255,255,255,255)),
        ((1, 1, 0), (py[2], py[1]), (255,255,255,255)),
        ((0, 1, 0), (py[2], py[3]), (255,255,255,255)),
        ((0, 1, 1), (py[0], py[3]), (255,255,255,255)),
    ]
    model.faces = [
        ([0, 1, 2, 3], chunkrenderer.FACE_TYPE_NX, side),
        ([4, 5, 6, 7], chunkrenderer.FACE_TYPE_NZ, side),
        ([8, 9, 10, 11], chunkrenderer.FACE_TYPE_PX, side),
        ([12, 13, 14, 15], chunkrenderer.FACE_TYPE_PZ, side),
        ([16, 17, 18, 19], chunkrenderer.FACE_TYPE_NY, bottom),
        ([20, 21, 22, 23], chunkrenderer.FACE_TYPE_PY, top),
    ]
    bd.add(BlockDefinition(model, transparent=True), 81)
