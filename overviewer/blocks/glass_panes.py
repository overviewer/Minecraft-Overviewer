from overviewer.blockdefinitions import BlockModel, BlockDefinition
from overviewer import chunkrenderer

tex = "assets/minecraft/textures/blocks/glass.png"

sticky_neighbors = set([

    ])

glass = BlockDefinition(datavalue=chunkrenderer.BLOCK_DATA_STICKY, dataparameter=sticky_neighbors)

def add(bd):
    pass
