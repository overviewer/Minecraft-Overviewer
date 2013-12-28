"""
This package contains block definitions. Specifically, each file defines one or
more BlockDefinition objects and a mapping of block ids to BlockDefinition
objects.

To add new files, also add them to this file.

Each block definition module should define a top-level function called "add",
which takes a BlockDefinitions object and inserts into it the block definitions
the file defines.

"""
import importlib
import os.path
import os

from .. import util
from .. import objparser

block_modules = [
        "cubes",
        "cactus",
        "glass_panes",
        ]

def get_all(bd):
    for modname in block_modules:
        full_name = "overviewer.blocks." + modname
        mod = importlib.import_module(full_name)

        mod.add(bd)
    
    # load all obj files
    objspath = os.path.join(util.get_program_path(), "overviewer", "data", "blocks")
    for root, _, files in os.walk(objspath):
        for fname in files:
            if not fname.endswith(".obj"):
                continue
            objparser.add_from_path(bd, os.path.join(root, fname))



