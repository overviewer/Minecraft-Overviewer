"""
Very basic player.dat inspection script
"""

import sys, os

# incantation to be able to import overviewer_core
if not hasattr(sys, "frozen"):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0], '..')))

from overviewer_core.nbt import load
from overviewer_core import items

def print_player(data, sub_entry=False):

    indent = ""
    if sub_entry:
        indent = "\t"
    print "%sPosition:\t%i, %i, %i\t(dim: %i)" % (indent,
            data['Pos'][0], data['Pos'][1], data['Pos'][2], data['Dimension'])
    try:
        print "%sSpawn:\t\t%i, %i, %i" % (indent,
                data['SpawnX'], data['SpawnY'], data['SpawnZ'])
    except KeyError:
        pass
    print "%sHealth:\t%i\tLevel:\t\t%i\t\tGameType:\t%i" % (indent,
            data['Health'], data['XpLevel'], data['playerGameType'])
    print "%sFood:\t%i\tTotal XP:\t%i" % (indent,
            data['foodLevel'], data['XpTotal'])
    print "%sInventory: %d items" % (indent, len(data['Inventory']))
    if not sub_entry:
        for item in data['Inventory']:
            print "  %-3d %s" % (item['Count'], items.id2item(item['id']))

if __name__ == '__main__':
    print "Inspecting %s" % sys.argv[1]

    if os.path.isdir(sys.argv[1]):
        directory = sys.argv[1]
        if len(sys.argv) > 2:
            selected_player = sys.argv[2]
        else:
            selected_player = None
        for player_file in os.listdir(directory):
            player = player_file.split(".")[0]
            if selected_player in [None, player]:
                print
                print player
                data  = load(os.path.join(directory, player_file))[1]
                print_player(data, sub_entry=(selected_player is None))
    else:
        data  = load(sys.argv[1])[1]
        print_player(data)

