#!/usr/bin/env python3
"""
Very basic player.dat inspection script
"""

import os
import sys
import argparse
from pathlib import Path

# incantation to be able to import overviewer_core
if not hasattr(sys, "frozen"):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0], '..')))


from overviewer_core.nbt import load
from overviewer_core import items

def print_player(data, sub_entry=False):
    indent = ""
    if sub_entry:
        indent = "\t"
    print("%sPosition:\t%i, %i, %i\t(dim: %i)"
          % (indent, data['Pos'][0], data['Pos'][1], data['Pos'][2], data['Dimension']))
    try:
        print("%sSpawn:\t\t%i, %i, %i"
              % (indent, data['SpawnX'], data['SpawnY'], data['SpawnZ']))
    except KeyError:
        pass
    print("%sHealth:\t%i\tLevel:\t\t%i\t\tGameType:\t%i"
          % (indent, data['Health'], data['XpLevel'], data['playerGameType']))
    print("%sFood:\t%i\tTotal XP:\t%i"
          % (indent, data['foodLevel'], data['XpTotal']))
    print("%sInventory: %d items" % (indent, len(data['Inventory'])))
    if not sub_entry:
        for item in data['Inventory']:
            print("  %-3d %s" % (item['Count'], items.id2item(item['id'])))


def find_all_player_files(dir_path):
    for player_file in dir_path.iterdir():
        player = player_file.stem
        yield player_file, player


def find_player_file(dir_path, selected_player):
    for player_file, player in find_all_player_files(dir_path):
        if selected_player == player:
            return player_file, player
    raise FileNotFoundError()


def load_and_output_player(player_file_path, player, sub_entry=False):
    with player_file_path.open('rb') as f:
        player_data = load(f)[1]
    print("")
    print(player)
    print_player(player_data, sub_entry=sub_entry)


def dir_or_file(path):
    p = Path(path)
    if not p.is_file() and not p.is_dir():
        raise argparse.ArgumentTypeError("Not a valid file or directory path")
    return p


def main(path, selected_player=None):
    print("Inspecting %s" % args.path)

    if not path.is_dir():
        load_and_output_player(args.path)
        return

    if selected_player is None:
        for player_file, player in find_all_player_files(args.path):
            load_and_output_player(player_file, player)
        return

    try:
        player_file, player = find_player_file(args.path, args.selected_player)
        load_and_output_player(player_file, player, sub_entry=True)
    except FileNotFoundError:
        print("No %s.dat in %s" % (args.selected_player, args.path))
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', metavar='<Player.dat or directory>', type=dir_or_file)
    parser.add_argument('selected_player', nargs='?', default=None)

    args = parser.parse_args()
    main(args.path, selected_player=args.selected_player)
