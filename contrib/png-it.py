"""
Outputs one huge PNG file using the tiles from an Overviewer map.
"""

from __future__ import print_function

import sys
from argparse import ArgumentParser
from glob import glob
from os.path import exists, join, split

from PIL import Image


def main():
    parser = ArgumentParser()

    parser.add_argument('--memory-limit', '-m', metavar='SIZE', type=int, dest='memory_limit',
                        required=True, help="Limit the amount of RAM to use in MiB. If it is "
                        "expected that we'll exceed the limit, the script will abort.")
    parser.add_argument('--zoom-level', '-z', metavar='LEVEL', type=int, dest='zoom_level',
                        required=True, help="Which zoom level to use from the Overviewer map. "
                        "NOTE: the RAM usage will increase exponentially with the zoom level.")
    parser.add_argument('--crop', '-c', metavar='CROP', type=int, dest='crop', default=0,
                        help="Crop a frame around the image, as a percentage of the original. "
                        "For example in a image of 1000x2000 pixels, a 10%% crop will crop 100 "
                        "pixels in the left and right sides and 200 pixels in the bottom and "
                        "top sides. NOTE: this is not exact but will be rounded to the nearest "
                        "Overviewer map tile.")
    parser.add_argument('--center', '-e', metavar='X,Y', dest='center', default=None,
                        help="Mark what will be the center of the image, two comma separated "
                        "percentage values.")
    parser.add_argument('--autocrop', '-a', dest='autocrop', default=False, action='store_true',
                        help="Calculate the center and crop vales automatically to show all the "
                        "tiles in the smallest possible image size. Unless you want a very "
                        "specific image this option is recommended.")
    parser.add_argument('--output', '-o', type=str, dest='output', default="output.png",
                        metavar='OUTPUT', help="Path for the resulting PNG. The image will be "
                        "saved as a PNG file, no matter what extension you use.")
    parser.add_argument('tileset', metavar='TILESET')

    args = parser.parse_args()

    if args.autocrop and (args.center or args.crop):
        parser.error("You cannot specify --center or --crop with --autocrop.")

    # check for the output
    if args.output != '-':
        folder, filename = split(args.output)
        if folder != '' and not exists(folder):
            parser.error("The destination folder '{0}' doesn't exist.".format(folder))

    # calculate stuff
    n = args.zoom_level
    length_in_tiles = 2**n
    tile_size = (384, 384)
    px_size = 4     # bytes

    # create a list with all the images in the zoom level
    path = args.tileset
    for i in range(args.zoom_level):
        path = join(path, "?")
    path += ".png"

    all_images = glob(path)
    if not all_images:
        print("Error! No images found in this zoom level. Is this really an Overviewer tile set "
              "directory?", file=sys.stderr)
        sys.exit(1)

    # autocrop will calculate the center and crop values automagically
    if args.autocrop:
        min_x = min_y = length_in_tiles
        max_x = max_y = 0
        counter = 0
        total = len(all_images)
        print("Checking tiles for autocrop calculations:", file=sys.stderr)
        # get the maximum and minimum tiles coordinates of the map
        for path in all_images:
            t = get_tuple_coords(args, path)
            c = get_tile_coords_from_tuple(args, t)
            min_x = min(min_x, c[0])
            min_y = min(min_y, c[1])
            max_x = max(max_x, c[0])
            max_y = max(max_y, c[1])
            counter += 1
            if (counter % 100 == 0 or counter == total or counter == 1):
                print("Checked {0} of {1}.".format(counter, total), file=sys.stderr)

        # the center of the map will be in the middle of the occupied zone
        center = (int((min_x + max_x) / 2.0), int((min_y + max_y) / 2.0))
        # see the next next comment to know what center_vector is
        center_vector = (int(center[0] - (length_in_tiles / 2.0 - 1)),
                         int(center[1] - (length_in_tiles / 2.0 - 1)))
        # I'm not completely sure why, but the - 1 factor in  ^  makes everything nicer.

        # min_x - center_vector[0] will be the unused amount of tiles in
        # the left and the right of the map (and this is true because we
        # are in the actual center of the map)
        crop = (min_x - center_vector[0], min_y - center_vector[1])

    else:
        # center_vector is the vector that joins the center tile with
        # the new center tile in tile coords
        # tile coords are how many tile are on the left, x, and
        # how many above, y. The top-left tile has coords (0,0)
        if args.center:
            center_x, center_y = args.center.split(",")
            center_x = int(center_x)
            center_y = int(center_y)
            center_tile_x = int(2**n * (center_x / 100.0))
            center_tile_y = int(2**n * (center_y / 100.0))
            center_vector = (int(center_tile_x - length_in_tiles / 2.0),
                             int(center_tile_y - length_in_tiles / 2.0))
        else:
            center_vector = (0, 0)

        # crop if needed
        tiles_to_crop = int(2**n * (args.crop / 100.0))
        crop = (tiles_to_crop, tiles_to_crop)

    final_img_size = (tile_size[0] * length_in_tiles, tile_size[1] * length_in_tiles)
    final_cropped_img_size = (final_img_size[0] - 2 * crop[0] * tile_size[0],
                              final_img_size[1] - 2 * crop[1] * tile_size[1])

    mem = final_cropped_img_size[0] * final_cropped_img_size[1] * px_size    # bytes!
    print("The image size will be {0}x{1}."
          .format(final_cropped_img_size[0], final_cropped_img_size[1]), file=sys.stderr)
    print("A total of {0} MB of memory will be used.".format(mem / 1024**2), file=sys.stderr)
    if mem / 1024.0**2.0 > args.memory_limit:
        print("Error! The expected RAM usage exceeds the specified limit. Exiting.",
              file=sys.stderr)
        sys.exit(1)

    # Create a new huge image
    final_img = Image.new("RGBA", final_cropped_img_size, (26, 26, 26, 0))

    # Paste ALL the images
    total = len(all_images)
    counter = 0
    print("Pasting images:", file=sys.stderr)
    for path in all_images:
        img = Image.open(path)
        t = get_tuple_coords(args, path)
        x, y = get_cropped_centered_img_coords(args, tile_size, center_vector, crop, t)
        final_img.paste(img, (x, y))
        counter += 1
        if (counter % 100 == 0 or counter == total or counter == 1):
            print("Pasted {0} of {1}.".format(counter, total), file=sys.stderr)
    print("Done!", file=sys.stderr)
    print("Saving image... (this may take a while)", file=sys.stderr)
    final_img.save(args.output if args.output != '-' else sys.stdout, "PNG")


def get_cropped_centered_img_coords(options, tile_size, center_vector, crop, t):
    """ Returns the new image coords used to paste tiles in the big
    image. Takes options, the size of tiles, center vector, crop value
    (see calculate stuff) and a tuple from get_tuple_coords. """
    x, y = get_tile_coords_from_tuple(options, t)
    new_tile_x = x - crop[0] - center_vector[0]
    new_tile_y = y - crop[1] - center_vector[1]

    new_img_x = new_tile_x * tile_size[0]
    new_img_y = new_tile_y * tile_size[1]

    return new_img_x, new_img_y


def get_tile_coords_from_tuple(options, t):
    """ Gets a tuple of coords from get_tuple_coords and returns
    the number of tiles from the top left corner to this tile.
    The top-left tile has coordinates (0,0) """
    x = 0
    y = 0
    z = options.zoom_level
    n = 1

    for i in t:
        if i == 1:
            x += 2**(z - n)
        elif i == 2:
            y += 2**(z - n)
        elif i == 3:
            x += 2**(z - n)
            y += 2**(z - n)
        n += 1
    return (x, y)


def get_tuple_coords(options, path):
    """ Extracts the "quadtree coordinates" (the numbers in the folder
    of the tile sets) from an image path. Returns a tuple with them.
    The upper most folder is in the left of the tuple."""
    l = []
    path, head = split(path)
    head = head.split(".")[0]   # remove the .png
    l.append(int(head))
    for i in range(options.zoom_level - 1):
        path, head = split(path)
        l.append(int(head))
    # the list is reversed
    l.reverse()
    return tuple(l)


def get_image(tileset, t):
    """ Returns the path of an image, takes a tuple with the
    "quadtree coordinates", these are the numbers in the folders of the
    tile set. """
    path = tileset
    for d in t:
        path = join(path, str(d))
    path += ".png"
    return path


if __name__ == '__main__':
    main()
