"""
Outputs a huge PNG file using the tiles from a overviewer map.
"""

from optparse import OptionParser
from PIL import Image
from os.path import join, split, exists
from glob import glob
import sys

def main():
    
    usage = 'usage: %prog [options] <tile-set-folder>'

    parser = OptionParser(description='',\
    prog = 'png_it', version='0.0.1', usage=usage)

    parser.add_option('--memory-limit', '-m', help = 'Limit the amount of ram to use in MB. If it\'s expected to exceed the limit it won\'t do anything.',\
        metavar = '<memory>', type = int, dest = 'memory_limit', default = None)
    
    parser.add_option('--zoom-level', '-z', help = 'Which zoom level to use from the overviewer map. NOTE: the RAM usage will increase exponentially with the zoom level.',\
        metavar = '<zoom-level>', type = int, dest = 'zoom_level', default = None)

    parser.add_option('--crop', '-c', help = 'It will crop a frame around the image, give it in percentage. For example in a image of 1000x2000 pixels, a 10% crop will crop 100 pixels in the left, right sides and 200 pixels in the bottom and top sides. NOTE: this is no exact, it will be rounded to the nearest overviewer map tile.',\
        metavar = '<crop>', type = int, dest = 'crop', default = 0)

    parser.add_option('--center', '-e', help = 'Mark what will be the center of the image, two percentage values comma separated',\
        metavar = '<center>', type = str, dest = 'center', default = None)

    parser.add_option('--autocrop', '-a', help = 'Calculates the center and crop vales automatically to show all the tiles in the minimun image size.Unless you want a very specific image this options is very recommendedable.',\
         action = 'store_true', dest = 'autocrop', default = False)

    parser.add_option('--output', '-o', help = 'Path for the resulting PNG. It will save it as PNG, no matter what extension do you use.',\
        metavar = '<output>', type = str, dest = 'output', default = "output.png")

    (options, args) = parser.parse_args()
    

    # arg is overviewer tile set folder
    if len(args) > 1:
        parser.error("Error! Only one overviewer tile set accepted as input. Use --help for a complete list of options.")

    if not args:
        parser.error("Error! Need an overviewer tile set folder. Use --help for a complete list of options.")
    tileset = args[0] #set the tileset dir after ensuring it's been provided, not before.
    
    if not options.zoom_level:
        parser.error("Error! The option zoom-level is mandatory.")
    
    if options.autocrop and (options.center or options.crop):
        parser.error("Error! You can't mix --autocrop with --center or --crop.")
    
    # check for the output
    folder, filename = split(options.output)
    if folder != '' and not exists(folder):
        parser.error("The destination folder \'{0}\' doesn't exist.".format(folder))
        
    # calculate stuff
    n = options.zoom_level
    length_in_tiles = 2**n
    tile_size = (384,384)
    px_size = 4 # bytes

    # create a list with all the images in the zoom level
    path = tileset
    for i in range(options.zoom_level):
        path = join(path, "?")
    path += ".png"
    
    all_images = glob(path)
    if not all_images:
        print "Error! No images found in this zoom level. Is this really an overviewer tile set directory?"
        sys.exit(1)

    # autocrop will calculate the center and crop values automagically
    if options.autocrop:
        min_x = min_y = length_in_tiles
        max_x = max_y = 0
        counter = 0
        total = len(all_images)
        print "Checking tiles for autocrop calculations:"
        # get the maximun and minimun tiles coordinates of the map
        for path in all_images:
            t = get_tuple_coords(options, path)
            c = get_tile_coords_from_tuple(options, t)
            min_x = min(min_x, c[0])
            min_y = min(min_y, c[1])
            max_x = max(max_x, c[0])
            max_y = max(max_y, c[1])
            counter += 1
            if (counter % 100 == 0 or counter == total or counter == 1): print "Checked {0} of {1}".format(counter, total)
        
        # the center of the map will be in the middle of the occupied zone
        center = (int((min_x + max_x)/2.), int((min_y + max_y)/2.))
        # see the next next comment to know what's center_vector
        center_vector = (int(center[0] - (length_in_tiles/2. - 1)), int(center[1] - (length_in_tiles/2. - 1)))
        # I'm not completely sure why, but the - 1 factor in  ^  makes everything nicer.
        
        # min_x - center_vector[0] will be the unused amount of tiles in
        # the left and the right of the map (and this is true because we
        # are in the actual center of the map)
        crop = (min_x - center_vector[0], min_y - center_vector[1])
        
    else:
        # center_vector is the vector that joins the center tile with
        # the new center tile in tile coords
        #(tile coords are how many tile are on the left, x, and 
        # how many above, y. The top-left tile has coords (0,0)
        if options.center:
            center_x, center_y = options.center.split(",")
            center_x = int(center_x)
            center_y = int(center_y)
            center_tile_x = int(2**n*(center_x/100.))
            center_tile_y = int(2**n*(center_y/100.))
            center_vector = (int(center_tile_x - length_in_tiles/2.), int(center_tile_y - length_in_tiles/2.))
        else:
            center_vector = (0,0)

        # crop if needed
        tiles_to_crop = int(2**n*(options.crop/100.))
        crop = (tiles_to_crop, tiles_to_crop)

    final_img_size = (tile_size[0]*length_in_tiles,tile_size[1]*length_in_tiles)
    final_cropped_img_size = (final_img_size[0] - 2*crop[0]*tile_size[0],final_img_size[1] - 2*crop[1]*tile_size[1])

    mem = final_cropped_img_size[0]*final_cropped_img_size[1]*px_size # bytes!
    print "The image size will be {0}x{1}".format(final_cropped_img_size[0],final_cropped_img_size[1])
    print "A total of {0} MB of memory will be used.".format(mem/1024**2)
    if mem/1024.**2. > options.memory_limit:
        print "Warning! The expected RAM usage exceeds the specified limit. Exiting."
        sys.exit(1)

    # Create a new huge image
    final_img = Image.new("RGBA", final_cropped_img_size, (26, 26, 26, 0))

    # Paste ALL the images
    total = len(all_images)
    counter = 0
    print "Pasting images:"
    for path in all_images:
        
        img = Image.open(path)
        t = get_tuple_coords(options, path)
        x, y = get_cropped_centered_img_coords(options, tile_size, center_vector, crop, t)
        final_img.paste(img, (x, y))
        counter += 1
        if (counter % 100 == 0 or counter == total or counter == 1): print "Pasted {0} of {1}".format(counter, total)
    print "Done!"
    print "Saving image... (this can take a while)"
    final_img.save(options.output, "PNG")


def get_cropped_centered_img_coords(options, tile_size, center_vector, crop, t):
    """ Returns the new image coords used to paste tiles in the big 
    image. Takes options, the size of tiles, center vector, crop value 
    (see calculate stuff) and a tuple from get_tuple_coords. """
    x, y = get_tile_coords_from_tuple(options, t)
    new_tile_x = x - crop[0] - center_vector[0]
    new_tile_y = y - crop[1] - center_vector[1]
    
    new_img_x = new_tile_x*tile_size[0]
    new_img_y = new_tile_y*tile_size[1]
    
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
            x += 2**(z-n)
        elif i == 2:
            y += 2**(z-n)
        elif i == 3:
            x += 2**(z-n)
            y += 2**(z-n)
        n += 1
    return (x,y)

def get_tuple_coords(options, path):
    """ Extracts the "quadtree coordinates" (the numbers in the folder
    of the tile sets) from an image path. Returns a tuple with them.
    The upper most folder is in the left of the tuple."""
    l = []
    path, head = split(path)
    head = head.split(".")[0] # remove the .png
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
