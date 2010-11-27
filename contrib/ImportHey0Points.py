#!/usr/bin/python
'''
Generates an overviewer.dat for points of interest from Hey0 server mod
'''
import sys
import os
import cPickle
from optparse import OptionParser

sys.path.append('.')


def loadfromfile(filetype, filename):
    '''
    Parses a text file and returns a list of points
    '''
    points = []
    try:
        with open(filename,'r') as infile:
            for line in infile:
                parts = line.split(':')
                points.append([filetype, parts[0], parts[1], parts[2], parts[3]])
        return points
    except IOError:
        print "Could not open", filename
        sys.exit(1)

def main():
    '''
    Main function. Sets up OptionParser, does basic validation
    '''

    helptext = """
%prog [--home] [--warps] --cachedir dir --bindir dir"""

    parser = OptionParser(usage = helptext)
    parser.add_option('--home', 
                      dest = 'parsehome', 
                      help = 'Adds markers for homes.txt', 
                      action = 'store_true')
    parser.add_option('--warps', 
                      dest = 'parsewarps', 
                      help = 'Adds markers for warps.txt', 
                      action = 'store_true')
    parser.add_option('--cachedir', 
                      dest = 'cachedir', 
                      help = 'Directory where Overviewer has its cache dir')
    parser.add_option('--bindir', 
                      dest = 'bindir', 
                      help = 'The bin directory for Hey0 server mod (Location of homes.txt and warps.txt)')

    options, args = parser.parse_args()
    if (options.bindir != '' and options.cachedir != '') and (not options.parsehome or not options.parsewarps):
        print "You must specify --cachedir --bindir and and at least one file to process"
        parser.print_help()
        sys.exit(1)

    picklefile = os.path.join(options.cachedir,'overviewer.dat')

    pois = []
    if options.parsehome:
        print "Processing Homes"
        homefile = os.path.join(options.bindir, 'homes.txt')
        pois.extend(loadfromfile('home', homefile))

    if options.parsewarps:
        print "Processing Warps"
        warpfile = os.path.join(options.bindir, 'warps.txt')
        pois.extend(loadfromfile('warp', warpfile))

    print "Reticulating Splines"
    outpois = []
    for poi in pois:
        print poi
        newpoi = dict(
            type = poi[0],
            x = poi[2],
            y = poi[3],
            z = poi[4],
            msg = poi[1],
            chunk = (int(float(poi[2])/16), int(float(poi[4])/16)))
        outpois.append(newpoi)
    
    print "Saving POIs"
    with open(picklefile, 'wb') as outfile:
        cPickle.dump(dict(POI = outpois), outfile)
if __name__ == '__main__':
    main()
