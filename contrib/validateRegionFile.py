#!/usr/bin/env python

'''
Validate a region file

TODO description here'''

import os
import sys

# incantation to be able to import overviewer_core
if not hasattr(sys, "frozen"):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0], '..')))

from overviewer_core import nbt

def check_region(region_filename):
    chunk_errors = []
    if not os.path.exists(region_filename):
        raise Exception('Region file not found: %s' % region_filename)
    try:
        region = nbt.load_region(region_filename, 'lower-left')
    except IOError, e:
        raise Exception('Error loading region (%s): %s' % (region_filename, e))
    try:
        region.get_chunk_info(False)
        chunks = region.get_chunks()
    except IOError, e:
        raise Exception('Error reading region header (%s): %s' % (region_filename, e))
    except Exception, e:
        raise Exception('Error reading region (%s): %s' % (region_filename, e))
    for x,y in chunks:
        try:
            check_chunk(region, x, y)
        except Exception, e:
            chunk_errors.append(e)
    return (chunk_errors, len(chunks))
    
def check_chunk(region, x, y):
    try:
        data = region.load_chunk(x ,y)
    except Exception, e:
        raise Exception('Error reading chunk (%i, %i): %s' % (x, y, e))
    if data is None:
        raise Exception('Chunk (%i, %i) is unexpectedly empty' % (x, y))
    else:
        try:
            processed_data = data.read_all()
        except Exception, e:
            raise Exception('Error reading chunk (%i, %i) data: %s' % (x, y, e))
        if processed_data == []:
            raise Exception('Chunk (%i, %i) is an unexpectedly empty set' % (x, y))

if __name__ == '__main__':
    try:
        from optparse import OptionParser

        parser = OptionParser(usage='python contrib/%prog [OPTIONS] <path/to/regions|path/to/regions/*.mcr|regionfile1.mcr regionfile2.mcr ...>',
                              description='This script will valide a minecraft region file for errors.')
        parser.add_option('-v', dest='verbose', action='store_true', help='Print additional information.')
        opts, args = parser.parse_args()
        
        region_files = []
        for path in args:
            if os.path.isdir(path):
                for dirpath, dirnames, filenames in os.walk(path, True):
                    for filename in filenames:
                        if filename.startswith('r.') and filename.endswith('.mcr'):
                            if filename not in region_files:
                                region_files.append(os.path.join(dirpath, filename))
                        elif opts.verbose:
                            print('Ignoring non-region file: %s' % os.path.join(dirpath, filename))
            elif os.path.isfile(path):
                dirpath,filename = os.path.split(path)
                if filename.startswith('r.') and filename.endswith('.mcr'):
                    if path not in region_files:
                        region_files.append(path)
                else:
                    print('Ignoring non-region file: %s' % path)
            else:
                if opts.verbose:
                    print('Ignoring arg: %s' % path)
        if len(region_files) < 1:
            print 'You must list at least one region file.'
            parser.print_help()
            sys.exit(1)
        else:
            overall_chunk_total = 0
            bad_chunk_total = 0
            bad_region_total = 0
            for region_file in region_files:
                try:
                    (chunk_errors, region_chunks) = check_region(region_file)
                    bad_chunk_total += len(chunk_errors)
                    overall_chunk_total += region_chunks
                except Exception, e:
                    bad_region_total += 1
                    print('FAILED(%s): %s' % (region_file, e))
                else:
                    if len(chunk_errors) is not 0:
                        print('WARNING(%s) Chunks: %i/%' % (region_file, region_chunks - len(chunk_errors), region_chunks))
                        if opts.verbose:
                            for error in chunk_errors:
                                print(error)
                    elif opts.verbose:
                            print ('PASSED(%s) Chunks: %i/%i' % (region_file, region_chunks - len(chunk_errors), region_chunks))
            if opts.verbose:
                print 'REGIONS: %i/%i' % (len(region_files) - bad_region_total, len(region_files))
                print 'CHUNKS: %i/%i' % (overall_chunk_total - bad_chunk_total, overall_chunk_total)
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception, e:
        print('ERROR: %s' % e)

