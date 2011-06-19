################################################################################
# Please see the README or https://github.com/brownan/Minecraft-Overviewer/wiki/DTT-Upgrade-Guide
# for more details.

# This file is not meant to be used directly, but instead it is supposed to
# provide examples of interesting things you can do with the settings file. Most
# of the time, a simple 'setting_name = value' will work.

# This file is a python script, so you can import any python module you wish or
# use any built-in python function, though this is not normally necessary

# Lines that start with a hash mark are comments

# Some variables come with defaults (like procs or rendermode)
# If you specify a configuration option in both a settings.py file and on the 
# command line, the value from the command line will take precedence

################################################################################
### procs
## Specify the number of processors to use for rendering
## Default: The number of CPU cores on your machine
## Type: integer
## Example: set the number of processors to use to be 1 less than the number of
##          CPU cpus in your machine

import multiprocessing
procs = multiprocessing.cpu_count() - 1
if procs < 1: procs = 1


################################################################################
### zoom
## Sets the zoom level manually instead of calculating it. This can be useful
## if you have outlier chunks that make your world too big.  This value will
## make the highest zoom level contain (2**ZOOM)^2 tiles
## Normally you should not need to set this variable.
## Default: Automatically calculated from your world
## Type: integer
## Example:

zoom = 9

################################################################################
### regionlist
## A file containing, on each line, a path to a chunkfile to update. Instead
## of scanning the world directory for chunks, it will just use this list. 
## Normal caching rules still apply.
## Default: not yet
## Type: string
## Example:  Dynamically create regionlist of only regions older than 2 days

import os, time
# the following two lines are needed to the lambda to work
globals()['os'] = os
globals()['time'] = time
regionDir = os.path.join(args[0], "region")
regionFiles = filter(lambda x: x.endswith(".mcr"), os.listdir(regionDir))
def olderThanTwoDays(f):
    return time.time() - os.stat(os.path.join(args[0], 'region',f)).st_mtime > (60*60*24*2)
oldRegionFiles = filter(olderThanTwoDays, regionFiles)
with open("regionlist.txt", "w") as f:
    f.write("\n".join(oldRegionFiles))


################################################################################
### rendermode
## Specifies the render types
## Default: "normal"
## Type: Either a list of strings, or a single string containing modes separated
##       by commas
## Example:  Render the using the 'lighting' mode, but if today is Sunday, then 
##           also render the 'night' mode

import time
rendermode=["lighting"]
if time.localtime().tm_wday == 6:
    rendermode.append("night")


################################################################################
### imgformat
## The image output format to use. Currently supported: png(default), jpg. 
##  NOTE: png will always be used as the intermediate image format.
## Default: not yet
## Type: string
## Example:

imgformat = "jpg"



################################################################################
### optimizeimg
## If using png, perform image file size optimizations on the output. Specify 1
## for pngcrush, 2 for pngcrush+advdef, 3 for pngcrush+advdef with more agressive
## options. Option 1 gives around 19% of reduction, option 2 gives around 21% 
## (it doubles the optimizing time) and option 3 gives around 23% (it doubles, 
## again, the optimizing time). Using this option may double (or more) 
## render times. NOTE: requires corresponding programs in $PATH or %PATH%
## Default: not set
## Type: integer
## Example:

if imgformat != "jpg":
    optimizeimg = 2



################################################################################
### web_assets_hook
## If provided, run this function after the web assets have been copied, but 
## before actual tile rendering beings.  It should accept a QuadtreeGen
## object as its only argument.  Note: this is only called if skipjs is True
## Default: not yet
## Type: function
## Example:  Call an external program to generate something useful

def web_assets_hook(o):
    import subprocess
    p = subprocess.Popen(["/path/to/my/script.pl", "--output_dir", args[1]])
    p.wait()
    if p.returncode != 0:
        raise Exception("web_assets_hook failed")



################################################################################
### quiet
## Print less output.  You can specify higher values to suppress additional output
## Default: 0
## Type: integer
## Example:
quiet = 1


################################################################################
### verbose
## Print more output.  You can specify higher values to print additional output
## Default: 0
## Type: integer
## Example:
verbose = 1


################################################################################
### skipjs
## Don't output marker.js or region.js
## Default: False
## Type: boolean
## Example: Set skipjs if web_assets_hook is defined

if "web_assets_hook" in locals():
    skipjs = True




### As a reminder, don't use this file verbatim, it should only be used as
### a guide.
import sys
sys.exit("This sample-settings file shouldn't be used directly!")
