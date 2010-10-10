import os
import subprocess
import shlex

def optimize_image(imgpath, imgformat, optimizeimg):
    if imgformat == 'png':
	if optimizeimg == "1" or optimizeimg == "2":
            # we can't do an atomic replace here because windows is terrible
            # so instead, we make temp files, delete the old ones, and rename
            # the temp files. go windows!
	    subprocess.Popen(shlex.split("pngcrush " + imgpath + " " + imgpath + ".tmp"),
		stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
	    os.remove(imgpath)
            os.rename(imgpath+".tmp", imgpath)

	if optimizeimg == "2":
	    subprocess.Popen(shlex.split("optipng " + imgpath), stderr=subprocess.STDOUT,
	        stdout=subprocess.PIPE).communicate()[0]
	    subprocess.Popen(shlex.split("advdef -z4 " + imgpath), stderr=subprocess.STDOUT,
	    	stdout=subprocess.PIPE).communicate()[0]

