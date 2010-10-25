import os
import functools
import nbt
from array import array

base36decode = functools.partial(int, base=36)

rootdir='/minecraft/world_snapshot/'
#rootdir='/minecraft-dev/world_dev/'
nbtfile = nbt.NBTFile(rootdir+'level.dat','rb')

spawnx = nbtfile["Data"]["SpawnX"].value
spawny = nbtfile["Data"]["SpawnZ"].value

zoom=9
replaceBorder=True
replaceVoid=False
usespawn=False

if usespawn:
    originx = spawnx
    originy = spawny
else:
    originx = 0
    originy = 0

voidChunk = array('b')
for index in range(32768):
    voidChunk.append(0)

mossChunk = array('b')
for index in range(32768):
    mossChunk.append(35)

maxdistance = (2**zoom) * 12
print "Finding chunks further than " + str(maxdistance)
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file and file.startswith("c.") and file.endswith(".dat"):
            p = file.split(".")
            chunkx = base36decode(p[1])
            chunky = base36decode(p[2])
            distance = max(abs(chunkx*16 - originx), abs(chunky*16 - originy))
            #if distance > 500 and distance < 550 :
            if distance == maxdistance:
                if replaceBorder:
                        print subdir + "/" + file
                        print str(chunkx) +","+ str(chunky) + " - " + str(distance)
                        chunknbt = nbt.NBTFile(os.path.join(subdir, file))
                        chunknbt['Level']['Blocks'].value = mossChunk
                        chunknbt.write_file(os.path.join(subdir, file))
            if distance > maxdistance:
                if replaceVoid:
                        print subdir + "/" + file
                        print str(chunkx) +","+ str(chunky) + " - " + str(distance)
                        chunknbt = nbt.NBTFile(os.path.join(subdir, file))
                        chunknbt['Level']['Blocks'].value = voidChunk
                        chunknbt.write_file(os.path.join(subdir, file))
