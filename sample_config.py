# This is a sample config file, meant to give you an idea of how to format your
# config file and what's possible.

# Define the path to your world here. 'My World' in this case will show up as
# the world name on the map interface. If you change it, be sure to also change
# the referenced world names in the render definitions below.

worlds['My World'] = "C:\\Users\\barre\\AppData\\Local\\Packages\\Microsoft.MinecraftUWP_8wekyb3d8bbwe\\LocalState\\games\\com.mojang\\minecraftWorlds\\hHeCYhnHAAA="
# worlds['My World'] = "C:\\Users\\barre\\AppData\\Roaming\\.minecraft\\saves\\Demo_World"

# Define where to put the output here.
outputdir = "C:\\Overviewer"

# This is an item usually specified in a renders dictionary below, but if you
# set it here like this, it becomes the default for all renders that don't
# define it.
# Try "smooth_lighting" for even better looking maps!
rendermode = "normal"

renders["render1"] = {
        'world': 'My World',
        'title': 'A regular render',
}

processes = 1