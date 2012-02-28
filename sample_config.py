# This is a sample config file, meant to give you an idea of how to format your
# config file and what's possible.

# Define the path to your world here
worlds['My World'] = "/path/to/your/world"

# Define where to put the output here
outputdir = "/tmp/test_render"

# This is an item usually specified in a renders dictionary below, but if you
# set it here like this, it becomes the default for all renders that don't
# define it.
rendermode = "lighting"

renders["render1"] = {
        'world': 'My World',
        'title': 'A regular render',
}

renders["render2"] = {
        'world': 'My World',
        'northdirection': 'upper-right',
        'title': 'Upper-right north direction',
}

renders["render3"] = {
        'world': 'My World',
        'title': 'Nighttime',
        # Notice how this overrides the rendermode default specified above
        'rendermode': 'night',
}

