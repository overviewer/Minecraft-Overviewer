# This is a sample config file, meant to give you an idea of how to format your
# config file and what's possible.

# Define the path to your world here. 'My World' in this case will show up as
# the world name on the map interface. If you change it, be sure to also change
# the referenced world names in the render definitions below.
worlds['My World'] = "/path/to/your/world"

# Define where to put the output here.
outputdir = "/tmp/test_render"

# The Google Maps javascript API now requires an API key. They're free and
# easy to get. Go here: https://console.developers.google.com/apis/credentials
google_api_key = "your-api-key-here"

# This is an item usually specified in a renders dictionary below, but if you
# set it here like this, it becomes the default for all renders that don't
# define it.
# Try "smooth_lighting" for even better looking maps!
rendermode = "lighting"

renders["render1"] = {
        'world': 'My World',
        'title': 'A regular render',
}

# This example is the same as above, but rotated
renders["render2"] = {
        'world': 'My World',
        'northdirection': 'upper-right',
        'title': 'Upper-right north direction',
}

# Here's how to do a nighttime render. Also try "smooth_night" instead of "night"
renders["render3"] = {
        'world': 'My World',
        'title': 'Nighttime',
        # Notice how this overrides the rendermode default specified above
        'rendermode': 'night',
}

