from distutils.core import setup
import py2exe

setup(console=['gmap.py'],
        data_files=[('textures', ['textures/lava.png', 'textures/water.png']),
		('', ['template.html'])],
        zipfile = None,
        options = {'py2exe': {
            'bundle_files': 1,
            }},
        
        
	)
