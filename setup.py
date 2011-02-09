from distutils.core import setup, Extension
from distutils.command.build import build
from distutils.command.clean import clean
from distutils.dir_util import remove_tree
from distutils import log
import os, os.path
import glob
import platform

try:
    import py2exe
except ImportError:
    py2exe = None

# now, setup the keyword arguments for setup
# (because we don't know until runtime if py2exe is available)
setup_kwargs = {}
setup_kwargs['options'] = {}
setup_kwargs['ext_modules'] = []
setup_kwargs['cmdclass'] = {}

#
# py2exe options
#

if py2exe != None:
    setup_kwargs['console'] = ['gmap.py']
    setup_kwargs['data_files'] = [('textures', ['textures/lava.png', 'textures/water.png', 'textures/fire.png']),
                                  ('', ['config.js', 'COPYING.txt', 'README.rst']),
                                  ('web_assets', glob.glob('web_assets/*'))]
    setup_kwargs['zipfile'] = None
    setup_kwargs['options']['py2exe'] = {'bundle_files' : 1, 'excludes': 'Tkinter'}

#
# _composite.c extension
#

setup_kwargs['ext_modules'].append(Extension('_composite', ['_composite.c'], include_dirs=['.'], extra_link_args=["/MANIFEST"] if platform.system() == "Windows" else []))
# tell build_ext to build the extension in-place
# (NOT in build/)
setup_kwargs['options']['build_ext'] = {'inplace' : 1}
# tell the build command to only run build_ext
build.sub_commands = [('build_ext', None)]

# custom clean command to remove in-place extension
class CustomClean(clean):
    def run(self):
        # do the normal cleanup
        clean.run(self)
        
        # try to remove '_composite.{so,pyd,...}' extension,
        # regardless of the current system's extension name convention
        build_ext = self.get_finalized_command('build_ext')
        pretty_fname = build_ext.get_ext_filename('_composite')
        fname = pretty_fname
        if os.path.exists(fname):
            try:
                if not self.dry_run:
                    os.remove(fname)
                log.info("removing '%s'", pretty_fname)
            except OSError:
                log.warn("'%s' could not be cleaned -- permission denied",
                         pretty_fname)
        else:
            log.debug("'%s' does not exist -- can't clean it",
                      pretty_fname)
setup_kwargs['cmdclass']['clean'] = CustomClean

###

setup(**setup_kwargs)
