#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from distutils.command.build import build
from distutils.command.clean import clean
from distutils.command.build_ext import build_ext
from distutils.command.sdist import sdist
from distutils.cmd import Command
from distutils.dir_util import remove_tree
from distutils.sysconfig import get_python_inc
from distutils import log
import sys, os, os.path
import glob
import platform
import time
import overviewer.util as util
import numpy

try:
    import py2exe
except ImportError:
    py2exe = None

try:
    import py2app
    from setuptools.extension import Extension
except ImportError:
    py2app = None

# make sure our current working directory is the same directory
# setup.py is in
curdir = os.path.split(sys.argv[0])[0]
if curdir:
    os.chdir(curdir)

# now, setup the keyword arguments for setup
# (because we don't know until runtime if py2exe/py2app is available)
setup_kwargs = {}
setup_kwargs['ext_modules'] = []
setup_kwargs['cmdclass'] = {}
setup_kwargs['options'] = {}

#
# metadata
#

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(fname).read()

setup_kwargs['name'] = 'Minecraft-Overviewer'
setup_kwargs['version'] = util.findGitVersion()
setup_kwargs['description'] = 'Generates large resolution images of a Minecraft map.'
setup_kwargs['url'] = 'http://overviewer.org/'
setup_kwargs['author'] = 'Andrew Brown'
setup_kwargs['author_email'] = 'brownan@gmail.com'
setup_kwargs['license'] = 'GNU General Public License v3'
setup_kwargs['long_description'] = read('README.rst')

# top-level files that should be included as documentation
doc_files = ['COPYING.txt', 'README.rst', 'CONTRIBUTORS.rst', 'sample_config.py']

# helper to create a 'data_files'-type sequence recursively for a given dir
def recursive_data_files(src, dest=None):
    if dest is None:
        dest = src

    ret = []
    for dirpath, dirnames, filenames in os.walk(src):
        current_dest = os.path.relpath(dirpath, src)
        if current_dest == '.':
            current_dest = dest
        else:
            current_dest = os.path.join(dest, current_dest)

        current_sources = map(lambda p: os.path.join(dirpath, p), filenames)

        ret.append((current_dest, current_sources))
    return ret

# helper to create a 'package_data'-type sequence recursively for a given dir
def recursive_package_data(src, package_dir='overviewer'):
    full_src = os.path.join(package_dir, src)
    ret = []
    for dirpath, dirnames, filenames in os.walk(full_src, topdown=False):
        current_path = os.path.relpath(dirpath, package_dir)
        for filename in filenames:
            ret.append(os.path.join(current_path, filename))

    return ret

#
# py2exe options
#

if py2exe is not None:
    setup_kwargs['comments'] = "http://overviewer.org"
    # py2exe likes a very particular type of version number:
    setup_kwargs['version'] = util.findGitVersion().replace("-",".")

    setup_kwargs['console'] = ['contribManager.py']
    setup_kwargs['data_files'] = [('', doc_files)]
    setup_kwargs['data_files'] += recursive_data_files('overviewer/data/web_assets', 'web_assets')
    setup_kwargs['data_files'] += recursive_data_files('overviewer/data/js_src', 'js_src')
    setup_kwargs['data_files'] += recursive_data_files('contrib', 'contrib')
    setup_kwargs['zipfile'] = None
    if platform.system() == 'Windows' and '64bit' in platform.architecture():
        b = 3
    else:
        b = 1
    setup_kwargs['options']['py2exe'] = {'bundle_files' : b, 'excludes': 'Tkinter', 'includes':
        ['fileinput', 'overviewer.aux_files.genPOI']}

#
# py2app options
#

#if py2app is not None:
#    setup_kwargs['app'] = ['overviewer.py']
#    setup_kwargs['options']['py2app'] = {'argv_emulation' : False}
#    setup_kwargs['setup_requires'] = ['py2app']

#
# script, package, and data
#

setup_kwargs['packages'] = ['overviewer', 'overviewer/aux_files']
setup_kwargs['scripts'] = []
setup_kwargs['package_data'] = {'overviewer': recursive_package_data('data/web_assets') + recursive_package_data('data/js_src')}

if py2exe is None:
    setup_kwargs['data_files'] = [('share/doc/minecraft-overviewer', doc_files)]

# oil extension
#
# STILL TODO: verify libpng is present
#

oil_files = [
    "oil-python.c",
    "oil-matrix.c",
    "oil-image.c",
    "oil-format.c",
    "oil-format-png.c",
    "oil-palette.c",
    "oil-dither.c",
    "oil-backend.c",
    "oil-backend-cpu.c",
    "oil-backend-debug.c",
    "oil-backend-cpu-sse.c",
    "oil-backend-opengl.c",
]

oil_includes = [
    "oil.h",
    "oil-python.h",
    "oil-dither-private.h",
    "oil-format-private.h",
    "oil-image-private.h",
    "oil-palette-private.h",
    "oil-backend-private.h",
    "oil-backend-cpu.def",
]

oil_files = ['overviewer/oil/' + s for s in oil_files]
oil_includes = ['overviewer/oil/' + s for s in oil_includes]
setup_kwargs['ext_modules'].append(Extension('overviewer.oil', oil_files, depends=oil_includes, libraries=['png']))

# chunkrenderer extension
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

chunkrenderer_files = ['chunkrenderer.c']
chunkrenderer_includes = []

chunkrenderer_files = ['overviewer/chunkrenderer/' + s for s in chunkrenderer_files]
chunkrenderer_includes = ['overviewer/chunkrenderer/' + s for s in chunkrenderer_includes]

# todo: better oil handling!
setup_kwargs['ext_modules'].append(Extension('overviewer.chunkrenderer', chunkrenderer_files, include_dirs=[numpy_include, 'overviewer/oil'], depends=chunkrenderer_includes, extra_objects=['overviewer/oil.so']))


# tell build_ext to build the extension in-place
# (NOT in build/)
setup_kwargs['options']['build_ext'] = {'inplace' : 1}

# custom clean command to remove in-place extension
# and the version file, primitives header
class CustomClean(clean):
    def run(self):
        # do the normal cleanup
        clean.run(self)

        # try to remove '_composite.{so,pyd,...}' extension,
        # regardless of the current system's extension name convention
        build_ext = self.get_finalized_command('build_ext')
        ext_fname = build_ext.get_ext_filename('overviewer.chunkrenderer')
        oil_fname = build_ext.get_ext_filename('overviewer.oil')
        versionpath = os.path.join("overviewer", "overviewer_version.py")

        for fname in [ext_fname, oil_fname]:
            if os.path.exists(fname):
                try:
                    log.info("removing '%s'", fname)
                    if not self.dry_run:
                        os.remove(fname)

                except OSError:
                    log.warn("'%s' could not be cleaned -- permission denied",
                             fname)
            else:
                log.debug("'%s' does not exist -- can't clean it",
                          fname)

        # now try to purge all *.pyc files
        for root, dirs, files in os.walk(os.path.join(os.path.dirname(__file__), ".")):
            for f in files:
                if f.endswith(".pyc"):
                    if self.dry_run:
                        log.warn("Would remove %s", os.path.join(root,f))
                    else:
                        os.remove(os.path.join(root, f))

def generate_version_py():
    try:
        outstr = ""
        outstr += "VERSION=%r\n" % util.findGitVersion()
        outstr += "HASH=%r\n" % util.findGitHash()
        outstr += "BUILD_DATE=%r\n" % time.asctime()
        outstr += "BUILD_PLATFORM=%r\n" % platform.processor()
        outstr += "BUILD_OS=%r\n" % platform.platform()
        f = open("overviewer/overviewer_version.py", "w")
        f.write(outstr)
        f.close()
    except Exception:
        print "WARNING: failed to build overviewer_version file"

class CustomSDist(sdist):
    def run(self):
        # generate the version file
        generate_version_py()
        sdist.run(self)

class CustomBuild(build):
    def run(self):
        # generate the version file
        generate_version_py()
        build.run(self)
        print "\nBuild Complete"

class CustomBuildExt(build_ext):
    user_options = build_ext.user_options + [
        ('with-sse', None, "build with SSE CPU backend"),
        ('with-opengl', None, "build with OpenGL GPU backend"),
    ]

    def initialize_options(self):
        self.with_sse = False
        self.with_opengl = False
        build_ext.initialize_options(self)
    
    def build_extensions(self):
        c = self.compiler.compiler_type
        if c == "msvc":
            # customize the build options for this compilier
            for e in self.extensions:
                e.extra_link_args.append("/MANIFEST")
        if c == "unix":
            # customize the build options for this compilier
            for e in self.extensions:
                # optimizations
                e.extra_compile_args.append("-ffast-math")
                e.extra_compile_args.append("-O2")
                
                # warnings and errors (quality control \o/)
                e.extra_compile_args.append("-Wno-unused-variable")
                e.extra_compile_args.append("-Wno-unused-function")
                e.extra_compile_args.append("-Wdeclaration-after-statement")
                p = platform.linux_distribution()
                if not (p[0] == 'CentOS' and p[1][0] == '5'):
                    e.extra_compile_args.append("-Werror=declaration-after-statement")

        # various optional features
        for e in self.extensions:
            if self.with_sse:
                e.define_macros.append(("ENABLE_CPU_SSE_BACKEND", None))
            if self.with_opengl:
                e.define_macros.append(("ENABLE_OPENGL_BACKEND", None))
                e.libraries.append("X11")
                e.libraries.append("GL")
                e.libraries.append("GLEW")

        # build in place, and in the build/ tree
        self.inplace = True
        build_ext.build_extensions(self)
        self.inplace = False
        build_ext.build_extensions(self)


setup_kwargs['cmdclass']['clean'] = CustomClean
setup_kwargs['cmdclass']['sdist'] = CustomSDist
setup_kwargs['cmdclass']['build'] = CustomBuild
setup_kwargs['cmdclass']['build_ext'] = CustomBuildExt
###

setup(**setup_kwargs)

