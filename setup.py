#!/usr/bin/env python2

import sys
import traceback

# quick version check
if not (sys.version_info[0] == 2 and sys.version_info[1] >= 6):
    print("Sorry, the Overviewer requires at least Python 2.6 to run")
    if sys.version_info[0] >= 3:
        print("and will not run on Python 3.0 or later")
    sys.exit(1)

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
import os, os.path
import glob
import platform
import time
import overviewer_core.util as util
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
def recursive_package_data(src, package_dir='overviewer_core'):
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

    setup_kwargs['console'] = ['overviewer.py', 'contribManager.py']
    setup_kwargs['data_files'] = [('', doc_files)]
    setup_kwargs['data_files'] += recursive_data_files('overviewer_core/data/textures', 'textures')
    setup_kwargs['data_files'] += recursive_data_files('overviewer_core/data/web_assets', 'web_assets')
    setup_kwargs['data_files'] += recursive_data_files('overviewer_core/data/js_src', 'js_src')
    setup_kwargs['data_files'] += recursive_data_files('contrib', 'contrib')
    setup_kwargs['zipfile'] = None
    if platform.system() == 'Windows' and '64bit' in platform.architecture():
        b = 3
    else:
        b = 1
    setup_kwargs['options']['py2exe'] = {'bundle_files' : b, 'excludes': 'Tkinter', 'includes':
        ['fileinput', 'overviewer_core.items', 'overviewer_core.aux_files.genPOI']}

#
# py2app options
#

if py2app is not None:
    setup_kwargs['app'] = ['overviewer.py']
    setup_kwargs['options']['py2app'] = {'argv_emulation' : False}
    setup_kwargs['setup_requires'] = ['py2app']

#
# script, package, and data
#

setup_kwargs['packages'] = ['overviewer_core', 'overviewer_core/aux_files']
setup_kwargs['scripts'] = ['overviewer.py']
setup_kwargs['package_data'] = {'overviewer_core': recursive_package_data('data/textures') + recursive_package_data('data/web_assets') + recursive_package_data('data/js_src')}

if py2exe is None:
    setup_kwargs['data_files'] = [('share/doc/minecraft-overviewer', doc_files)]


#
# c_overviewer extension
#

# Third-party modules - we depend on numpy for everything
# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

try:
    pil_include = os.environ['PIL_INCLUDE_DIR'].split(os.pathsep)
except Exception:
    pil_include = [ os.path.join(get_python_inc(plat_specific=1), 'Imaging') ]
    if not os.path.exists(pil_include[0]):
        pil_include = [ ]


# used to figure out what files to compile
# auto-created from files in primitives/, but we need the raw names so
# we can use them later.
primitives = []
for name in glob.glob("overviewer_core/src/primitives/*.c"):
    name = os.path.split(name)[-1]
    name = os.path.splitext(name)[0]
    primitives.append(name)

c_overviewer_files = ['main.c', 'composite.c', 'iterate.c', 'endian.c', 'rendermodes.c']
c_overviewer_files += map(lambda mode: 'primitives/%s.c' % (mode,), primitives)
c_overviewer_files += ['Draw.c']
c_overviewer_includes = ['overviewer.h', 'rendermodes.h']

c_overviewer_files = map(lambda s: 'overviewer_core/src/'+s, c_overviewer_files)
c_overviewer_includes = map(lambda s: 'overviewer_core/src/'+s, c_overviewer_includes)

setup_kwargs['ext_modules'].append(Extension('overviewer_core.c_overviewer', c_overviewer_files, include_dirs=['.', numpy_include] + pil_include, depends=c_overviewer_includes, extra_link_args=[]))


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
        ext_fname = build_ext.get_ext_filename('overviewer_core.c_overviewer')
        versionpath = os.path.join("overviewer_core", "overviewer_version.py")
        primspath = os.path.join("overviewer_core", "src", "primitives.h")

        for fname in [ext_fname, primspath]:
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
        f = open("overviewer_core/overviewer_version.py", "w")
        f.write(outstr)
        f.close()
    except Exception:
        print("WARNING: failed to build overviewer_version file")

def generate_primitives_h():
    global primitives
    prims = [p.lower().replace('-', '_') for p in primitives]

    outstr = "/* this file is auto-generated by setup.py */\n"
    for p in prims:
        outstr += "extern RenderPrimitiveInterface primitive_{0};\n".format(p)
    outstr += "static RenderPrimitiveInterface *render_primitives[] = {\n"
    for p in prims:
        outstr += "    &primitive_{0},\n".format(p)
    outstr += "    NULL\n"
    outstr += "};\n"

    with open("overviewer_core/src/primitives.h", "w") as f:
        f.write(outstr)

class CustomSDist(sdist):
    def run(self):
        # generate the version file
        generate_version_py()
        generate_primitives_h()
        sdist.run(self)

class CustomBuild(build):
    def run(self):
        # generate the version file
        try:
            generate_version_py()
            generate_primitives_h()
            build.run(self)
            print("\nBuild Complete")
        except Exception:
            traceback.print_exc(limit=1)
            print("\nFailed to build Overviewer!")
            print("Please review the errors printed above and the build instructions")
            print("at <http://docs.overviewer.org/en/latest/building/>.  If you are")
            print("still having build problems, file an incident on the github tracker")
            print("or find us in IRC.")
            sys.exit(1)

class CustomBuildExt(build_ext):
    def build_extensions(self):
        c = self.compiler.compiler_type
        if c == "msvc":
            # customize the build options for this compilier
            for e in self.extensions:
                e.extra_link_args.append("/MANIFEST")
                e.extra_link_args.append("/DWINVER=0x060")
                e.extra_link_args.append("/D_WIN32_WINNT=0x060")
        if c == "unix":
            # customize the build options for this compilier
            for e in self.extensions:
                e.extra_compile_args.append("-Wno-unused-variable") # quell some annoying warnings
                e.extra_compile_args.append("-Wno-unused-function") # quell some annoying warnings
                e.extra_compile_args.append("-Wdeclaration-after-statement")
                e.extra_compile_args.append("-Werror=declaration-after-statement")


        # build in place, and in the build/ tree
        self.inplace = False
        build_ext.build_extensions(self)
        self.inplace = True
        build_ext.build_extensions(self)


setup_kwargs['cmdclass']['clean'] = CustomClean
setup_kwargs['cmdclass']['sdist'] = CustomSDist
setup_kwargs['cmdclass']['build'] = CustomBuild
setup_kwargs['cmdclass']['build_ext'] = CustomBuildExt
###

setup(**setup_kwargs)

