#!/usr/bin/env python3

import sys


# quick version check
if sys.version_info[0] == 2 or (sys.version_info[0] == 3 and sys.version_info[1] < 4):
    print("Sorry, the Overviewer requires at least Python 3.4 to run.")
    sys.exit(1)


from setuptools import setup
from setuptools.extension import Extension
from distutils.command.clean import clean
from distutils.command.build import build
from setuptools.command.build_ext import build_ext
from distutils.command.sdist import sdist
from distutils.cmd import Command
from distutils.dir_util import remove_tree
from distutils.sysconfig import get_python_inc
from distutils import log

import os
import os.path
import glob
import platform
import time
import traceback

from overviewer_core import util

try:
    import py2exe
except ImportError:
    py2exe = None

try:
    import py2app
except ImportError:
    py2app = None


BUILD_ERROR = """
Failed to build Overviewer!
Please review the errors printed above and the build instructions at
<http://docs.overviewer.org/en/latest/building/>.  If you are still having
build problems, file an incident on the github tracker or find us in IRC.
"""

# make sure our current working directory is the same directory
# setup.py is in
curdir = os.path.split(sys.argv[0])[0]
if curdir:
    os.chdir(curdir)

# now, setup the keyword arguments for setup
# (because we don't know until runtime if py2exe/py2app is available)
setup_kwargs = dict({
    'name': 'Minecraft-Overviewer',
    'url': 'http://overviewer.org/',
    'version': util.findGitVersion(),
    'author': 'Andrew Brown',
    'author_email': 'brownan@gmail.com',
    'description': 'Generates large resolution images of a Minecraft map.',
    'long_description': open('README.rst', 'r').read(),
    'license': open('COPYING.txt', 'r').read(),

    'ext_modules': [],
    'cmdclass': {},
    'options': {},
    'install_requires': ['numpy', 'pillow']
})

#
# metadata
#


# top-level files that should be included as documentation
doc_files = ['COPYING.txt',
             'README.rst',
             'CONTRIBUTORS.rst',
             'sample_config.py']


def data_files(src, dest=None):
    """helper to create a 'data_files'-type sequence recursively for a given
    dir

    :param src:  directory to include files from
    :param dest:
    :return: list

    """
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


def package_data(src, package_dir='overviewer_core'):
    """Helper to create a 'package_data'-type sequence recursively for a given
    dir

    :param src: source directory
    :param package_dir: package directory
    :return: list

    """
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
    setup_kwargs['comments'] = 'http://overviewer.org'
    # py2exe likes a very particular type of version number:
    setup_kwargs['version'] = setup_kwargs['version'].replace('-', '.')

    setup_kwargs['console'] = ['overviewer.py', 'contribManager.py']
    setup_kwargs['data_files'] = [('', doc_files)]
    setup_kwargs['data_files'] += data_files('overviewer_core/data/textures',
                                             'textures')
    setup_kwargs['data_files'] += data_files('overviewer_core/data/web_assets',
                                             'web_assets')
    setup_kwargs['data_files'] += data_files('overviewer_core/data/js_src',
                                             'js_src')
    setup_kwargs['data_files'] += data_files('contrib', 'contrib')
    setup_kwargs['zipfile'] = None
    if platform.system() == 'Windows' and '64bit' in platform.architecture():
        b = 3
    else:
        b = 1
    setup_kwargs['options']['py2exe'] = \
        {'bundle_files': b,
         'excludes': 'Tkinter',
         'includes': ['fileinput',
                      'overviewer_core.items',
                      'overviewer_core.aux_files.genPOI']}

#
# py2app options
#

if py2app is not None:
    setup_kwargs['app'] = ['overviewer.py']
    setup_kwargs['options']['py2app'] = {'argv_emulation': False}
    setup_kwargs['setup_requires'] = ['py2app']

#
# script, package, and data
#

setup_kwargs['packages'] = ['overviewer_core', 'overviewer_core/aux_files']
setup_kwargs['scripts'] = ['overviewer.py']
setup_kwargs['package_data'] = \
    {'overviewer_core': (package_data('data/textures') +
                         package_data('data/web_assets') +
                         package_data('data/js_src'))}

if py2exe is None:
    setup_kwargs['data_files'] = [('share/doc/minecraft-overviewer',
                                   doc_files)]


def get_numpy_include():
    """Obtain the numpy include directory. This logic works across numpy
    versions

    :rtype: list

    """
    print(sys.path)
    try:
        import numpy
    except ImportError:
        if sys.argv[1] != 'develop':
            print('numpy not found, please run "python setup.py develop"')
            sys.exit(1)
        return []
    try:
        return numpy.get_include()
    except AttributeError:
        return numpy.get_numpy_include()

# Third-party modules - we depend on numpy for everything
numpy_include = get_numpy_include()

# Try and find the Imaging path automatically
pil_include = []
imaging_path = os.environ.get('PIL_INCLUDE_DIR',
                              os.path.normpath('./Imaging/libImaging'))
if not os.path.exists(imaging_path):
    imaging_path = os.path.join(get_python_inc(plat_specific=1), 'Imaging')
if os.path.exists(imaging_path):
    pil_include.append(imaging_path)

if not pil_include:
    print('ERROR: Could not find Python Imaging library')
    print(BUILD_ERROR)
    sys.exit(1)

SRC_PATH = 'overviewer_core/src/'

# used to figure out what files to compile
# auto-created from files in primitives/, but we need the raw names so
# we can use them later.
primitives = []
for name in glob.glob(SRC_PATH + 'primitives/*.c'):
    name = os.path.split(name)[-1]
    name = os.path.splitext(name)[0]
    primitives.append(name)

c_overviewer_files = ['main.c', 'composite.c', 'iterate.c', 'endian.c', 'rendermodes.c', 'block_class.c']
c_overviewer_files += ['primitives/%s.c' % (mode) for mode in primitives]
c_overviewer_files += ['Draw.c']
c_overviewer_includes = ['overviewer.h', 'rendermodes.h']

c_overviewer_files = ['overviewer_core/src/' + s for s in c_overviewer_files]
c_overviewer_includes = ['overviewer_core/src/' + s for s in c_overviewer_includes]

setup_kwargs['ext_modules'].append(Extension('overviewer_core.c_overviewer', c_overviewer_files, include_dirs=['.', numpy_include] + pil_include, depends=c_overviewer_includes, extra_link_args=[]))


# tell build_ext to build the extension in-place
# (NOT in build/)
setup_kwargs['options']['build_ext'] = {'inplace' : 1}

include_dirs = ['.', numpy_include] + pil_include

setup_kwargs['ext_modules'].append(Extension('overviewer_core.c_overviewer',
                                             c_overviewer_files,
                                             include_dirs=include_dirs,
                                             depends=c_overviewer_includes,
                                             extra_link_args=[]))

# tell build_ext to build the extension in-place (NOT in build/)
setup_kwargs['options']['build_ext'] = {'inplace': 1}

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
                    log.warning("'%s' could not be cleaned -- permission denied",
                                fname)
            else:
                log.debug("'%s' does not exist -- can't clean it",
                          fname)

        # now try to purge all *.pyc files
        for root, dirs, files in os.walk(os.path.join(os.path.dirname(__file__), ".")):
            for f in files:
                if f.endswith(".pyc"):
                    if self.dry_run:
                        log.warning("Would remove %s", os.path.join(root,f))
                    else:
                        os.remove(os.path.join(root, f))

def generate_version_py():
    try:
        with open('overviewer_core/overviewer_version.py', 'w') as f:
            f.write('\n'.join(['VERSION=%r' % util.findGitVersion(),
                               'HASH=%r' % util.findGitHash(),
                               'BUILD_DATE=%r' % time.asctime(),
                               'BUILD_PLATFORM=%r' % platform.processor(),
                               'BUILD_OS=%r' % platform.platform()]) + '\n')
    except Exception:
        print('WARNING: failed to build overviewer_version file')


def generate_primitives_h():
    prims = [p.lower().replace('-', '_') for p in primitives]
    out = ['/* this file is auto-generated by setup.py */']
    for p in prims:
        out.append('extern RenderPrimitiveInterface primitive_{0};'.format(p))
    out.append('static RenderPrimitiveInterface *render_primitives[] = {')
    for p in prims:
        out.append('    &primitive_{0},'.format(p))
    out.append('    NULL')
    out.append('};\n')
    with open(SRC_PATH + 'primitives.h', 'w') as f:
        f.write('\n'.join(out))


class CustomBuild(build):

    def run(self):
        generate_version_py()
        generate_primitives_h()
        try:
            build.run(self)
            print('\nBuild Complete')
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
        elif c == "unix":
            # customize the build options for this compilier
            for e in self.extensions:
                e.extra_compile_args.append("-Wno-unused-variable") # quell some annoying warnings
                e.extra_compile_args.append("-Wno-unused-function") # quell some annoying warnings
                e.extra_compile_args.append("-Wdeclaration-after-statement")
                e.extra_compile_args.append("-Werror=declaration-after-statement")
                e.extra_compile_args.append("-O3")


        # build in place, and in the build/ tree
        self.inplace = False
        build_ext.build_extensions(self)

        self.inplace = True
        build_ext.build_extensions(self)


class CustomClean(clean):
    """Custom clean command to remove in-place extension and the version file,
    primitives header

    """
    def run(self):
        clean.run(self)
        # try to remove '_composite.{so,pyd,...}' extension,
        # regardless of the current system's extension name convention
        build_ext = self.get_finalized_command('build_ext')
        ext_fname = build_ext.get_ext_filename('overviewer_core.c_overviewer')
        primspath = os.path.join('overviewer_core', 'src', 'primitives.h')

        for fname in [ext_fname, primspath]:
            if os.path.exists(fname):
                try:
                    log.info("removing '%s'", fname)
                    if not self.dry_run:
                        os.remove(fname)
                except OSError as error:
                    log.warn("'%s' could not be cleaned: %s", fname, error)
            else:
                log.debug("'%s' does not exist -- can't clean it", fname)

        # now try to purge all *.pyc files
        for root, dirs, files in \
                os.walk(os.path.join(os.path.dirname(__file__), '.')):
            for f in files:
                if f.endswith('.pyc'):
                    if self.dry_run:
                        log.warn('Would remove %s', os.path.join(root, f))
                    else:
                        os.remove(os.path.join(root, f))


class CustomSDist(sdist):
    def run(self):
        generate_version_py()
        generate_primitives_h()
        sdist.run(self)


setup_kwargs['cmdclass']['clean'] = CustomClean
setup_kwargs['cmdclass']['sdist'] = CustomSDist
setup_kwargs['cmdclass']['build'] = CustomBuild
setup_kwargs['cmdclass']['build_ext'] = CustomBuildExt

setup(**setup_kwargs)
