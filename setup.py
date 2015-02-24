#!/usr/bin/env python

from setuptools import setup, find_packages, Extension

from distutils.command.build import build
from distutils.command.clean import clean
from distutils.command.build_ext import build_ext
from distutils.command.sdist import sdist
from distutils import log
import sys, os, os.path
import platform
import time
import overviewer.util as util

# TODO error out if cython not here
try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    cythonize = lambda x: x
    USE_CYTHON = False

# make sure our current working directory is the same directory
# setup.py is in
curdir = os.path.split(sys.argv[0])[0]
if curdir:
    os.chdir(curdir)

# now, setup the keyword arguments for setup
setup_kwargs = {}
setup_kwargs['ext_modules'] = []
setup_kwargs['cmdclass'] = {}
setup_kwargs['options'] = {}

#
# metadata
#

setup_kwargs['name'] = 'Minecraft-Overviewer'
setup_kwargs['version'] = util.findGitVersion()
setup_kwargs['description'] = 'Generates large resolution images of a Minecraft map.'
setup_kwargs['url'] = 'http://overviewer.org/'
setup_kwargs['author'] = 'Andrew Brown'
setup_kwargs['author_email'] = 'brownan@gmail.com'
setup_kwargs['license'] = 'GNU General Public License v3'
setup_kwargs['long_description'] = open('README.rst').read()

# top-level files that should be included as documentation
doc_files = ['COPYING.txt', 'README.rst', 'CONTRIBUTORS.rst', 'sample_config.py']

#
# requires
#

setup_kwargs['install_requires'] = ['Cython >= 0.19']
setup_kwargs['setup_requires'] = ['setuptools_git >= 0.3']

#
# script, package, and data
#

setup_kwargs['packages'] = find_packages()
setup_kwargs['include_package_data'] = True

# helper for extension file names
def make_files(base, pyx_to, ends):
    r = []
    for e in ends:
        if not USE_CYTHON and e.endswith('.pyx'):
            e, _ = e.rsplit('.', 1)
            e += '.' + pyx_to
        r.append(base + e)
    return r

# oil extension
#
# STILL TODO: verify libpng is present
# and libjpeg
#

oil_files = make_files('overviewer/oil/', 'c', [
    "oil.pyx",
    "oil-matrix.c",
    "oil-image.c",
    "oil-format.c",
    "oil-format-png.c",
    "oil-format-jpeg.c",
    "oil-palette.c",
    "oil-dither.c",
    "oil-backend.c",
    "oil-backend-cpu.c",
    "oil-backend-debug.c",
    "oil-backend-cpu-sse.c",
    "oil-backend-opengl.c",
])

oil_includes = make_files('overviewer/oil/', 'h', [
    "oil.h",
    "oil-dither-private.h",
    "oil-format-private.h",
    "oil-image-private.h",
    "oil-palette-private.h",
    "oil-backend-private.h",
    "oil-backend-cpu.def",
])

extra_link_args = []
if "nt" in os.name:
    extra_link_args.append("-Wl,--export-all-symbols")
setup_kwargs['ext_modules'].append(Extension('overviewer.oil', oil_files, depends=oil_includes, libraries=['png', 'z', 'jpeg'], extra_link_args=extra_link_args))

# chunkrenderer extension
chunkrenderer_files = make_files('overviewer/chunkrenderer/', 'c', [
        'chunkrenderer.c',
        'blockdata.c',
])
chunkrenderer_includes = make_files('overviewer/chunkrenderer/', 'h', [
        'buffer.h',
        'chunkrenderer.h',
])

# todo: better oil handling!
output_name = 'overviewer/oil.so'
if "nt" in os.name:
    output_name = 'overviewer/oil.pyd'
#setup_kwargs['ext_modules'].append(Extension('overviewer.chunkrenderer', chunkrenderer_files, include_dirs=['overviewer/oil'], depends=chunkrenderer_includes, extra_objects=[output_name]))

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
        print("WARNING: failed to build overviewer_version file")

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
        print("\nBuild Complete")

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
            if platform.system() == "Darwin":
                # we want to build dynamic libs, not bundles:
                if "-bundle" in self.compiler.linker_so:
                    self.compiler.linker_so.remove("-bundle")
                self.compiler.linker_so.append("-dynamiclib")
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

setup_kwargs['ext_modules'] = cythonize(setup_kwargs['ext_modules'])

setup(**setup_kwargs)

