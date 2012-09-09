#!/usr/bin/env python

from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
from distutils.command.clean import clean
from distutils import log

import os.path

class CustomBuildExt(build_ext):
    def build_extensions(self):
        c = self.compiler.compiler_type
        if c == "msvc":
            # customize the build options for this compilier
            for e in self.extensions:
                e.extra_link_args.append("/MANIFEST")
        if c == "unix":
            # customize the build options for this compilier
            for e in self.extensions:
                e.extra_compile_args.append("-Wdeclaration-after-statement")
                e.extra_compile_args.append("-Wall")
                e.extra_compile_args.append("-Werror")
        
        # build in place, and in the build/ tree
        self.inplace = False
        build_ext.build_extensions(self)
        self.inplace = True
        build_ext.build_extensions(self)

class CustomClean(clean):
    def run(self):
        # do the normal cleanup
        clean.run(self)

        # try to remove 'OIL.{so,pyd,...}' extension,
        # regardless of the current system's extension name convention
        build_ext = self.get_finalized_command('build_ext')
        fname = build_ext.get_ext_filename('OIL')
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

#
# STILL TODO: verify libpng is present
#

oil_headers = [
    "oil.h",
    "oil-dither-private.h",
    "oil-format-private.h",
    "oil-image-private.h",
    "oil-palette-private.h",
]

oil_sources = [
    "oil-python.c",
    "oil-image.c",
    "oil-format.c",
    "oil-format-png.c",
    "oil-palette.c",
    "oil-dither.c",
]

setup(name='OIL',
      version='0.0-git',
      cmdclass={'build_ext' : CustomBuildExt, 'clean' : CustomClean},
      ext_modules=[Extension('OIL', oil_sources, depends=oil_headers, libraries=['png'])],
)
