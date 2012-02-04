===================================
Building the Overviewer from Source
===================================

These instructions are for building the C extension for Overviewer. Once you
have finished with these instructions, head to :doc:`running`.

.. note::

    Pre-built Windows and Debian executables are available on the
    :doc:`installing` page.  These kits already contain the compiled code and
    require no further setup, so you can skip to the next section of the docs:
    :doc:`running`.

Get The Source
==============

First step: download the platform-independent source! Either clone with Git
(recommended if you know Git) or download the most recent snapshot

* Git URL to clone: ``git://github.com/overviewer/Minecraft-Overviewer.git``
* `Download most recent tar archive <https://github.com/overviewer/Minecraft-Overviewer/tarball/master>`_

* `Download most recent zip archive <https://github.com/overviewer/Minecraft-Overviewer/zipball/master>`_

Once you have the source, see below for instructions on building for your
system.

Build Instructions For Various Operating Systems
================================================

.. contents::
    :local:

Windows Build Instructions
--------------------------

First, you'll need a compiler.  You can either use Visual Studio, or
cygwin/mingw. The free `Visual Studio Express
<http://www.microsoft.com/express/Windows/>`_ is okay. You will want the C++
version (Microsoft® Visual C++® 2010 Express).  Note that the Express version of
Visual Studio will only build 32-bit executables.  We currently don't have a
recommended way of building Overviewer on 64-bit Windows using free tools.  If you
have bought a copy of Visual Studio, you can use it for 64-bit builds.


Prerequisites
~~~~~~~~~~~~~

You will need a copy of the `PIL sources <http://www.pythonware.com/products/pil/>`_.

Building with Visual Studio
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Get the latest Overviewer source code as per above.
2. From the Start menu, navigate to the 'Microsoft Visual Studio 2010 Express' and open the 'Visual Studio Command Prompt (2010)' shortcut.
3. cd to the folder containing the Overviewer source code.
4. Copy Imaging.h and ImPlatform.h from your PIL installation into the current working directory.
5. First try a build::

    c:\python26\python setup.py build

If you encounter the following errors::

    error: Unable to find vcvarsall.bat

then try the following::

    set DISTUTILS_USE_SDK=1
    set MSSdk=1
    c:\python26\python setup.py build

If the build was successful, there should be a c_overviewer.pyd file in your current working directory.

Building with mingw
~~~~~~~~~~~~~~~~~~~

1. Open a MinGW shell.
2. cd to the Overviewer directory.
3. Copy Imaging.h and ImPlatform.h from your PIL installation into the current working directory.
4. Build::

    python setup.py build --compiler=mingw32


Linux
-----

You will need the gcc compiler and a working build environment. On Ubuntu and
Debian, this can be done by installing the ``build-essential`` package. For
CentOS machines, see the :ref:`centos` section below

You will need the following packages (at least):

* python-imaging (for PIL)
* python-dev
* python-numpy

Then to build::

    python setup.py build

OSX
---

1. Download the source code for PIL from http://www.pythonware.com/products/pil/
2. Compile the PIL code (``python ./setup.py build``)
3. Install PIL (``sudo python ./setup.py install``)
4. Find the path to libImaging in the PIL source tree
5. Build Minecraft Overviewer with the path from step 3 as the value for C_INCLUDE_PATH::

    C_INCLUDE_PATH="path from step 3" python ./setup.py build

The following script (copied into your MCO source directory) should handle everything for you:

.. code-block:: bash

    #!/bin/bash

    # start with a clean place to work
    python ./setup.py clean

    # get PIL
    if [ ! -d "`pwd`/Imaging-1.1.7/libImaging" ]; then
        /usr/bin/curl -o imaging.tgz http://effbot.org/media/downloads/Imaging-1.1.7.tar.gz
        tar xzf imaging.tgz
        rm imaging.tgz
    fi

    # build MCO
    C_INCLUDE_PATH="`pwd`/Imaging-1.1.7/libImaging" python ./setup.py build

FreeBSD
-------
FreeBSD is similar to OSX and Linux, but ensure you're using Python 2.7. The port of Python 2.6 has bugs with threading under FreeBSD.
Everything else you should need is ported, in particular math/py-numpy and graphics/py-imaging.

You may need or want to add the line::

    PYTHON_VERSION=2.7

to the file /etc/make.conf, but read the ports documentation to be sure of what this might do to other Python applications on your system.

.. _centos:

CentOS
------
Since CentOS has an older version of Python (2.4), there are some difficulties
in getting the Overviewer to work. Follow these steps which have been reported
to work.

Note: commands prefixed with a "#" mean to run as root, and "$" mean to run as a
regular user.

1. Install the `EPEL repo <http://fedoraproject.org/wiki/EPEL>`_. Go to step #2 if you already have the EPEL repo installed.

  1. ``$ wget http://download.fedoraproject.org/pub/epel/5/i386/epel-release-5-4.noarch.rpm``
  2. ``# rpm -Uhv epel-release-5-4.noarch.rpm``

2. Install the python26 packages and build dependancies

  1. ``# yum install -y python26{,-imaging,-numpy}{,-devel} gcc``

3. Install and setup Overviewer

  1. ``$ git clone git://github.com/overviewer/Minecraft-Overviewer.git``
  2. ``$ cd Minecraft-Overviewer``
  3. ``$ python26 setup.py build``
  4. Change the first line of overviewer.py from ``#!/usr/bin/env python`` to ``#!/usr/bin/env python26`` so that the Python 2.6 interpreter is used instead of the default 2.4

4. Run Overviewer as usual

  1. ``$ ./overviewer.py path/to/world/ path/to/output/`` or ``$ python26 path/to/overviewer.py path/to/world/ path/to/output/``
  2. Proceed to the :doc:`Running <running>` instructions for more info.


Installing the Compiled Code
----------------------------

You can run the ``overviewer.py`` script from the build directory just fine;
installation is unnecessary. If you'd like to install, run

::

    python setup.py install
