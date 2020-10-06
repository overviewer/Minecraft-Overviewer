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
(recommended if you know Git) or download the most recent snapshot:

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
cygwin/mingw. The free `Visual Studio Community
<https://www.visualstudio.com/vs/community/>`_ is okay. You will need to select the "Desktop Development with C++" WORKLOAD. Microsoft has been changing up the names on this with the "Community" edition of Visual Studio. If nothing else works, just install every Individual Visual C++ component you can find :)


Prerequisites
~~~~~~~~~~~~~

You will need the following:

- `Python 3.x <https://www.python.org/downloads/windows/>`_
- A copy of the `Pillow sources <https://github.com/python-pillow/Pillow>`_.
- The Pillow Extension for Python.
- The Numpy Extension for Python.
- The extensions can be installed via::

    c:\python37\python.exe -m pip -U numpy pillow


Building with Visual Studio
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Get the latest Overviewer source code as per above.
2. From the Start menu, navigate to 'Visual Studio 2017' and open the **'Developer Command Prompt for VS 2017'** (*or whatever year*) shortcut. A regular command or powershell prompt will *NOT* work for this.
3. cd to the folder containing the Overviewer source code.
4. Copy Imaging.h and ImPlatform.h from your Pillow sources into the current working directory.
5. First try a build::

    c:\python37\python setup.py build

If you encounter the following errors::

    error: Unable to find vcvarsall.bat

then try the following::

    set DISTUTILS_USE_SDK=1
    set MSSdk=1
    c:\python37\python setup.py build

If the build was successful, there should be a c_overviewer.pyd file in your current working directory.

Building with mingw-w64 and msys2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the recommended way to build on Windows without MSVC.

1. Install msys2 by following **all** the instructions on 
   `the msys2 installation page <https://msys2.github.io/>`_.

2. Install the dependencies::

    pacman -S git mingw-w64-x86_64-python3-numpy mingw-w64-x86_64-python3-Pillow mingw-w64-x86_64-python3 mingw-w64-x86_64-toolchain

3. Clone the Minecraft-Overviewer git repository::

    git clone https://github.com/overviewer/Minecraft-Overviewer.git

   The source code will be downloaded to your msys2 home directory, e.g.
   ``C:\msys2\home\Potato\Minecraft-Overviewer``

4. Close the msys2 shell. Instead, open the MinGW64 shell.

5. Build the Overviewer by changing your current working directory into the source
   directory and executing the build script::

    cd Minecraft-Overviewer
    python3 setup.py build

After it finishes, you should now be able to execute ``overviewer.py`` from the MINGW64
shell.

Building with mingw
~~~~~~~~~~~~~~~~~~~

1. Open a MinGW shell.
2. cd to the Overviewer directory.
3. Copy Imaging.h and ImPlatform.h from your Pillow sources into the current working directory.
4. Build::

    python3 setup.py build --compiler=mingw32
    
If the build fails with complaints about ``-mno-cygwin``, open the file ``Lib/distutils/cygwincompiler.py``
in an editor of your choice, and remove all mentions of ``-mno-cygwin``. This is a bug in distutils,
filed as `Issue 12641 <http://bugs.python.org/issue12641>`_. 


Linux
-----

You will need the gcc compiler and a working build environment. On Ubuntu and
Debian, this can be done by installing the ``build-essential`` package.

You will need the following packages on Debian-derived distributions (e.g. Ubuntu):

* python3-pil or python3-pillow (the latter is usually aliased to the former)
* python3-dev
* python3-numpy

.. note::
    If you choose to install pillow through pip instead of your distribution's package
    manager, you won't get the pillow headers which Overviewer requires to build its C
    extension. In that case, you should manually download the header files specific to
    the version of pillow you installed, and point at them with the ``PIL_INCLUDE_DIR``
    environment variable. A version mismatch between the installed pillow library and
    the headers can lead to segfaults while running Overviewer due to an ABI mismatch.

Then to build::

    python3 setup.py build
    
At this point, you can run ``./overviewer.py`` from the current directory, so to run it you'll have to be in this directory and run ``./overviewer.py`` or provide the the full path to ``overviewer.py``.  Another option would be to add this directory to your ``$PATH``.   Note that there is a ``python3 setup.py install`` step that you can run which will install things into ``/usr/local/bin``, but this is strongly not recommended as it might conflict with other installs of Overviewer.

macOS
-----

#. Install the Xcode Command Line Tools by running the following command in a terminal (located in your /Applications/Utilities folder)::

    xcode-select --install

#. Install Python 3 if you don't already have it, for example from `the official Python website <https://www.python.org/downloads/mac-osx/>`_.
#. Install PIP, e.g. with::

    sudo easy_install pip

#. Install Pillow (overviewer needs PIL, Pillow is a fork of PIL that provides the same functionality)::

    pip install Pillow

#. Download the Pillow source files from https://github.com/python-pillow/Pillow/releases/latest and unpack the tar.gz file and move it to a directory you can remember
#. Download the Minercaft Overviewer source-code from https://overviewer.org/builds/overviewer-latest.tar.gz
#. Extract overviewer-[Version].tar.gz and move it to a directory you can remember
#. Go into your Pillow-[Version] folder and navigate to the /src/libImaging directory
#. Drag the following files from the Pillow-[Version]/src/libImaging folder to your overviewer-[Version] folder:
  - ``Imaging.h``
  - ``ImagingUtils.h``
  - ``ImPlatform.h``
#. Make sure your installation of Python 3 is in ``$PATH``
#. In a terminal, change your current working directory to your overviewer-[Version] folder (e.g. by using ``cd Desktop/overviewer-[Version]``)
#. Build::

    python3 setup.py build

You should now be able to run Overviewer with ``./overviewer.py`` inside of the
Overviewer directory.
