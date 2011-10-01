======================
Running the Overviewer
======================

Rendering your First Map
========================

Overviewer is a command-line application, and so it needs to be run from the command line. If you installed Overviewer from a package manager, the command is ``overviewer.py``. If you downloaded it manually, open a terminal window and navigate to wherever you downloaded Overviewer. For pre-compiled Windows builds, the command is ``overviewer.exe``. For other systems, it's ``./overviewer.py``.

To generate your map, run::

    overviewer.exe WorldName path\to\output\                 # on windows, or
    ./overviewer.py WorldName path/to/output/                 # on other systems

where ``WorldName`` is the name of the world you want to render, and
``path/to/output`` is the place where you want to store the rendered world. The
first render can take a while, depending on the size of your world. You can, if
you want to, provide a path to the world you want to render, instead of
providing a world name and having Overviewer auto-discover the world path.

When the render is done, open up *index.html* using your web-browser of choice. Pretty cool, huh? You can even upload this map to a web server to share with others! Simply upload the entire folder to a web server and point your users to index.html!

Incremental updates are just as easy, and a lot faster. If you go and change something inside your world, run the command again and Overviewer will automatically rerender only what's needed.

