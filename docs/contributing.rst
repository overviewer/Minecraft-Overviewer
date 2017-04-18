============
Contributing
============

In this page, you'll be given some pointers on how to start contributing to the
Minecraft-Overviewer project. This is useful for people who want to help develop
the Overviewer, but don't quite know where to start.

This page is mostly focused on where to look for things and how to get your
changes back into the project, for help on how to compile the Overviewer, check
:doc:`Building <building>`.

Prerequisites
=============

Ideally you're familiar with Python (Overviewer uses Python 2), and know the
basics of Git. Both have various very good resources online that help you in
learning them, but the best way of learning is always to use them in the real
world, so don't hesitate to jump right in after having a basic grasp and ask
questions along the way.

Additionally, some parts of Overviewer are written in C, though unless you're
interested in the drawing and compositing routines or the rendermodes, you don't
need to know C.

Last but not least, some of the Overviewer's code is written in JavaScript,
namely the part that runs in your browser when you view the map.

Acquiring the Source Code
=========================

First, you'll need to get the Overviewer source code. We do version management
of code through Git_, which allows multiple people to work on the code at the
same time. Naturally, this means you'll also be getting the source code through
Git. For this to work, you'll have to install Git on your computer.

.. _Git: https://git-scm.com/

Our source code is hosted on GitHub_, so it's a good idea to make an account
there if you don't already have one.

.. _GitHub: https://github.com/overviewer/Minecraft-Overviewer

This page won't go into the details of how to use Git, but it'll give you some
advice on how your workflow should be to avoid some trouble.

Finding Your Way around the Code Base
=====================================

At first glance, all the code can be a bit overwhelming. So here's a quick
overview of the important parts.

* ``setup.py`` is the build script. If you need to make any changes to how the
  Overviewer is built, you'll want to look there.

* ``overviewer.py`` is the entry-point of the application. It imports all the
  other functionality, and does the command line parsing.

* ``overviewer_core/`` is the directory where the vast majority of the
  Overviewer's functionality is. More on that below.

* ``overviewer_core/aux_files/genPOI.py`` is where the genPOI functionality is
  implemented. If you're looking into changing the way markers are generated,
  look there.

* ``overviewer_core/src/`` is the directory for all the files that are part of
  Overviewer's C extension. This includes things such as rendermodes, which are
  stored in the ``primitives`` sub-directory.

* ``overviewer_core/data/`` mostly contains the parts that make up Overviewer's
  web front-end, with ``js_src`` containing the JS files and ``web_assets``
  containing the ``index.html``, CSS files and image files such as icons or the
  compass.

* ``docs/`` contains the documentation, which can be built with the included
  Makefile if you have sphinx installed.

overviewer_core
---------------

Let's take a closer look at the ``overviewer_core/`` directory:

* ``assetmanager.py`` controls how the HTML and JS output are written out, as
  well as the ``overviewerConfig.js`` format.

* ``cache.py`` implements a Least-Recently-Used (LRU) cache, which is used for
  caching chunks in memory as the rendering happens.

* ``configParser.py`` contains some code that sets up how the config is parsed,
  but is not really involved in the definitions of individual settings therein.

* ``dispatcher.py`` is the code that sets up multiprocessing, so Overviewer can
  use all available CPU threads on a machine.

* ``files.py`` implements helpful routines which allow you to determine whether
  some file operations such as replacing a file work in a given directory, and
  also implements the ``FileReplacer`` class which can then safely replace a
  file given the capabilities of the filesystem.

* ``items.py`` is a remnant of the past and entirely unused.

* ``logger.py`` sets up and implements Overviewer's logging facilities.

* ``nbt.py`` contains the code that is used to parse the Minecraft NBT file
  structure.

* ``observer.py`` defines all the observers that are available. If you want to
  add a new observer, this is the place where you'll want to look.

* ``optimizeimages.py`` defines all the optimizeimg tools and how they're
  called.

* ``progressbar.py`` implements the fancy progress bar that the Overviewer has.

* ``rcon.py`` implements an rcon client for the Minecraft server, used by the
  RConObserver.

* ``rendermodes.py`` contains definitions and glue code for the rendermodes in
  the C extension.

* ``settingsDefinitions.py`` includes all definitions for the Overviewer
  configuration file. If you want to add a new configuration option, this is
  where you'll want to start.

* ``settingsValidators.py`` contains validation code for the settings
  definitions, which ensures that the values are all good.

* ``signals.py`` is multiprocessing communication code. Scary stuff.

* ``textures.py`` contains all the block definitions and how Overviewer should
  render them. If you want to add a new block to the Overviewer, this is where
  you'll want to do it. Additionally, this code also controls how the textures
  are loaded.

* ``tileset.py`` contains code that maps a render dict entry to the output tiled
  image structure.

* ``util.py`` contains random utility code that has no home anywhere else.

* ``world.py`` is a whole lot of code that does things like choosing which
  chunks to load and to cache, and general functionality revolving around the
  concept of Minecraft worlds.

docs
----

The documentation is written in reStructuredText_, a markup format. It can be
compiled into an HTML output using the Makefile in the ``docs/`` subtree by
typing ``make``. You'll need to have sphinx_ installed for this to work.

.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _sphinx: http://www.sphinx-doc.org/en/stable/

The theme that will be used in the locally generated HTML is different than what
is used on http://docs.overviewer.org. However, it should still be sufficient
to get a good idea of how your changes will end up looking like when they're on
the main docs page.

Code Style
==========

To be honest, currently the Overviewer's codebase is a bit of a mess. There is
no consistent code style in use right now. However, it's probably a good idea
to stick to PEP8_ when writing new code. If you're refactoring old code, it
would be great if you were to fix it to make it PEP8 compliant as well.

To check whether the code is PEP8 compliant, you can use pycodestyle_. You can
easily install it with pip by using ``pip2 install pycodestyle``.

.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _pycodestyle: https://pypi.python.org/pypi/pycodestyle


Example Scenarios
=================

This section will demonstrate by example how a few possible contributions might
be made. These serve as guidelines on how to quickly get started if you're
interested in doing a specific task that many others before you have done too
in some other form.

Adding a Block
--------------

Let's assume you want to add support for a new block to the Overviewer. This is
probably one of the most common ways people start contributing to the project,
as all blocks in the Overviewer are currently hardcoded and code to handle them
needs to be added by hand.

The place to look here is ``textures.py``. It contains the block definitions,
which are assisted by Python decorators_, which make it quite a bit simpler to
add new blocks.

The big decorator in question is ``@material``, which takes arguments such as
the ``blockid`` (a list of block IDs this block definition should handle), and
``data`` (a list of possible data values for this block). Additionally, it can
also take various additional arguments for the different block properties, such
as ``solid=True`` to indicate that the block is a solid block.

.. _decorators: https://en.wikipedia.org/wiki/Python_syntax_and_semantics#Decorators

Simple Solid 6-Sided Block
~~~~~~~~~~~~~~~~~~~~~~~~~~

A lot of times, new blocks are basically just your standard full-height block
with a new texture. For a block this simple, we don't even really need to use
the material decorator. As an example, check out the definition of the coal
block::

    block(blockid=173, top_image="assets/minecraft/textures/blocks/coal_block.png")

Block with a Different Top
~~~~~~~~~~~~~~~~~~~~~~~~~~

Another common theme is a block where the top is a different texture than the
sides. Here we use the ``@material`` decorator to create the jukebox block::

    @material(blockid=84, data=range(16), solid=True)
    def jukebox(self, blockid, data):
        return self.build_block(self.load_image_texture("assets/minecraft/textures/blocks/jukebox_top.png"), self.load_image_texture("assets/minecraft/textures/blocks/noteblock.png"))

As you can see, we define a method called ``jukebox``, taking the parameters
``blockid`` and ``data``, decorated by a decorator stating that the following
definition is a material with a ``blockid`` of ``84`` and a data value range
from ``0`` to ``15`` (or ``range(16)``), which we won't use as it doesn't affect
the rendering of the block. We also specify that the block is solid.

Inside the method, we then return the return value of ``self.build_block()``,
which is a helper method that takes a texture for the top and a texture for the
side as its arguments.

Block with Variable Colors
~~~~~~~~~~~~~~~~~~~~~~~~~~

Occasionally, blocks can have colors stored in their data values.
``textures.py`` includes an easy mapping list, called ``color_map``, to map
between data values and Minecraft color names. Let's take stained hardened clay
as an example of how this is used::

    @material(blockid=159, data=range(16), solid=True)
    def stained_clay(self, blockid, data):
        texture = self.load_image_texture("assets/minecraft/textures/blocks/hardened_clay_stained_%s.png" % color_map[data])

        return self.build_block(texture,texture)

As you can see, we specify that the block has 16 data values, then depending
on the data value we load the right block texture by looking up the color name
in the ``color_map`` list, formatting a string for the filename with it.

Good Git Practices
==================

How you structure your Git workflow is ultimately up to you, but here are a few
recommendations to make your life and the life of the people who want to merge
your pull requests easier.

* **Commit your changes in a separate branch, and then submit a pull request
  from that branch.** This makes it easier for you to rebase your changes, and
  allows you to keep your repository's master branch in-sync with our master
  branch, so you can easily split off a new branch from master if you want to
  develop a new change while your old change still isn't merged into the master.

* **Format your commit messages properly.** The first line should be a 50
  character long summary of the change the commit makes, in present tense, e.g.
  "Add a spinner to the progress bar". This should be followed by a blank line,
  and a longer explanation of the change the commit actually does, wrapped at
  72 characters.

* **Don't merge master into your branch.** If you plan on submitting a change as
  a pull request and the master branch has moved in the meantime, then don't
  merge the master branch into the branch of your pull request. Instead, rebase
  your branch on top of the updated master.

* **Keep commits logically separated.** Don't try to cram unrelated changes into
  just one commit unless it's a commit full of small fixes. If you find yourself
  struggling to keep the commit summary below 50 characters, and find yourself
  using the word "and" in it, rethink whether the changes you're making should
  be just one commit.

It's also a good idea to look at the output of ``git diff`` before committing a
change, to make sure nothing was unintentionally changed in the file where you
weren't expecting it. ``git diff`` will also highlight blank lines with spaces
in them with a solid red background.

Talking with other Developers
=============================

Occasionally, the issue tracker simply doesn't cut it. You need to talk with
another developer, maybe to brainstorm a new feature or ask a question about
the code. For this, we have `an IRC channel on freenode`_, which allows you to
talk with other developers that are on the IRC channel in real-time.

.. _an IRC channel on freenode: https://overviewer.org/irc/

Since most developers have jobs or are in college or university, it may
sometimes take a few moments to get a reply. So it's useful to stick around and
wait for someone who can help you to be around.
