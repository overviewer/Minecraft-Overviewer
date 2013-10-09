===============================
Overviewer V2 (with oil) Design
===============================


.. warning::

    The `oil` branch is still experimental and in constant flux. Do not expect
    these documents to be completely accurate. Until the branch stabilizes,
    these docs are merely notes.

OIL Overview
============

OIL is the Overviewer Imaging Library. It is also a nickname for the sweeping
changes and generalizations to The Overviewer that have been kicking around for
some time.

At its core `oil` is a tiny 3D engine. It draws texture-mapped triangles. This
is aimed to replace the old style sprite-renderer of the original Overviewer.
Before, Overviewer would take Minecraft textures, transform them with a
manually-coded sequence of transformations (one for each block type) into a
block sprite, and paste these block sprites on the tiles.

This was reasonably fast, but had some limitations:

* It's cumbersome to add new block types, especially for complex models that
  aren't just cubes. It requires someone figuring out a sequence of
  transformations to take textures and turn them into blocks at just the right
  angle. Wouldn't it be great to import a 3-D model in a standard format and
  have Overviewer figure out how to render it itself?

  Being able to add blocks easily is especially important for supporting
  Minecraft mods that add new blocks.

* It's not very general. Every block sprite is hard coded to display at a
  particular angle and size. If you wanted a lower or higher angle, more zoomed
  in, less zoomed in, you were out of luck.

* Image format limitations: using our own image library instead of PIL is
  important since we can then support indexed PNGs, drastically reducing the
  size of the output images. PIL doesn't support this and probably never will.

* PIL and numpy as dependencies were never all that justified. PIL was only ever
  used to read and write the PNG format (and as mentioned above it's not very
  good at that). Numpy was used for the integer arrays representing the raw data
  from Minecraft worlds, and for a few matrix operations. Both are fairly large
  dependencies that don't have much justification in our cases. If we can get
  rid of those dependencies, not only will Overviewer be easier to install, but
  we may get better compatibility with other Python implementations such as
  PyPy, or make the transition to Python 3 easier.

Big moving Parts
----------------

Here are the main parts of the new oil-based overviewer to be familiar with. Not
all of this is done yet.

* A generalized renderer system. This lets us abstract the implementation of *how* to
  render something from the implementation of the output format, i.e. a quadtree
  of tiles. See `Renderer Objects`_

* Specifically, the :class:`.IsometricRenderer` abstraction lets you render from
  any angle in an isometric perspective. This is a new layer of abstraction that
  sits between the :class:`.TileSet` object and the :class:`.RegionSet` object,
  and determines what Minecraft chunks should be drawn where depending on the
  angle specified. It is supported by a C backend to do the actual rendering.

* A more flexible way to define blocks. Now we can specify a set of verticies
  and faces in 3-D space to draw blocks, instead of having to figure out a
  manual set of operations for transforming textures into block sprites. See
  `Block definitions`_.

* A completely re-done rendering engine written in C based on the new imaging
  library.

* A new configuration format to reflect the additional flexibility and available
  options.

What hasn't changed? Here's what we don't plan on changing.

* The quadtree format. We're still going to render a quadtree of tiles. The leaf
  tiles in the tree are rendered from block and world data. The inner nodes of
  the tree are composed of the children by zooming them out. (although making a
  new :class:`.Canvas` object for a format of your choice is no longer a messy
  and complicated task)



Walkthrough
===========

Summary
-------

Here are the high level steps and components of a render:

1. First create a :class:`.RegionSet` object to represent the minecraft
   dimension to render. This can be created by first creating a :class:`.World`
   object, which has methods to scan for and create RegionSet objects of that
   world.

2. Then create a :class:`.IsometricRenderer` object, giving it the RegionSet
   from step 1.

3. Then create a :class:`.TileSet` object, giving it the renderer object from
   step 2.

4. Then create a :class:`.Dispatcher` object, or a
   :class:`.MultiprocessorDispatcher` for multiprocessor rendering.

5. Add the TileSet object from step 3 to the Dispatcher as a worker. The
   dispatcher will tell the TileSet to do its work.

.. warning::

    You must wrap the RegionSet in a CachedRegionSet object or rendering will be
    exceedingly slow. We really ought to make caching non-optional.

.. note::

    An AssetMananger, Textures, BlockDefinitions objects are also required, and
    so the above steps are not complete and true to the current code. 
    I want to make sensible defaults to simplify the above workflow; the above
    is a goal.

Details
-------

This section attempts to detail the workflow of a typical Overviewer render, with cross references to other sections. Not everything in here will make perfect sense without additional knowledge of the different classes, their purposes, and their interactions, but this is a good place to start.

#. Processing starts in the main() method.

#. The configuration is processed.

#. A :class:`.World` object is created.

#. The :class:`.World` object scans the directory for dimensions and creates a :class:`.RegionSet` object for each one.

#. The :class:`.RegionSet` constructor creates a map of region x,z coords to regionfile names.

#. Next the main() method creates a :class:`.Textures` object with the current texture options, or gets the default texture options from :func:`.textures.get_default`, which finds a Minecraft.jar and uses it as the Textures object constructor parameter.

#. The :class:`.World` object has its :meth:`.World.get_regionset` method called.

#. RegionSet caching is applied: the RegionSet object is wrapped in a :class:`.CachedRegionSet` object.

#. The following happens for each render requested:

   #. Any region transformations are applied

   #. A function called :func:`.blockdefinitions.get_default` is called. See the section on `Block Definitions`_.

   #. A :class:`.Matrix` object is created and transformed in a specific way such as to define the render's perspective and projection. See the section on `Matrix Objects`_.

   #. An :class:`.IsometricRenderer` object is created using the :class:`.RegionSet` object, the :class:`.Textures` object, the :class:`.BlockDefinitions` object, and the above matrix. See the `Renderer Objects`_ section.

   #. A :class:`.TileSet` object (a kind of :class:`.Worker` and :class:`.Canvas`) is created. Parameters to the constructor include the :class:`.World` object, the :class:`.RegionSet` object, the asset manager, the tileset options, the :class:`.IsometricRenderer` object, and the output directory.

   #. The :class:`.TileSet` object is appended to a list of all tilesets for this job.

#. For each :class:`.TileSet` object, its :meth:`.TileSet.do_preprocessing` method is called

   * This checks if the tiles should be arranged if the number of zoom levels has changed

   * Then it performs a chunk scan, which builds a set of "dirty tiles" that need to be rendered. See the `Chunk Scanning`_ section.

#. Next the asset manager object is initialized. Passed in is the list of tileset objects. All this does is create and write out the assets (which is done again at the end), so that one can potentially view the map being built in progress.

#. Next a :class:`.Dispatcher` object is created. See the section on `Dispatchers and Workers`_.

#. The dispatcher is started with the list of workers (tileset) objects.

#. The dispatcher gets a list of "work items" from the tileset's :meth:`.TileSet.iterate_work_items` method, and passes them to the tileset's :meth:`.TileSet.do_work` method.

#. In the tileset's do_work method:
   
   * the work item parameter is a tuple of integers in the range 0-3 representing a path in the quad tree for which tile to render.

   * if the length of the path is equal to the tree depth, then we render the tile. Otherwise, the tile is composed of its 4 child tiles.

#. For render tiles:

   #. :meth:`.TileSet._render_rendertile` is called after making a :class:`.RenderTile` object from the path

   #. The path on the filesystem to save the tile is determined

   #. The directory is created if it doesn't exist

   #. The tileset calls :meth:`.IsometricRenderer.get_render_sources_in_rect` which returns all chunks that are required to render the given area of the canvas. This is used along with :meth:`.IsometricRenderer.get_render_source_mtime` to determine the maximum mtime of this rect, which is later used to set the mtime of the tile on disk.

   #. An oil :class:`.Image` object is created

   #. :meth:`.IsometricRenderer.render` is called.

    #. This method calls :meth:`.IsometricRenderer._get_tile_chunks` to get a list of all chunks that apply to the given tile

    #. A projection matrix is set up for this tile

    #. For every chunk from the above call, exactly where this chunk is to be rendered on the tile is calculated, and a new translated matrix is computed.

    #. Then the C code is entered: :func:`.chunkrenderer.render` is called, given the :class:`.RegionSet` object, the chunk coordinates, the :class:`Image` object, the matrix, and the compiled block definitions.


Block Definitions
=================
A block definition defines how to render a particular Minecraft block. There are two relevant objects defined in the :mod:`blockdefinitions` module.

.. module:: overviewer.blockdefinitions

.. class:: BlockDefinitions

    This is simply a container for many :class:`BlockDefinition` objects. Block
    definitions are added to this object using the :func:`add` method. Once all
    blocks are defined, this object is passed to the
    :func:`.compile_block_definitions` function.

    .. method:: add(blockdef, blockid, data=0)

        Adds a block definition object

        :param blockdef: the block definition to add
        :type blockdef: :class:`BlockDefinition`
        :param blockid: the block ID that renders this block
        :type blockid: int or iterable of ints
        :param data: extra data used by the renderer. (I think this has to do
            with block metadata, so different models may be used for different
            situations for the same block? I'm not sure)
        :type data: int or iterable of ints

    .. attribute:: max_blockid

        The maximum of all blockids added with :meth:`add`

    .. attribute:: max_data

        The maximum of all data parameters passed to :meth:`add`

    .. attribute:: blocks

        A dictionary mapping (`blockid`, `data`) to a block definition object.

.. class:: BlockDefinition

    These objects contain the definition for how to draw a particular type of block. It only contains data and is used by the C code to tell how to render something.

    These objects have the following attributes:

    .. attribute:: verticies

        This is a list of (coord, texcoords, color) tuples

        * coord is a 3-tuple of numbers which define this point. The units are world units, so a block typically renders itself between 0,0,0 and 1,1,1, but it's possible to have larger blocks.

        * texcoords is a 2-tuple. It specifies in UV coords where in the texture image maps to this vertex. Specified as numbers between 0 and 1.

        * color is a 4-tuple of 8-bit integers. It applies a multiplicative color mask to this vertex. Faces are colored by an interpolation between its verticies.

    .. attribute:: faces

        This is a list of ([point index list], face_type)

        * [point index list] is a list of integers representing an index into the :attr:`verticies` list.

        * face_type is one of the chunkrenderer.FACE_TYPE_* constants. See `Face Types`_

    .. attribute:: tex

        A texture path

    .. attribute:: transparent

        a boolean indicating to draw this block with transparency enabled

    .. attribute:: solid

        a boolean

    .. attribute:: fluid

        a boolean

    .. attribute:: nospawn

        a boolean

    .. attribute:: nodata

        a boolean

    .. attribute:: triangles

        This is a property, dynamically computed from the :attr:`faces` attribute. It yields ((i,j,k), facetype) tuples where each i,j,k are indicies into the :attr:`verticies` list, together representing a triangle face.

.. function:: get_default

    This function is called from the main() method. It makes a new :class:`BlockDefinitions` object and adds hard-coded definitions for several types of blocks. Specifically, it performs the following actions:

    #. Creates a :class:`BlockDefinitions` object.

    #. For each cube block to add:

        #. Calls :func:`make_simple` giving it the texture path to use

        #. Passes the result to :meth:`Blockdefinitions.add` along with the Minecraft block ID.

.. function:: make_simple

    This function takes a texture path and kwargs, and calls :func:`make_box`. It passes in (0,0,1,1) for each of the texture coord parameters.

.. function:: make_box

    This function constructs a new :class:`BlockDefinition` object. It assumes the object is a cube, and has the same texture on each side. The verticies and faces are hard coded.

Matrix Objects
==============

.. currentmodule:: overviewer.oil

.. class:: Matrix

    This is a 4×4 matrix of floating point numbers, written in C, exposed as a
    Python object. The object is defined in the file oil-python.c, which hold
    the Python binding wrappers around the code in oil-matrix.c, which is code
    adapted from some other GPL matrix implementation.

    .. method:: get_data

        Returns a nested tuple of the matrix data.

    .. method:: transform(x,y,z)

        takes 3 floating point numbers and returns the three floats after
        applying the matrix as a transformation. (computes the dot product of
        the given vector on the matrix as if it were 3×3)

    .. method:: get_inverse

        returns a copy of the matrix, inverted

    .. method:: invert

        inverts this matrix in place. Returns None.

    .. method:: translate(x,y,z)

        takes 3 floats and translates the matrix in place by the given offsets
        on each axis

    .. method:: scale(x,y,z)

        takes 3 floats and scales each axis by the corresponding value.
        Transforms in place.

    .. method:: rotate(x,y,z)

        rotates in place. Takes 3 floats. Represents a rotation in 3D space with
        an amount in radians about the respective axes. This builds a rotation
        matrix and then multiplies it intothe current matrix in place.

    .. method:: orthographic(x1,x2,y1,y2,z1,z2)

        Takes 2 sets of x,y,z floats. Builds a transformation matrix and then
        multiplies it in place to the current matrix. This has to do with
        projection I'm pretty sure but I forget how this stuff works.

    .. attribute:: data

        A nested tuple of the matrix data, same as returned from :meth:`get_data`. You
        can also assign to this property.

    .. attribute:: inverse

        The inverse of this matrix. This is the property version of :meth:`get_inverse`.

    This class also implements standard python arithmetic operators: add,
    subtract, multiply, negative, positive, nonzero, inplace_add,
    inplace_subtract, inplace_multiply.

    * The python type is :c:type:`PyOILMatrix` in oil-python.h
    * The matrix type is :c:type:`OILMatrix` in oil.h
    * The python type definition is PyOILMatrixType in oil-python.c
    * The python bindings are defined in oil-python.c
    * The actual matrix implementation is in oil-matrix.c

C Type
------
.. c:type:: OILMatrix

    This is a struct defined in oil.h. It has a single member:

    .. c:member:: float data[4][4]

        This member defines the 16 floating point values for the matrix

.. c:type:: PyOILMatrix

    This is the python type defined in oil-python.h which holds a
    :c:type:`OILMatrix` and provides a python interface for it.

    .. c:member:: PyObject_HEAD
    .. c:member:: OILMatrix matrix

Renderer Objects
================

.. currentmodule:: overviewer.canvas

.. class:: Renderer

    This class is used in conjunction with a :class:`Canvas` object. Together
    they abstract the concept of drawing one large image on a canvas but having
    it split into tiles.

        The Canvas / Renderer interface lets you implement a Renderer that turns
        arbitrary, opaque "render source" objects into images renderer on a
        virtual canvas. The Renderer informs the Canvas of what render sources
        are available, where they are located, and how big a virtual canvas is
        needed. The Canvas object then uses this information to decide how to
        render this virtual canvas, and provides the Renderer with image objects
        to render to.

    Renderer objects provide the following interface definitions

    .. method:: get_render_sources

        Returns an iterator over all render sources. A render source is an
        opaque object that is passed back to the renderer to decide how to
        proceed with rendering. In Overviewer, render sources are Minecraft
        chunks. Canvas objects are agnostic to this, they only know that a
        particular area on the canvas depends on some number of "render
        sources". The Renderer is aware of what these render sources mean.

    .. method:: get_render_source_mtime(render_source)

        Returns the last modified time for the given render source. Canvas
        objects use this information to do incremental updates and decide what
        sections need rendering.

    .. method:: get_rect_for_render_source(render_source)

        Returns a (xmin, ymin, xmax, ymax) tuple representing the bounding
        rectangle on the virtual canvas that bounds the given render source.

    .. method:: get_render_sources_in_rect(rect)

        Given a bounding rectangle on the virtual canvas, returns all render
        sources that might possibly be contained within it.

        `rect` is (xmin, ymin, xmax, ymax)

    .. method:: get_full_rect

        Returns the bounding box of all render sources. i.e. the entire virtual
        canvas size.

    .. method::  render(origin, im)

        The renderer object is requested to render a portion of the virtual
        canvas. `origin` is a 2-tuple specifying where on the virtual canvas the
        image lies. `im` is the image object (:class:`overviewer.oil.Image` or
        :class:`PIL.Image` ?)

        The renderer is responsible for computing the relevant render sources
        and calling to the underlying renderer machinery to do so.

IsometricRenderer objects
-------------------------

.. module:: overviewer.isometricrenderer

.. class:: IsometricRenderer(Renderer)

    This class is a "renderer implementation that renders :class:`.RegionSet`
    objects in an isometric perspective. It uses the
    :mod:`~overviewer.chunkrenderer` module, a generalized (C-based) 3D chunk
    renderer"

    .. note::

        Not all methods are documented unless the implementation is notable. See
        docs for :class:`.Renderer` for info on what methods are available and what
        they do.

    This object uses Minecraft chunks as render sources. The render source
    object is a 3-tuple: `(x, z, mtime)` where `x` and `z` specify the chunk
    location and `mtime` is the chunk mtime from the region file header.

    .. method:: __init__(regionset, textures, blockdefs, matrix)

        Initialize the renderer with the given parameters

        :param regionset: The regionset object to pull world data from and
            render
        :type regionset: :class:`.RegionSet`
        :param textures: The textures to use in rendering
        :type textures: :class:`.Textures`
        :param blockdefs: The block definitions that define how to render each
            block
        :type blockdefs: :class:`.BlockDefinitions`
        :param matrix: The perspective matrix that defines the rendering angle.
            This is assumed to be isometric.
        :type matrix: :class:`.oil.Matrix`

    .. method:: get_render_sources_in_rect(rect)

        Iterates over :meth:`_get_chunks_in_rect`

    .. method:: _get_chunks_in_rect(rect)

        Does some matrix computations to determine which render sources are
        relevant to this rectangle. Makes calls to
        :meth:`.RegionSet.get_chunk_mtime` to get chunk mtimes.

    .. method:: get_render_source_mtime(render_source)

        Returns `render_source[2]` since render sources are (x, z, mtime)

    .. method:: render(origin, im)

        Makes a call to :meth:`_get_tile_chunks`, does some matrix calculations,
        and calls into :func:`overviewer.chunkrenderer.render` for each relevant chunk.

    .. method:: _get_tile_chunks(origin, im)

        Calculates the bounding rectangle of the given image and origin, and
        makes a call to :meth:`_get_chunks_in_rect`. Chunks are sorted, and a
        (minz, maxz, chunk_list) tuple is returned.

TileSet Objects
===============

.. module:: overviewer.tileset

.. class:: TileSet(Canvas)

    A TileSet object represents a set of output tiles. It takes as a parameter
    to its initializer a :class:`.World` object, a :class:`.RegionSet` object,
    an asset manager, an options dict, the :class:`.Renderer` object, and an
    output directory path.

    The TileSet inherits from a :class:`.canvas.Canvas` object, and so it
    implements the canvas interface and the :class:`.dispatcher.Worker`
    interface. Thus it represents a virtual canvas that is split up into tiles
    and drawn individually. The worker interface means it has work to be done,
    and this class represents each tile as a work item.

    The TileSet is responsible for figuring out how to divide up the virtual
    canvas into tiles, and then giving each tile to the dispatcher. The
    dispatcher figures out when and where to render a tile. This class then uses
    the underlying :class:`.Renderer` object to do the rendering work for each
    tile.

    .. note::

        The main interface to the actual render work is through the
        :class:`.Renderer` object. This class historically used to operate on
        the :class:`.World` object and/or :class:`.RegionSet` object directly.
        With the additional abstraction, this is probably unnecessary, and the
        TileSet object should remain unaware of the underlying Regionset or
        World objects, only interfacing with the Renderer.

        This appears to already be mostly the case; the World object isn't
        actually used in the body of the class it appears, and RegionSet is only
        used for a couple of irrelevant things that could probably be moved
        elsewhere (persistent data handling). Mark this as TODO.

    .. method:: __init__(worldobj, regionsetobj, assetmanagerobj, options, rendererobj, outputdir)

        Initialize the TileSet object with the given parameters.

        :param worldobj: The world object to render from. This parameter is obsolete and will
            be removed.
        :type worldobj: :class:`.World`
        :param regionsetobj: The regionset to render from. This parameter is
            obsolete and will be removed.
        :type regionsetobj: :class:`.RegionSet`
        :param assetmanagerobj: Since the TileSet object needs to output some
            metadata, this is done through the assetmanager. This will also probably
            be removed in the future, in favor of the TileSet handling its own
            metadata.
        :type assetmanagerobj: :class:`~overviewer.assetmanager.AssetManager`
        :param options: The options dictionary. See the class docstring for
            valid and required options.
        :type options: dict
        :param rendererobj: Typically an :class:`.IsometricRenderer` object,
            this is any renderer object. This determines what we are actually
            rendering.
        :type rendererobj: :class:`.Renderer`
        :param outputdir: The path on the filesystem to output our files.
        :type outputdir: str

Chunk Scanning
--------------

The tilesets, as part of the preprocessing step, perform a chunk scan. This scans all chunks of the world and creates a set of tiles that need rendering, depending on the chunk’s last modified time and the configuration file’s last render timestamp.

The chunk scan operates in one of three modes. The normal mode is to scan all chunks and then render tiles that contain at least one chunk whose mtime is greater than the last render time.

Mode 1 is to mark all tiles as needing rendering, and then during rendering check the mtime of the tile itself on disk. This is used when a render was interrupted, since the last render timestamp is not accurate.

Mode 2 is to mark all tiles as needing rendering and then unconditionally render all of them.

The chunk scan loops over all “render sources” (chunks) from the Renderer object by calling :meth:`.get_render_sources()`. It then asks the Renderer for the mtime and the bounding rectangle on the virtual canvas of that chunk.

The bounding rectangle is then passed to :meth:`.RenderTile.from_rect`, which returns all the render tiles that intersect that rectangle. For each of those tile objects, it’s added to the dirty tile set depending on the scan mode.



As a side effect of the chunk scan, the maximum mtime of all chunks is determined, and this is eventually used as the “last render timestamp” of the current render.

Dispatchers and Workers
=======================
Dispatcher objects coordinate work done by worker objects. Multiple workers can
be managed by a dispatcher. Dispatchers provide features such as progress bars
and multiprocessing dispatching among workers.

.. module:: overviewer.dispatcher

.. class:: Worker

    Worker objects are objects that implement the worker interface (this class
    only defines the interface). Objects that implement this interface have some
    work that needs to be done, and has methods that are used to get how much
    work needs to be done and such.

    Specifically, work is broken down into one or more "phases". All work from
    one phase must be done before the next phase. Each phase may have a
    different amount of work. Workers may have only one phase.

    This class is subclassed by the :class:`.Canvas` object, another abstract
    class that represents a virtual canvas that needs rendering. That in turn is
    subclassed by the :class:`.TileSet` class, representing a bunch of tiles
    that need rendering (which represents the virtual canvas). For the TileSet
    class, each tile is a work item.

    .. note::

        The original idea was for phases to represent each level of the quadtree
        for TileSet objects, but this was removed when we decided to change the
        quadtree rendering order from a layer-by-layer order to a
        post-tree-traversal order. Currently, TileSet objects represent all
        their work with a single phase.

    .. method:: get_num_phases

        Returns the number of phases of work this worker has

    .. method:: get_phase_length(phasenum)

        Returns in integer indicating some notion of how much work is in the
        given phase. This is purely informational, for progress bars and such. A
        worker may return None if no reasonable estimate exists.

    .. method:: iterate_work_items(phasenum)

        This returns an iterator over all "work items" in the given phase. A
        work item is an opaque object that is passed back to the :meth:`do_work`
        method by the dispatcher. work items must be picklable and may be passed
        to a different, identically configured worker object running in a
        possibly different thread or process.

        The actual expected return type is an iterable over (work_item, [d1, d2,
        ...]) tuples where work_item is some opaque object, and `d1, d2` are
        dependency work items, indicating those work items must be completed
        before this one is started. The dispatcher is responsible for correctly
        handling dependencies. Dependency objects are compared with the equality
        operator (==) so the work item object must correctly implement that.

        The worker is guaranteed to always have a work item submitted *after*
        all its dependencies are finished. No other ordering guarantees are
        made. Dependencies, however, must come before dependent work items in
        the iterator given by this method.

    .. method:: do_work(work_item)

        performs the work for the given work item. This method is not expected
        to return anything. For TileSet objects, this calls into
        :meth:`.Renderer.render` (for leaf nodes in the quadtree; inner nodes,
        however, are composed out of their children)

.. class:: Dispatcher

    Dispatchers are given a list of worker objects to the dispatch_all method.
    Dispatchers are responsible for handing out work items to their
    corresponding workers and managing the dependencies. Dispatchers may also
    choose to copy worker objects and run some work in other threads /
    processes, but this is tricky since all state of the worker must be
    picklable and any file system or other resources the workers use must also
    be accessible by all workers.

World and RegionSet objects
===========================

These objects represent data from Minecraft worlds. They contain methods to retrieve information about the worlds.

.. module:: overviewer.world

.. class:: World

    This represents a Minecraft world. It usually contains one or more :class:`RegionSet` objects. The constructor of this object scans the directory and automatically creates a :class:`RegionSet` object for each dimension it finds.

    .. method:: __init__(worlddir)

        Initialize this World object with the given world directory. This will
        load and parse the world metadata level.dat file, and also scan for
        dimensions and create a :class:`RegionSet` object for each one.

    .. method:: get_regionsets
    
        Returns an iterable over all RegionSet objects found in this world.

    .. method:: get_regionset(index)

        Returns the RegionSet found at the given index. `index` can also specify
        the RegionSet type (as returned by :meth:`RegionSet.get_type`). The
        first matching type is returned.

        :type index: int or str
        :param index: Specifies which RegionSet to return, the one at the given
            index or the first one with the given type.
        :return: The :class:`RegionSet` object, or None if none were found with
            the given type.
        :raises: IndexError if the given index does not exist.

    .. method:: get_level_dat_data

        Returns a dictionary of the data parsed from level.dat

    .. method:: find_true_spawn

        Returns the spawn point given in level.dat, adjusted to attempt to find
        where players will actually spawn. Minecraft won't spawn users in the
        middle of a mountain for example, so this opens up the relevant chunk
        and finds the first air block above the given spawn point.

.. class:: RegionSet

    This represents a set of region files. Minecraft worlds may contain more than one "dimension", each represented by a RegionSet object.

    .. method:: __init__(regiondir, rel)

        Creates a new RegionSet. In the process, this also creates a map of
        region x,z coordinates to regionfile names.

        :param str regiondir: The path on the filesystem to the directory of
            region files.
        :param str rel: The relative path to the regiondir with respect to the world
            directory. This is only used to determine the RegionSet type.

    .. method:: get_type

        Returns the regionset type. This is `None` for the main world, or is the
        name of the directory within the world directory that holds this
        regionset, e.g. "DIM-1"

    .. method:: get_chunk(self, x, z)

        Returns a dictionary representing the parsed NBT data for the given chunk.

    .. method:: iterate_chunks(self)

        Returns an iterator over all chunks. Each item is a (x, z, mtime) tuple representing one chunk.

    .. method:: get_chunk_mtime(self, x, z)
    
        Returns the mtime for the given chunk.

Texture Objects
===============

.. module:: overviewer.textures

.. class:: Textures

    This class encapsulates a set of textures that come from a Minecraft
    resource pack or a Minecraft installation. It can find a Minecraft
    installation and use those textures, or a resource pack can be specified
    explicitly.

    This class contains routines for loading texture images into memory.

    .. method:: __init__(path)

        Initialize a Textures object from the resource pack at the given path.
        The path must name a Minecraft jar or resource pack zip (they are the
        same format). `path` may also refer to the directory of an unzipped
        resource pack.

    .. method:: load(filename)

        This loads the image with the given filename. The file is searched
        within the resource pack and loaded as an image. `filename` can also be
        the full path to an image relative to the resource pack root.

        Returned images are always square. If the given image has more than one
        frame, the first one is used.

        :returns: :class:`.oil.Image`

.. function:: get_default

    This searches for a Minecraft installation and makes a Textures object with
    it.

    TODO: This ought to be a classmethod of the Textures object, or just rolled
    into the default constructor and have path be an optional parameter.

The Chunkrenderer
=================

.. module:: overviewer.chunkrenderer

The :mod:`overviewer.chunkrenderer` module is a C extension module and is the
main gateway to the rendering routines. It is primarily used by the
:class:`.IsometricRenderer` class to do the actual rendering, but is not
restricted to isometric renders.

There are two functions exported by this module, and several constants.

Face Types
----------
These constants are used in the :attr:`~.BlockDefinition.faces` attribute of the
:class:`.BlockDefinition` object.

.. data:: FACE_TYPE_NX
.. data:: FACE_TYPE_NY
.. data:: FACE_TYPE_NZ
.. data:: FACE_TYPE_PX
.. data:: FACE_TYPE_PY
.. data:: FACE_TYPE_PZ

    These constants define the (rough) normal vector of the face.
    Positive/negative X, Y, or Z. The renderer uses this value to determine
    whether to render a face or not. A face will only be rendered if the
    neighboring block in the given direction has the TRANSPARENT property.

Python-exported Functions
-------------------------
The renderer is written in C, but there are 2 functions exposed to Python.

.. function:: compile_block_definitions(textures, blockdefs)

    This compiles the block definitions for use in rendering. It is a C function
    exported to Python.

    :param textures: is a :class:`.Textures` object
    :param blockdefs: is a :class:`.BlockDefinitions` object
    :return: an opaque object that should be passed back to the :func:`render`
        function

    This first creates a new :c:type:`BlockDefs` struct. The
    :attr:`.BlockDefinitions.max_blockid` and :attr:`.BlockDefinitions.max_data`
    attributes are saved into the struct.

    Then the code looks at the max blockid and max data parameters and allocates
    an array of :c:type:`BlockDef` structs with length max_blockid * max_data,
    and assigns it to the :c:data:`BlockDefs.defs` pointer.

    Then for each block definition in the :attr:`.BlockDefinitions.blocks` dict,
    a :c:type:`BlockDef` struct is allocated and added to the :c:data:`BlockDefs.defs`
    array. It does this by calling :c:func:`compile_block_definition`, a
    staticly defined inline function.

.. function:: render(regionset, chunk_x, chunk_y, chunk_z, image, matrix, compiled_blockdefs)

    This function is called to render a single chunk section on the given image
    using the given perspective matrix, pulling chunk data from the given
    regionset, using the given block definitions.

    :param regionset: is the :class:`.RegionSet` object to pull chunk
        information from
    :param chunk_x: is the x coordinate of the chunk to render
    :param chunk_y: is the section number of the chunk to render
    :param chunk_z: is the z coordinate of the chunk to render
    :param image: is the :class:`.oil.Image` object to render onto.
    :param matrix: is the :class:`.oil.Matrix` object to use as the projection
        matrix
    :param compiled_blockdefs: is the object returned from the
        :func:`compile_block_definitions` function.

    This is the main entry point for rendering. See the :ref:`rendering` section
    for details and a complete walkthrough of how it works.

Compiling the Blocks
--------------------

This C function compiles the block definitions.

.. c:function:: int compile_block_definition(PyObject* pytextures, BlockDef* def, PyObject* pydef)

    This is a static inline C function used to compile a particular block's
    definition and store it in the BlockDef struct pointer given.
    
    `pytextures` is the :class:`.Textures` object given to
    :func:`compile_block_definitions`.

    `def` is a pointer to the appropriate element of the
    :c:data:`BlockDefs.defs` array created above.

    `pydef` is a pointer to the :class:`.BlockDefinition` object.
    
    This function does the following:

    #. Gets the :attr:`~.BlockDefinition.verticies`,
       :attr:`~.BlockDefinition.triangles`, and
       :attr:`~.BlockDefinition.tex` attributes from the
       :class:`.BlockDefinition` object.

    #. Calls :meth:`.Textures.load` with the value returned from the `tex`
       attribute. The return value is expected to be a :class:`.oil.Image` type.

    #. Checks the `transparent`, `solid`, `fluid`, `nospawn`, and `nodata`
       attributes and sets the corresponding booleans on the struct

    #. Checks that the verticies and triangles objects returned were sequences

    #. Initializes the verticies buffer by calling :c:func:`buffer_init`

    #. For every face, initializes 6 indices buffers by calling
       :c:func:`buffer_init`

    #. Loops over the verticies sequence. Creates a new :c:type:`OILVertex`
       struct and fills it with the values from the current sequence element.
       Calls :c:func:`buffer_append` to add this struct to the verticies buffer.

    #. Loops over the triangles sequence. Does the same as  for the verticies
       except instead of a OILVertex object, it's just an array of 3 ints.
       Triangles are added to the corresponding buffer that they were declared
       with depending on the face type declaration.

    #. Sets the :c:data:`BlockDef.tex` member to the underlying image structure
       of the texture object returned from Textures.load.

    #. Sets the known member to 1.


Structs Used
~~~~~~~~~~~~

.. c:type:: BlockDefs

    A structure representing all the block definitions

    .. c:member:: BlockDef* defs
    .. c:member:: unsigned int max_blockid
    .. c:member:: unsigned int max_data

.. c:type:: BlockDef

    A single definition of a block. Holds the same data as the corresponding
    :class:`.BlockDefinition` object: verticies, indicies, texture, and several
    booleans.

    .. c:member:: int known

        A value that is by default 0 since the :data:`BlockDefs.defs` array is
        initialized with `calloc`. This is therefore set to nonzero if this
        block is defined.

    .. c:member:: Buffer verticies
    .. c:member:: Buffer indicies[FACE_TYPE_COUNT]

        This holds the face definitions from the
        :attr:`.BlockDefinition.triangles` attribute. It is grouped by the face
        type declaration.

    .. c:member:: OILImage* tex

    .. c:member:: int transparent
    .. c:member:: int solid
    .. c:member:: int fluid
    .. c:member:: int nospawn
    .. c:member:: int nodata


Buffers
-------
Buffers are used in a few places and are described in this section.

A Buffer is a dynamic array of some arbitrary element type. It has an efficient
append method that doubles the allocated space on expansion.

.. c:type:: Buffer

    .. c:member:: void* data
    .. c:member:: unsigned int element_size
    .. c:member:: unsigned int length

        The number of elements currently stored in the buffer.

    .. c:member:: unsigned int reserved
        
        The amount of space actually allocated to the :c:member:`data`
        pointer. This is given in number of elements.

.. c:function:: void buffer_init(Buffer* buffer, unsigned int element_size, unsigned int initial_length)

    Initializes a buffer given a newly-allocated Buffer struct. This sets the
    :c:member:`~Buffer.data` element to `NULL`, :c:member:`~Buffer.length` to 0,
    and the other two members to their corresponding parameter values.

.. c:function:: void buffer_append(Buffer* buffer, const void* newdata, unsigned int newdata_length)
    
    Calls :c:func:`buffer_reserve` to make sure there is enough space allocated
    in the buffer.

    Then calls `memcpy` to append the new data to the buffer and changes the
    buffer's length attribute appropriately.

.. c:function:: void buffer_reserve(Buffer* buffer, unsigned int length)

    Makes sure there is space for at least :c:member:`Buffer.length` + `length`
    elements allocated in the buffer. This function may call `realloc` on the
    buffer's :c:member:`~Buffer.data` member.

.. _rendering:

Rendering
---------

Rendering starts at the :func:`overviewer.chunkrenderer.render` function. This calls into
the C function :c:func:`render` defined in chunkrenderer.c.

.. c:function:: PyObject* render(PyObject* self, PyObject* args)

    This function does the following:

    #. Creates a :c:type:`RenderState` struct on the stack.

    #. The following attributes of the `RenderState` struct are saved from the
       arguments to the function: `blockdefs`, `im`, `matrix`, the chunk
       coordinates, `regionset`.

    #. Sets the :c:member:`ChunkData.loaded` property of each element of the
       :c:member:`RenderState.chunks` member to 0.

    #. :c:func:`load_chunk` is called with the address of the `state` struct,
       and the parameters 0, 0, 1. This indicates that it should load the chunk
       at relative position 0,0 (the chunk we're actually rendering) and that it
       is required. See docs for that function for more information.

       This loads the chunk into slot [1][1] of the `chunks` member.
    
    #. If the requested section of the requested chunk was not loaded
       successfully or does not exist, the function bails.

    #. Two :c:type:`Buffer` structs are declared on the stack: `blockverticies`
       and `blockindicies`. Calls :c:func:`buffer_init` twice to "set up the
       mesh buffers". Once with the address of each buffer.

       `blockverticies` will hold :c:type:`OILVertex` structs, and
       `blockindicies` will hold unsigned ints.

    #. Local variables `blocks` and `datas` are set to point to the PyObjects
       for the :c:member:`~section.blocks` and :c:member:`~section.data` values
       of the current chunk section.

    #. Seeds the random number generator with a fixed constant

    #. Does the following in a loop over every block in the current chunk
       section

       #. Calls :c:func:`get_array_short_3d` to get the current block ID.

       #. Calls :c:func:`get_array_byte_3d` to get the current data byte.

       #. Calls :c:func:`get_block_definition` to get a :c:type:`BlockDef`
          pointer for the current block.

       #. Sets the `blockverticies` buffer length to 0, effectively clearing the
          array (but not deallocating it)

       #. Calls :c:func:`buffer_append` to append the block data verticies to
          this buffer.

       #. For each vertex in the vertex buffer, adjusts the x, y, and z
          coordinates by the block's x,y,z coordinates within the chunk. This
          sets each vertex coordinate relative to the chunk, not to the block.

       #. Sets the `blockindicies` buffer length to 0.

       #. Calls :c:func:`get_data` for each neighboring block.

       #. For the neighboring blocks who have the property `TRANSPARENT`,
          appends the faces whose face type property is facing the transparent
          neighboring block to the `blockindicies` buffer.

       #. If the `blockindicies` buffer is not empty, calls :c:func:`emit_mesh`
          to do the actual drawing of the culled faces and verticies.

    #. Memory is freed and None is returned.

.. c:function:: int load_chunk(RenderState* state, int relx, int relz, int required)
.. c:function:: get_array_short_3d
.. c:function:: get_array_byte_3d
.. c:function:: BlockDef* get_block_definition(RenderState* state, int x, int y, int z, unsigned int blockid, unsigned int data)

    Returns the `BlockDef` struct that defines the given block/data pair, or
    NULL if none exists.

    The x,y,z parameters are for blocks that need to consider data from
    neighboring blocks—"pseudo data". This is not currently implemented.

.. c:function:: get_data
.. c:function:: void emit_mesh(RenderState* state, OILImage* tex, const Buffer* verticies, const Buffer* indicies)

    This function is a one-line dispatch to :c:func:`oil_image_draw_triangles`


Structs Used
~~~~~~~~~~~~

.. c:type:: RenderState

    This defines the configuration parameters for a particular call to the
    :c:func:`render` function.

    .. c:member:: PyObject* regionset
    .. c:member:: int chunkx
    .. c:member:: int chunky
    .. c:member:: int chunkz

        Holds the chunk coordinates and chunk section that is being rendered by
        the current call to :c:func:`render`. chunkx and z specify the chunk
        address, and chunky specifies the section within the chunk.

    .. c:member:: ChunkData chunks

        This is a [3][3] array holding the chunk and 8 surrounding chunks used
        in the current render. Since some blocks depend on neighboring blocks,
        those chunks may be loaded and stored here. (These are not loaded
        preemptively, however.)

    .. c:member:: OILImage* im
    .. c:member:: OILMatrix *matrix
    .. c:member:: BlockDefs* blockdefs

.. c:type:: ChunkData

    Holds the loaded and parsed data for a single chunk (including all chunk
    sections)

    .. c:member:: int loaded
    .. c:member:: PyObject* biomes
    .. c:member:: section sections

        This is actually an inline-defined struct array. It contains
        :c:macro:`SECTIONS_PER_CHUNK` elements.

.. c:type:: section

    This is a struct that only exists within the scope of the
    :c:type:`ChunkData` struct. It's defined separately in the docs due to
    limitations in the documentation software.

    Each member represents all there is to know about a particular chunk section

    .. c:member:: PyObject* blocks
    .. c:member:: PyObject* data
    .. c:member:: PyObject* skylight
    .. c:member:: PyObject* blocklight

.. c:type:: OILVertex

    Defines a single vertex.

    .. c:member:: float x
    .. c:member:: float y
    .. c:member:: float z
    .. c:member:: float s
    .. c:member:: float t
    .. c:member:: OILPixel color

.. c:type:: OILPixel

    Defines a color plus alpha channel

    .. c:member:: unsigned char r
    .. c:member:: unsigned char g
    .. c:member:: unsigned char b
    .. c:member:: unsigned char a

OIL
===

This section describes OIL, the core Overviewer Imaging Library.

Python Functions
----------------
This module exports the following functions, as well as
:class:`~overviewer.oil.Image` and :class:`~overviewer.oil.Matrix` already
documented elsewhere.

.. module:: overviewer.oil

.. function:: backend_set

    This sets the OIL backend to use. Set it to one of the BACKEND_* constants
    described below. This calls :c:func:`oil_backend_set`

.. data:: BACKEND_CPU
.. data:: BACKEND_DEBUG
.. data:: BACKEND_CPU_SSE

    Only defined when compiled with ENABLE_CPU_SSE_BACKEND

.. data:: BACKEND_OPENGL

    Only defined when compiled with ENABLE_OPENGL_BACKEND

The above constants also have C equivalents called OIL_BACKEND_*, defined in an
enum in oil.h with some macros.

.. c:function:: int oil_backend_set(OILBackendName backend)

    Defined in oil-backend.c, this sets the global pointer :c:data:`oil_backend`

.. c:var:: OILBackend* oil_backend

    This is a global pointer indicating which backend is set. It is by default
    set to the address of :c:data:`oil_backend_cpu`

Backends
--------

The :c:type:`OILBackend` struct
defines a number of function pointers that control how to do the rendering.

CPU backends are defined in oil-backend-cpu.def. Both CPU backend includes this
file after setting some other macros appropriately, since they share a lot of
code. It appears the only difference is that the SSE backend also includes
emmintrin.h and stdint.h.

.. c:type:: OILBackend

    This struct is defined in oil-backend-private.h. This struct defines a
    rendering backend, which is a set of primitive operations needed to support
    our rendering requirements. It holds the following function pointers:

    .. c:member:: int (*initialize)(void)

        Called when starting up this backend. Return 0 for failure.

    .. c:member:: void (*new)(OILImage *im)

        Called when creating an image

    .. c:member:: void (*free)(OILImage *im)

        Called when destroying an image

    .. c:member:: void (*load)(OILImage *im)

        Load data out of backend and into im. called during (for example)
        get_data()

    .. c:member:: void (save)(OILImage *im)

        Save data from im into the backend. called during (for example) unlock()

    .. c:member:: int (*composite)(OILImage *im, OILImage *src, unsigned char alpha, int dx, int dy, unsigned int sx, unsigned int sy, unsigned int xsize, unsigned int ysize)

        Do a composite operation

    .. c:member:: void (*draw_triangles)(OILImage *im, OILMatrix *matrix, OILImage *tex, OILVertex *verticies, unsigned int verticies_length, unsigned int *indicies, unsigned int indicies_length, OILTriangleFlags flags)

        Draw triangles

    .. c:member:: (*resize_half)(OILImage *im, OILImage *src)

        Cut src in half and write to im

    .. c:member:: (*clear)(OILImage *im)

        Clear the image

Oil Image Objects
-----------------

.. currentmodule:: overviewer.oil

.. class:: Image

    This is a python-exported object to an :c:type:`OILImage` struct and some
    associated methods.

    .. classmethod:: load(path)

        :type path: str or file-like object
        :param path: The file to load       

        Load the given path name into an Image object. This dispatches to
        :c:func:`oil_image_load_ex` (either through :c:func:`oil_image_load` or
        by calling it itself, depending on whether `path` is a str or a file
        object).

    .. method:: save

        Save the image object to a file

    .. method:: get_size

        Return a (width, height) tuple

    .. method:: composite

        Composite another image on top of this one

    .. method:: draw_triangles

        Draw 3D triangles on top of the image
        
    .. method:: resize_half

        Shrink the given image by half and copy onto self

    .. method:: clear

        Clear the image
    

.. c:type:: OILImage

    .. c:member:: unsigned int width
    .. c:member:: unsigned int height
    .. c:member:: OILPixel *data
    .. c:member:: int locked
    .. c:member:: OILBackend *backend
    .. c:member:: void *backend_data

.. c:function:: OILImage *oil_image_load(const char *path)

    Sets up a :c:type:`OILFile` with the `oil_image_read` reader and calls
    :c:func:`oil_image_load_ex`

.. c:function:: OILImage *oil_image_load_ex(OILFile *file)

    Loads an image from the specified file. Right now this is hard coded to read
    from element 0 of the :c:data:`oil_formats` array, which is the PNG reader.
    TODO: have this support other formats.

Oil Files and Image Loaders
---------------------------

.. c:var:: OILFormat *oil_formats

    In oil-format.c is an array of image formats that oil supports reading and
    writing. Each element of this array is a struct that has two function
    pointers, one for reading and one for writing. Without going into too many
    details, the load function takes an :c:type:`OILFile` struct and returns an
    :c:type:`OILImage` pointer.

.. c:type:: OILFile

    This is a struct defined in oil.h. It has 4 members

    .. c:member:: void *file
    .. c:member:: size_t (*read)(void *file, void *data, size_t length)
    .. c:member:: size_t (*write)(void *file, void *data, size_t length)
    .. c:member:: void (*flush)(void *file)

    This struct abstracts different ways of loading an image. This way we can
    support reading from a raw FD, a python file object, etc.

    Implementations include:
    
    * `oil_python_read`, `oil_python_write`, and `oil_python_flush` defined in
      oil-python.c, for reading/writing to python file objects. `*file` is a
      pointer to the Python File object.
    * `oil_image_read`, `oil_image_write`, `oil_image_flush` defined in
      oil-image.c for reading/writing plain old FILEs. `*file` is the FILE
      pointer.

PNG Format
~~~~~~~~~~
This is currently the only image format defined in oil-format.c. The png
functions are defined in oil-format-png.c.

There's no need to go into details here, it's just a lot of libpng boilerplate.
Worth noting is the support of indexed PNGs in the output writer.

C Rendering Functions
---------------------

.. c:function:: void oil_image_draw_triangles(OILImage *im, OILMatrix *matrix, OILImage *tex, OILVertex *vertices, unsigned int vertices_length, unsigned int *indices, unsigned int indices_length, OILTriangleFlags flags)

    This is the main entry point for the C renderer functionality. It is called
    from the :func:`overviewer.chunkrenderer.render` function's implementation,
    as well as from :meth:`.Image.draw_triangles` (only the former is used in
    the main overviewer workflow)

    It is defined in oil-image.c

    Since it is drawing triangles, `indices_length` must be a multiple of 3. The
    destination `im` and the texture `tex` must have the same backend. This
    dispatches to the `draw_triangles` routine of the backend, which is saved as
    :c:data:`OILImage.backend`
