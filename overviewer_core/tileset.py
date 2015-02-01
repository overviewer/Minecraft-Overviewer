#    This file is part of the Minecraft Overviewer.
#
#    Minecraft Overviewer is free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or (at
#    your option) any later version.
#
#    Minecraft Overviewer is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#    Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with the Overviewer.  If not, see <http://www.gnu.org/licenses/>.

import itertools
import logging
import os
import os.path
import sys
import shutil
import random
import functools
import time
import errno
import stat
import platform
from collections import namedtuple
from itertools import product, izip, chain

from PIL import Image

from .util import roundrobin
from . import nbt
from .files import FileReplacer, get_fs_caps
from .optimizeimages import optimize_image
import rendermodes
import c_overviewer
from c_overviewer import resize_half

"""

tileset.py contains the TileSet class, and in general, routines that manage a
set of output tiles corresponding to a requested rendermode for a world. In
general, there will be one TileSet object per world per rendermode requested by
the user.

The TileSet class implements the Worker interface. This interface has the
following methods:

do_preprocessing()
    This method is called before iterate_work_items(). It should do any work
    that needs to be done prior to iterate_work_items(). It is not called for
    instances that will not have iterate_work_items() called.

get_num_phases()
    This method returns an integer indicating how many phases of work this
    worker has to perform. Each phase of work is completed serially with the
    other phases... all work done by one phase is done before the next phase is
    started.

get_phase_length(phase)
    This method returns an integer indicating how many work items there are in
    this phase. This number is used for purely informational purposes. It can
    be exact, or an estimate. If there is no useful information on the size of
    a phase, return None.

iterate_work_items(phase)
    Takes a phase number (a non-negative integer). This method should return an
    iterator over work items and a list of dependencies i.e. (work_item, [d1,
    d2, ...]). The work items and dependencies can be any pickelable object;
    they are treated as opaque by the Dispatcher. The work item objects are
    passed back in to the do_work() method (perhaps in a different, identically
    configured instance).

    The dependency items are other work items that are compared for equality
    with work items that are already in the queue. The dispatcher guarantees
    that dependent items which are currently in the queue or in progress finish
    before the corresponding work item is started. Note that dependencies must
    have already been yielded as work items before they can be used as
    dependencies; the dispatcher requires this ordering or it cannot guarantee
    the dependencies are met.

do_work(workobj)
    Does the work for a given work object. This method is not expected to
    return anything, so the results of its work should be reflected on the
    filesystem or by sending signals.


"""

# small but useful
def iterate_base4(d):
    """Iterates over a base 4 number with d digits"""
    return product(xrange(4), repeat=d)

# A named tuple class storing the row and column bounds for the to-be-rendered
# world
Bounds = namedtuple("Bounds", ("mincol", "maxcol", "minrow", "maxrow"))

# A note about the implementation of the different rendercheck modes:
#
# For reference, here's what the rendercheck modes are:
#   0
#       Only render tiles that have chunks with a greater mtime than the last
#       render timestamp, and their ancestors.
#
#       In other words, only renders parts of the map that have changed since
#       last render, nothing more, nothing less.
#
#       This is the fastest option, but will not detect tiles that have e.g.
#       been deleted from the directory tree, or pick up where a partial
#       interrupted render left off.

#   1
#       For render-tiles, render all whose chunks have an mtime greater than
#       the mtime of the tile on disk, and their composite-tile ancestors.
#
#       Also rerender any tiles rendered before forcerendertime. It is nonzero
#       whenever a mode=2 render has been interrupted.
#
#       Also check all other composite-tiles and render any that have children
#       with more rencent mtimes than itself.
#
#       This is slower due to stat calls to determine tile mtimes, but safe if
#       the last render was interrupted.

#   2
#       Render all tiles unconditionally. This is a "forcerender" and is the
#       slowest, but SHOULD be specified if this is the first render because
#       the scan will forgo tile stat calls. It's also useful for changing
#       texture packs or other options that effect the output.

#   3
#       A very special mode. Using this will not actually render
#       anything, but will leave this tileset in the resulting
#       map. Useful for renders that you want to keep, but not
#       update. Since this mode is so simple, it's left out of the
#       rest of this discussion.

#
# For 0 our caller has explicitly requested not to check mtimes on disk to
# speed things up. So the mode 0 chunk scan only looks at chunk mtimes and the
# last render mtime from the asset manager, and marks only the tiles that need
# rendering based on that.  Mode 0 then iterates over all dirty render-tiles
# and composite-tiles that depend on them. It does not check mtimes of any
# tiles on disk, so this is only a good option if the last render was not
# interrupted.

# For mode 2, this is a forcerender, the caller has requested we render
# everything. The mode 2 chunk scan marks every tile as needing rendering, and
# disregards mtimes completely. Mode 2 then iterates over all render-tiles and
# composite-tiles that depend on them, which is every tile. It therefore
# renders everything.

# In both 0 and 2 the render iteration is the same: the dirtytile tree built is
# authoritive on every tile that needs rendering.

# In mode 1, things are most complicated. Mode 1 chunk scan is identical to a
# forcerender, or mode 2 scan: every render tile that should exist is marked in
# the dirtytile tree. But instead of iterating over that tree directly, a
# special recursive algorithm goes through and checks every tile that should
# exist and determines whether it needs rendering. This routine works in such a
# way so that every tile is stat()'d at most once, so it shouldn't be too bad.
# This logic happens in the iterate_work_items() method, and therefore in the
# master process, not the worker processes.

# In all three rendercheck modes, the results out of iterate_work_items() is
# authoritive on what needs rendering. The do_work() method does not need to do
# any additional checks.

__all__ = ["TileSet"]
class TileSet(object):
    """The TileSet object manages the work required to produce a set of tiles
    on disk. It calculates the work that needs to be done and tells the
    dipatcher (through the Worker interface) this information. The Dispatcher
    then tells this object when and where to do the work of rendering the tiles.

    """

    def __init__(self, worldobj, regionsetobj, assetmanagerobj, texturesobj, options, outputdir):
        """Construct a new TileSet object with the given configuration options
        dictionary.

        options is a dictionary of configuration parameters (strings mapping to
        values) that are interpreted by the rendering engine.
        
        worldobj is the World object that regionsetobj is from.

        regionsetobj is the RegionSet object that is used to render the tiles.

        assetmanagerobj is the AssetManager object that represents the
        destination directory where we'll put our tiles.

        texturesobj is the Textures object to pass into the rendering routine.
        This class does nothing with it except pass it through.

        outputdir is the absolute path to the tile output directory where the
        tiles are saved. It is created if it doesn't exist

        Current valid options for the options dictionary are shown below. All
        the options must be specified unless they are not relevant. If the
        given options do not conform to the specifications, behavior is
        undefined (this class does not do any error checking and assumes items
        are given in the correct form).

        bgcolor
            A hex string specifying the background color for jpeg output.
            e.g.: "#1A1A1A". Not relevant unless rendering jpeg.

        renderchecks
            An integer indicating how to determine which tiles need updating
            and which don't. This key is optional; if not specified, an
            appropriate mode is determined from the persistent config obtained
            from the asset manager. This is one of three levels:

            0
                Only render tiles that have chunks with a greater mtime than
                the last render timestamp, and their ancestors.

                In other words, only renders parts of the map that have changed
                since last render, nothing more, nothing less.

                This is the fastest option, but will not detect tiles that have
                e.g. been deleted from the directory tree, or pick up where a
                partial interrupted render left off.

            1
                "check-tiles" mode. For render-tiles, render all whose chunks
                have an mtime greater than the mtime of the tile on disk, and
                their upper-tile ancestors.

                Also check all other upper-tiles and render any that have
                children with more rencent mtimes than itself.

                Also remove tiles and directory trees that do exist but
                shouldn't.

                This is slower due to stat calls to determine tile mtimes, but
                safe if the last render was interrupted.

            2
                Render all tiles unconditionally. This is a "forcerender" and
                is the slowest, but SHOULD be specified if this is the first
                render because the scan will forgo tile stat calls. It's also
                useful for changing texture packs or other options that effect
                the output.

            3
                A very special mode. Using this will not actually render
                anything, but will leave this tileset in the resulting
                map. Useful for renders that you want to keep, but not
                update. Since this mode is so simple, it's left out of the
                rest of this discussion.

        imgformat
            A string indicating the output format. Must be one of 'png' or
            'jpeg'

        imgquality
            An integer 1-100 indicating the quality of the jpeg output. Only
            relevant in jpeg mode.

        optimizeimg
            A list of optimizer instances to use.

        rendermode
            Perhaps the most important/relevant option: a string indicating the
            render mode to render. This rendermode must have already been
            registered with the C extension module.

        rerenderprob
            A floating point number between 0 and 1 indicating the probability
            that a tile which is not marked for render by any mtime checks will
            be rendered anyways. 0 disables this option.

        changelist
            Optional: A file descriptor which will be opened and used as the
            changelist output: each tile written will get outputted to the
            specified fd.

        Other options that must be specified but aren't really documented
        (oops. consider it a TODO):
        * worldname_orig
        * dimension
        * title
        * name

        """
        self.options = options
        self.world = worldobj
        self.regionset = regionsetobj
        self.am = assetmanagerobj
        self.textures = texturesobj
        self.outputdir = os.path.abspath(outputdir)

        config = self.am.get_tileset_config(self.options.get("name"))
        self.config = config

        self.last_rendertime = config.get('last_rendertime', 0)
        self.forcerendertime = config.get('forcerendertime', 0)

        if "renderchecks" not in self.options:
            # renderchecks was not given, this indicates it was not specified
            # in either the config file or the command line. The following code
            # attempts to detect the most appropriate mode
            if not config:
                # No persistent config?
                if os.path.exists(self.outputdir):
                    # Somehow there's no config but the output dir DOES exist.
                    # That's strange!
                    logging.warning(
                        "For render '%s' I couldn't find any persistent config, "
                        "but I did find my tile directory already exists. This "
                        "shouldn't normally happen, something may be up, but I "
                        "think I can continue...", self.options['name'])
                    logging.info("Switching to --check-tiles mode")
                    self.options['renderchecks'] = 1
                else:
                    # This is the typical code path for an initial render, make
                    # this a "forcerender"
                    self.options['renderchecks'] = 2
                    logging.debug("This is the first time rendering %s. Doing" +
                            " a full-render",
                            self.options['name'])
            elif not os.path.exists(self.outputdir):
                # Somehow the outputdir got erased but the metadata file is
                # still there. That's strange!
                logging.warning(
                        "This is strange. There is metadata for render '%s' but "
                        "the output directory is missing. This shouldn't "
                        "normally happen. I guess we have no choice but to do a "
                        "--forcerender", self.options['name'])
                self.options['renderchecks'] = 2
            elif config.get("render_in_progress", False):
                # The last render must have been interrupted. The default should be
                # a check-tiles render then
                logging.warning(
                        "The last render for '%s' didn't finish. I'll be " +
                        "scanning all the tiles to make sure everything's up "+
                        "to date.",
                        self.options['name'],
                        )
                logging.warning("The total tile count will be (possibly "+
                        "wildly) inaccurate, because I don't know how many "+
                        "tiles need rendering. I'll be checking them as I go")
                if self.forcerendertime != 0:
                    logging.info(
                            "The unfinished render was a --forcerender. " +
                            "Rerendering any tiles older than %s",
                            time.strftime("%x %X", time.localtime(self.forcerendertime)),
                            )
                self.options['renderchecks'] = 1
            else:
                logging.debug("No rendercheck mode specified for %s. "+
                        "Rendering tile whose chunks have changed since %s",
                        self.options['name'],
                        time.strftime("%x %X", time.localtime(self.last_rendertime)),
                        )
                self.options['renderchecks'] = 0

        if not os.path.exists(self.outputdir):
            if self.options['renderchecks'] != 2:
                logging.warning(
                "The tile directory didn't exist, but you have specified "
                "explicitly not to do a --fullrender (which is the default for "
                "this situation). I'm overriding your decision and setting "
                "--fullrender for just this run")
                self.options['renderchecks'] = 2
            os.mkdir(self.outputdir)

        # must wait until outputdir exists
        self.fs_caps = get_fs_caps(self.outputdir)

        if self.options['renderchecks'] == 2:
            # Set forcerendertime so that upon an interruption the next render
            # will continue where we left off.
            self.forcerendertime = int(time.time())

        # Set the image format according to the options
        if self.options['imgformat'] == 'png':
            self.imgextension = 'png'
        elif self.options['imgformat'] in ('jpeg', 'jpg'):
            self.imgextension = 'jpg'
        else:
            raise ValueError("imgformat must be one of: 'png' or 'jpg'")

        # This sets self.treedepth, self.xradius, and self.yradius
        self._set_map_size()

    # Only pickle the initial state. Don't pickle anything resulting from the
    # do_preprocessing step
    def __getstate__(self):
        return self.world, self.regionset, self.am, self.textures, self.options, self.outputdir
    def __setstate__(self, state):
        self.__init__(*state)

    def do_preprocessing(self):
        """For the preprocessing step of the Worker interface, this does the
        chunk scan and stores the resulting tree as a private instance
        attribute for later use in iterate_work_items()

        """

        # skip if we're told to
        if self.options['renderchecks'] == 3:
            return
        
        # REMEMBER THAT ATTRIBUTES ASSIGNED IN THIS METHOD ARE NOT AVAILABLE IN
        # THE do_work() METHOD (because this is only called in the main process
        # not the workers)

        # This warning goes here so it's only shown once
        if self.treedepth >= 15:
            logging.warning("Just letting you know, your map requires %s zoom levels. This is REALLY big!",
                    self.treedepth)

        # Do any tile re-arranging if necessary. Skip if there was no config
        # from the asset-manager, which typically indicates this is a new
        # render
        if self.config:
            self._rearrange_tiles()

        # Do the chunk scan here
        self.dirtytree = self._chunk_scan()

    def get_num_phases(self):
        """Returns the number of levels in the quadtree, which is equal to the
        number of phases of work that need to be done.

        """
        return 1

    def get_phase_length(self, phase):
        """Returns the number of work items in a given phase.
        """
        # Yeah functional programming!
        # and by functional we mean a bastardized python switch statement
        return {
                0: lambda: self.dirtytree.count_all(),
                #there is no good way to guess this so just give total count
                1: lambda: (4**(self.treedepth+1)-1)/3,
                2: lambda: self.dirtytree.count_all(),
                3: lambda: 0,
                }[self.options['renderchecks']]()

    def iterate_work_items(self, phase):
        """Iterates over the dirty tiles in the tree and return them in the
        appropriate order with the appropriate dependencies.

        This method returns an iterator over (obj, [dependencies, ...])
        """

        # skip if asked to
        if self.options['renderchecks'] == 3:
            return
        
        # The following block of code implementes the changelist functionality.
        fd = self.options.get("changelist", None)
        if fd:
            logging.debug("Changelist activated for %s (fileno %s)", self, fd)
            # This re-implements some of the logic from do_work()
            def write_out(tilepath):
                if len(tilepath) == self.treedepth:
                    rt = RenderTile.from_path(tilepath)
                    imgpath = rt.get_filepath(self.outputdir, self.imgextension)
                elif len(tilepath) == 0:
                    imgpath = os.path.join(self.outputdir, "base."+self.imgextension)
                else:
                    dest = os.path.join(self.outputdir, *(str(x) for x in tilepath[:-1]))
                    name = str(tilepath[-1])
                    imgpath = os.path.join(dest, name) + "." + self.imgextension
                # We use low-level file output because we don't want open file
                # handles being passed to subprocesses. fd is just an integer.
                # This method is only called from the master process anyways.
                # We don't use os.fdopen() because this fd may be shared by
                # many tileset objects, and as soon as this method exists the
                # file object may be garbage collected, closing the file.
                os.write(fd, imgpath + "\n")


        # See note at the top of this file about the rendercheck modes for an
        # explanation of what this method does in different situations.
        #
        # For modes 0 and 2, self.dirtytree holds exactly the tiles we need to
        # render. Iterate over the tiles in using the posttraversal() method.
        # Yield each item. Easy.
        if self.options['renderchecks'] in (0,2):
            for tilepath in self.dirtytree.posttraversal(robin=True):
                dependencies = []
                # These tiles may or may not exist, but the dispatcher won't
                # care according to the worker interface protocol It will only
                # wait for the items that do exist and are in the queue.
                for i in range(4):
                    dependencies.append( tilepath + (i,) )
                if fd:
                    write_out(tilepath)
                yield tilepath, dependencies

        else:
            # For mode 1, self.dirtytree holds every tile that should exist,
            # but invoke _iterate_and_check_tiles() to determine which tiles
            # need rendering.
            for tilepath, mtime, needs_rendering in self._iterate_and_check_tiles(()):
                if needs_rendering:
                    dependencies = []
                    for i in range(4):
                        dependencies.append( tilepath + (i,) )
                    if fd:
                        write_out(tilepath)
                    yield tilepath, dependencies

    def do_work(self, tilepath):
        """Renders the given tile.

        tilepath is yielded by iterate_work_items and is an iterable of
        integers representing the path of the tile to render.

        """
        if len(tilepath) == self.treedepth:
            # A render-tile
            self._render_rendertile(RenderTile.from_path(tilepath))
        else:
            # A composite-tile
            if len(tilepath) == 0:
                # The base tile
                dest = self.outputdir
                name = "base"
            else:
                # All others
                dest = os.path.join(self.outputdir, *(str(x) for x in tilepath[:-1]))
                name = str(tilepath[-1])
            self._render_compositetile(dest, name)

    def get_initial_data(self):
        """This is called similarly to get_persistent_data, but is called after
        do_preprocessing but before any work is acutally done.

        """
        d = self.get_persistent_data()
        # This is basically the same as get_persistent_data() with the
        # following exceptions:
        # * last_rendertime is not changed
        # * A key "render_in_progress" is set to True
        # * forcerendertime is set so that an interrupted mode=2 render will
        #   finish properly.
        d['last_rendertime'] = self.last_rendertime
        d['render_in_progress'] = True
        d['forcerendertime'] = self.forcerendertime
        return d

    def get_persistent_data(self):
        """Returns a dictionary representing the persistent data of this
        TileSet. Typically this is called by AssetManager

        """
        def bgcolorformat(color):
            return "#%02x%02x%02x" % color[0:3]
        isOverlay = self.options.get("overlay") or (not any(isinstance(x, rendermodes.Base) for x in self.options.get("rendermode")))

        # don't update last render time if we're leaving this alone
        last_rendertime = self.last_rendertime
        if self.options['renderchecks'] != 3:
            last_rendertime = self.max_chunk_mtime        
        
        d = dict(name = self.options.get('title'),
                zoomLevels = self.treedepth,
                defaultZoom = self.options.get('defaultzoom'),
                maxZoom = self.options.get('maxzoom', self.treedepth) if self.options.get('maxzoom', self.treedepth) >= 0 else self.treedepth+self.options.get('maxzoom'),
                path = self.options.get('name'),
                base = self.options.get('base'),
                bgcolor = bgcolorformat(self.options.get('bgcolor')),
                world = self.options.get('worldname_orig') +
                    (" - " + self.options.get('dimension')[0] if self.options.get('dimension')[1] != 0 else ''),
                last_rendertime = last_rendertime,
                imgextension = self.imgextension,
                isOverlay = isOverlay,
                poititle = self.options.get("poititle"),
                showlocationmarker = self.options.get("showlocationmarker")
                )
        d['maxZoom'] = min(self.treedepth, d['maxZoom'])
        d['minZoom'] = min(max(0, self.options.get("minzoom", 0)), d['maxZoom'])
        d['defaultZoom'] = max(d['minZoom'], min(d['defaultZoom'], d['maxZoom']))

        if isOverlay:
            d.update({"tilesets": self.options.get("overlay")})

        # None means overworld
        if (self.regionset.get_type() == None and self.options.get("showspawn", True)):
            d.update({"spawn": self.options.get("spawn")})
        else:
            d.update({"spawn": "false"});

        try:
            d['north_direction'] = self.regionset.north_dir
        except AttributeError:
            d['north_direction'] = 0

        return d

    def _find_chunk_range(self):
        """Finds the chunk range in rows/columns and stores them in
        self.minrow, self.maxrow, self.mincol, self.maxcol

        """
        minrow = mincol = maxrow = maxcol = 0

        for c_x, c_z, _ in self.regionset.iterate_chunks():
            # Convert these coordinates to row/col
            col, row = convert_coords(c_x, c_z)

            minrow = min(minrow, row)
            maxrow = max(maxrow, row)
            mincol = min(mincol, col)
            maxcol = max(maxcol, col)
        return Bounds(mincol, maxcol, minrow, maxrow)

    def _set_map_size(self):
        """Finds and sets the depth of the map's quadtree, as well as the
        xradius and yradius of the resulting tiles.

        Sets self.treedepth, self.xradius, self.yradius

        """
        # Calculate the min and max column over all the chunks.
        # This returns a Bounds namedtuple
        bounds = self._find_chunk_range()

        # Calculate the depth of the tree
        for p in xrange(2,33): # max 32
            # Will 2^p tiles wide and high suffice?

            # X has twice as many chunks as tiles, then halved since this is a
            # radius
            xradius = 2**p
            # Y has 4 times as many chunks as tiles, then halved since this is
            # a radius
            yradius = 2*2**p
            # The +32 on the y bounds is because chunks are very tall, and in
            # rare cases when the bottom of the map is close to a border, it
            # could get cut off
            if xradius >= bounds.maxcol and -xradius <= bounds.mincol and \
                    yradius >= bounds.maxrow + 32 and -yradius <= bounds.minrow:
                break
        self.treedepth = p
        self.xradius = xradius
        self.yradius = yradius

    def _rearrange_tiles(self):
        """If the target size of the tree is not the same as the existing size
        on disk, do some re-arranging

        """
        try:
            curdepth = self.config['zoomLevels']
        except KeyError:
            return

        if curdepth == 1:
            # Skip a depth 1 tree. A depth 1 tree pretty much can't happen, so
            # when we detect this it usually means the tree is actually empty
            return
        logging.debug("Current tree depth for %s is reportedly %s. Target tree depth is %s",
                self.options['name'],
                curdepth, self.treedepth)
        if self.treedepth != curdepth:
            if self.treedepth > curdepth:
                logging.warning("Your map seems to have expanded beyond its previous bounds.")
                logging.warning( "Doing some tile re-arrangements... just a sec...")
                for _ in xrange(self.treedepth-curdepth):
                    self._increase_depth()
            elif self.treedepth < curdepth:
                logging.warning("Your map seems to have shrunk. Did you delete some chunks? No problem. Re-arranging tiles, just a sec...")
                for _ in xrange(curdepth - self.treedepth):
                    self._decrease_depth()
                logging.info(
                        "There done. I'm switching to --check-tiles mode for "
                        "this one render. This will make sure any old tiles that "
                        "should no longer exist are deleted.")
                self.options['renderchecks'] = 1

    def _increase_depth(self):
        """Moves existing tiles into place for a larger tree"""
        getpath = functools.partial(os.path.join, self.outputdir)

        # At top level of the tree:
        # quadrant 0 is now 0/3
        # 1 is now 1/2
        # 2 is now 2/1
        # 3 is now 3/0
        # then all that needs to be done is to regenerate the new top level

        def rollback_mkdir(dnum):
            p = getpath("new" + str(dnum))
            if os.path.exists(p):
                os.rmdir(p)

        def rollback_filerename(dnum):
            newnum = (3,2,1,0)[dnum]
            qimg = getpath("new%d/%d.%s" % (dnum, newnum, self.imgextension))
            qdir = getpath("new%d/%d" % (dnum, newnum))

            if os.path.exists(qimg):
                os.rename(qimg, getpath("%d.%s" % (dnum, self.imgextension)))
            if os.path.exists(qdir):
                os.rename(qdir, getpath(str(dnum)))

        def rollback_dirrename(dnum):
            os.rename(getpath(str(dnum)), getpath("new" + str(dnum)))

        for dirnum in range(4):
            newnum = (3,2,1,0)[dirnum]

            newdir = "new" + str(dirnum)
            newdirpath = getpath(newdir)

            files = [str(dirnum)+"."+self.imgextension, str(dirnum)]
            newfiles = [str(newnum)+"."+self.imgextension, str(newnum)]

            try:
                try:
                    os.mkdir(newdirpath)
                    try:
                        for f, newf in zip(files, newfiles):
                            p = getpath(f)
                            if os.path.exists(p):
                                os.rename(p, getpath(newdir, newf))
                    except:
                        rollback_filerename(dirnum)
                        raise
                except:
                    rollback_mkdir(dirnum)
                    raise
                os.rename(newdirpath, getpath(str(dirnum)))
            except:
                logging.warning("Overviewer was interrupted during tile "
                                "re-arrangement.")
                logging.warning("Rolling back changes...")
                # Moonwalk the fuck out of here
                for lastdir in range(dirnum - 1, -1, -1):
                    rollback_dirrename(lastdir)
                    rollback_filerename(lastdir)
                    rollback_mkdir(lastdir)
                raise

    def _decrease_depth(self):
        """If the map size decreases, or perhaps the user has a depth override
        in effect, re-arrange existing tiles for a smaller tree"""
        getpath = functools.partial(os.path.join, self.outputdir)

        # quadrant 0/3 goes to 0
        # 1/2 goes to 1
        # 2/1 goes to 2
        # 3/0 goes to 3
        # Just worry about the directories here, the files at the top two
        # levels are cheap enough to replace
        if os.path.exists(getpath("0", "3")):
            os.rename(getpath("0", "3"), getpath("new0"))
            shutil.rmtree(getpath("0"))
            os.rename(getpath("new0"), getpath("0"))

        if os.path.exists(getpath("1", "2")):
            os.rename(getpath("1", "2"), getpath("new1"))
            shutil.rmtree(getpath("1"))
            os.rename(getpath("new1"), getpath("1"))

        if os.path.exists(getpath("2", "1")):
            os.rename(getpath("2", "1"), getpath("new2"))
            shutil.rmtree(getpath("2"))
            os.rename(getpath("new2"), getpath("2"))

        if os.path.exists(getpath("3", "0")):
            os.rename(getpath("3", "0"), getpath("new3"))
            shutil.rmtree(getpath("3"))
            os.rename(getpath("new3"), getpath("3"))

        # Delete the files in the top directory to make sure they get re-created.
        files = [str(num)+"."+self.imgextension for num in xrange(4)] + ["base." + self.imgextension]
        for f in files:
            try:
                os.unlink(getpath(f))
            except OSError, e:
                # Ignore file doesn't exist errors
                if e.errno != errno.ENOENT:
                    raise

    def _chunk_scan(self):
        """Scans the chunks of this TileSet's world to determine which
        render-tiles need rendering. Returns a RendertileSet object.

        For rendercheck mode 0: only compares chunk mtimes against last render
        time of the map, and marks tiles as dirty if any chunk has a greater
        mtime than the last render time.

        For rendercheck modes 1 and 2: marks every tile in the tileset
        unconditionally, does not check any mtimes.

        As a side-effect, the scan sets self.max_chunk_mtime to the max of all
        the chunks' mtimes

        """
        # See note at the top of this file about the rendercheck modes for an
        # explanation of what this method does in different situations.

        # Local vars for slightly faster lookups
        depth = self.treedepth
        xradius = self.xradius
        yradius = self.yradius

        dirty = RendertileSet(depth)

        chunkcount = 0
        stime = time.time()

        rendercheck = self.options['renderchecks']
        markall = rendercheck in (1,2)

        rerender_prob = self.options['rerenderprob']

        last_rendertime = self.last_rendertime

        max_chunk_mtime = 0


        # For each chunk, do this:
        #   For each tile that the chunk touches, do this:
        #       Compare the last modified time of the chunk and tile. If the
        #       tile is older, mark it in a RendertileSet object as dirty.


        for chunkx, chunkz, chunkmtime in self.regionset.iterate_chunks() if (markall or platform.system() == 'Windows') else self.regionset.iterate_newer_chunks(last_rendertime): 
            chunkcount += 1

            if chunkmtime > max_chunk_mtime:
                max_chunk_mtime = chunkmtime

            # Convert to diagonal coordinates
            chunkcol, chunkrow = convert_coords(chunkx, chunkz)

            for c, r in get_tiles_by_chunk(chunkcol, chunkrow):

                # Make sure the tile is in the boundary we're rendering.
                # This can happen when rendering at lower treedepth than
                # can contain the entire map, but shouldn't happen if the
                # treedepth is correctly calculated.
                if (
                        c < -xradius or
                        c >= xradius or
                        r < -yradius or
                        r >= yradius
                        ):
                    continue

                # Computes the path in the quadtree from the col,row coordinates
                tile = RenderTile.compute_path(c, r, depth)

                if markall:
                    # markall mode: Skip all other checks, mark tiles
                    # as dirty unconditionally
                    dirty.add(tile.path)
                    continue

                # Check if this tile has already been marked dirty. If so,
                # no need to do any of the below.
                if dirty.query_path(tile.path):
                    continue

                # Stochastic check. Since we're scanning by chunks and not
                # by tiles, and the tiles get checked multiple times for
                # each chunk, this is only an approximation. The given
                # probability is for a particular tile that needs
                # rendering, but since a tile gets touched up to 32 times
                # (once for each chunk in it), divide the probability by
                # 32.
                if rerender_prob and rerender_prob/32 > random.random():
                    dirty.add(tile.path)
                    continue

                # Check mtimes and conditionally add tile to the set
                if chunkmtime > last_rendertime:
                    dirty.add(tile.path)

        t = int(time.time()-stime)
        logging.debug("Finished chunk scan for %s. %s chunks scanned in %s second%s",
                self.options['name'], chunkcount, t,
                "s" if t != 1 else "")

        self.max_chunk_mtime = max_chunk_mtime
        return dirty

    def __str__(self):
        return "<TileSet for %s>" % os.path.basename(self.outputdir)

    def _render_compositetile(self, dest, name):
        """
        Renders a tile at os.path.join(dest, name)+".ext" by taking tiles from
        os.path.join(dest, name, "{0,1,2,3}.png")

        If name is "base" then render tile at os.path.join(dest, "base.png") by
        taking tiles from os.path.join(dest, "{0,1,2,3}.png")
        """
        imgformat = self.imgextension
        imgpath = os.path.join(dest, name) + "." + imgformat

        if name == "base":
            # Special case for the base tile. Its children are in the same
            # directory instead of in a sub-directory
            quadPath = [
                    ((0,0),os.path.join(dest, "0." + imgformat)),
                    ((192,0),os.path.join(dest, "1." + imgformat)),
                    ((0, 192),os.path.join(dest, "2." + imgformat)),
                    ((192,192),os.path.join(dest, "3." + imgformat)),
                    ]
        else:
            quadPath = [
                    ((0,0),os.path.join(dest, name, "0." + imgformat)),
                    ((192,0),os.path.join(dest, name, "1." + imgformat)),
                    ((0, 192),os.path.join(dest, name, "2." + imgformat)),
                    ((192,192),os.path.join(dest, name, "3." + imgformat)),
                    ]

        # Check each of the 4 child tiles, getting their existance and mtime
        # infomation. Also keep track of the max mtime of all children
        max_mtime = 0
        quadPath_filtered = []
        for path in quadPath:
            try:
                quad_mtime = os.stat(path[1])[stat.ST_MTIME]
            except OSError:
                # This tile doesn't exist or some other error with the stat
                # call. Move on.
                continue
            # The tile exists, so we need to use it in our rendering of this
            # composite tile
            quadPath_filtered.append(path)
            if quad_mtime > max_mtime:
                max_mtime = quad_mtime

        # If no children exist, delete this tile
        if not quadPath_filtered:
            try:
                os.unlink(imgpath)
            except OSError, e:
                # Ignore errors if it's "file doesn't exist"
                if e.errno != errno.ENOENT:
                    raise
            logging.warning("Tile %s was requested for render, but no children were found! This is probably a bug", imgpath)
            return

        #logging.debug("writing out compositetile {0}".format(imgpath))

        # Create the actual image now
        img = Image.new("RGBA", (384, 384), self.options['bgcolor'])
		
        # we'll use paste (NOT alpha_over) for quadtree generation because
        # this is just straight image stitching, not alpha blending

        for path in quadPath_filtered:
            try:
                #quad = Image.open(path[1]).resize((192,192), Image.ANTIALIAS)
                src = Image.open(path[1])
                # optimizeimg may have converted them to a palette image in the meantime
                if src.mode != "RGB" and src.mode != "RGBA":
                    src = src.convert("RGBA")
                src.load()

                quad = Image.new("RGBA", (192, 192), self.options['bgcolor'])
                resize_half(quad, src)
                img.paste(quad, path[0])
            except Exception, e:
                logging.warning("Couldn't open %s. It may be corrupt. Error was '%s'", path[1], e)
                logging.warning("I'm going to try and delete it. You will need to run the render again and with --check-tiles")
                try:
                    os.unlink(path[1])
                except Exception, e:
                    logging.error("While attempting to delete corrupt image %s, an error was encountered. You will need to delete it yourself. Error was '%s'", path[1], e)

        # Save it
        with FileReplacer(imgpath, capabilities=self.fs_caps) as tmppath:
            if imgformat == 'jpg':
                img.save(tmppath, "jpeg", quality=self.options['imgquality'], subsampling=0)
            else: # png
                img.save(tmppath, "png")

            if self.options['optimizeimg']:
                optimize_image(tmppath, imgformat, self.options['optimizeimg'])

            try:
                os.utime(tmppath, (max_mtime, max_mtime))
            except OSError, e:
                # Ignore errno ENOENT: file does not exist. Due to a race
                # condition, two processes could conceivably try and update
                # the same temp file at the same time
                if e.errno != errno.ENOENT:
                    raise

    def _render_rendertile(self, tile):
        """Renders the given render-tile.

        This function is called from the public do_work() method in the child
        process. The tile is assumed to need rendering and is rendered
        unconditionally.

        The argument is a RenderTile object

        The image is rendered and saved to disk in the place this tileset is
        configured to save images.

        """

        imgpath = tile.get_filepath(self.outputdir, self.imgextension)

        # Calculate which chunks are relevant to this tile
        # This is a list of (col, row, chunkx, chunkz, chunk_mtime)
        chunks = list(get_chunks_by_tile(tile, self.regionset))

        if not chunks:
            # No chunks were found in this tile
            logging.warning("%s was requested for render, but no chunks found! This may be a bug", tile)
            try:
                os.unlink(imgpath)
            except OSError, e:
                # ignore only if the error was "file not found"
                if e.errno != errno.ENOENT:
                    raise
            else:
                logging.debug("%s deleted", tile)
            return

        # Create the directory if not exists
        dirdest = os.path.dirname(imgpath)
        if not os.path.exists(dirdest):
            try:
                os.makedirs(dirdest)
            except OSError, e:
                # Ignore errno EEXIST: file exists. Due to a race condition,
                # two processes could conceivably try and create the same
                # directory at the same time
                if e.errno != errno.EEXIST:
                    raise

        #logging.debug("writing out worldtile {0}".format(imgpath))

        # Compile this image
        tileimg = Image.new("RGBA", (384, 384), self.options['bgcolor'])

        colstart = tile.col
        rowstart = tile.row
        # col colstart will get drawn on the image starting at x coordinates -(384/2)
        # row rowstart will get drawn on the image starting at y coordinates -(192/2)
        max_chunk_mtime = 0
        for col, row, chunkx, chunky, chunkz, chunk_mtime in chunks:
            xpos = -192 + (col-colstart)*192
            ypos = -96 + (row-rowstart)*96 + (16-1 - chunky)*192

            if chunk_mtime > max_chunk_mtime:
                max_chunk_mtime = chunk_mtime

            # draw the chunk!
            try:
                c_overviewer.render_loop(self.world, self.regionset, chunkx, chunky,
                        chunkz, tileimg, xpos, ypos,
                        self.options['rendermode'], self.textures)
            except nbt.CorruptionError:
                # A warning and traceback was already printed by world.py's
                # get_chunk()
                logging.debug("Skipping the render of corrupt chunk at %s,%s and moving on.", chunkx, chunkz)
            except Exception, e:
                logging.error("Could not render chunk %s,%s for some reason. This is likely a render primitive option error.", chunkx, chunkz)
                logging.error("Full error was:", exc_info=1)
                sys.exit(1)

            ## Semi-handy routine for debugging the drawing routine
            ## Draw the outline of the top of the chunk
            #import ImageDraw
            #draw = ImageDraw.Draw(tileimg)
            ## Draw top outline
            #draw.line([(192,0), (384,96)], fill='red')
            #draw.line([(192,0), (0,96)], fill='red')
            #draw.line([(0,96), (192,192)], fill='red')
            #draw.line([(384,96), (192,192)], fill='red')
            ## Draw side outline
            #draw.line([(0,96),(0,96+192)], fill='red')
            #draw.line([(384,96),(384,96+192)], fill='red')
            ## Which chunk this is:
            #draw.text((96,48), "C: %s,%s" % (chunkx, chunkz), fill='red')
            #draw.text((96,96), "c,r: %s,%s" % (col, row), fill='red')

        # Save them
        with FileReplacer(imgpath, capabilities=self.fs_caps) as tmppath:
            if self.imgextension == 'jpg':
                tileimg.save(tmppath, "jpeg", quality=self.options['imgquality'], subsampling=0)
            else: # png
                tileimg.save(tmppath, "png")

            if self.options['optimizeimg']:
                optimize_image(tmppath, self.imgextension, self.options['optimizeimg'])
            
            os.utime(tmppath, (max_chunk_mtime, max_chunk_mtime))

    def _iterate_and_check_tiles(self, path):
        """A generator function over all tiles that should exist in the subtree
        identified by path. This yields, in order, all tiles that need
        rendering in a post-traversal order, including this node itself.

        This method takes one parameter:

        path
            The path of a tile that should exist


        This method yields tuples in this form:
            (path, mtime, needs_rendering)
        path
            is the path tuple of the tile that needs rendering
        mtime
            if the tile does not need rendering, the parent call determines if
            it should render itself by comparing its own mtime to the child
            mtimes. This should be set to the tile's mtime in the event that
            the tile does not need rendering, or None otherwise.
        needs_rendering
            is a boolean indicating this tile does in fact need rendering.

        (Since this is a recursive generator, tiles that don't need rendering
        are not propagated all the way out of the recursive stack, but are
        still yielded to the immediate parent because it needs to know its
        childs' mtimes)

        """
        if len(path) == self.treedepth:
            # Base case: a render-tile.
            # Render this tile if any of its chunks are greater than its mtime
            tileobj = RenderTile.from_path(path)
            imgpath = tileobj.get_filepath(self.outputdir, self.imgextension)
            try:
                tile_mtime = os.stat(imgpath)[stat.ST_MTIME]
            except OSError, e:
                if e.errno != errno.ENOENT:
                    raise
                tile_mtime = 0
            
            try:
                max_chunk_mtime = max(c[5] for c in get_chunks_by_tile(tileobj, self.regionset))
            except ValueError:
                # max got an empty sequence! something went horribly wrong
                logging.warning("tile %s expected contains no chunks! this may be a bug", path)
                max_chunk_mtime = 0

            if tile_mtime > 120 + max_chunk_mtime:
                # If a tile has been modified more recently than any of its
                # chunks, then this could indicate a potential issue with
                # this or future renders.
                logging.warning(
                        "I found a tile with a more recent modification time "
                        "than any of its chunks. This can happen when a tile has "
                        "been modified with an outside program, or by a copy "
                        "utility that doesn't preserve mtimes. Overviewer uses "
                        "the filesystem's mtimes to determine which tiles need "
                        "rendering and which don't, so it's important to "
                        "preserve the mtimes Overviewer sets. Please see our FAQ "
                        "page on docs.overviewer.org or ask us in IRC for more "
                        "information")
                logging.warning("Tile was: %s", imgpath)

            if max_chunk_mtime > tile_mtime or tile_mtime < self.forcerendertime:
                # chunks have a more recent mtime than the tile or the tile has
                # an older mtime than the forcerendertime from an interrupted
                # render. Render the tile.
                yield (path, None, True)
            else:
                # This doesn't need rendering. Return mtime to parent in case
                # its mtime is less, indicating the parent DOES need a render
                yield path, max_chunk_mtime, False

        else:
            # A composite-tile.
            render_me = False
            max_child_mtime = 0

            # First, recurse to each of our children
            for childnum in xrange(4):
                childpath = path + (childnum,)

                # Check if this sub-tree should actually exist, so that we only
                # end up checking tiles that actually exist
                if not self.dirtytree.query_path(childpath):
                    # Here check if it *does* exist, and if so, nuke it.
                    self._nuke_path(childpath)
                    continue

                for child_path, child_mtime, child_needs_rendering in \
                        self._iterate_and_check_tiles(childpath):
                    if len(child_path) == len(path)+1:
                        # Do these checks for our immediate children
                        if child_needs_rendering:
                            render_me = True
                        elif child_mtime > max_child_mtime:
                            max_child_mtime = child_mtime

                    # Pass this on up and out.
                    # Optimization: if it does not need rendering, we don't
                    # need to pass it any further. A tile that doesn't need
                    # rendering is only relevant to its immediate parent, and
                    # only for its mtime information.
                    if child_needs_rendering:
                        yield child_path, child_mtime, child_needs_rendering

            # Now that we're done with our children and descendents, see if
            # this tile needs rendering
            if render_me:
                # yes. yes we do. This is set when one of our children needs
                # rendering
                yield path, None, True
            else:
                # Check this tile's mtime
                imgpath = os.path.join(self.outputdir, *(str(x) for x in path))
                imgpath += "." + self.imgextension
                logging.debug("Testing mtime for composite-tile %s", imgpath)
                try:
                    tile_mtime = os.stat(imgpath)[stat.ST_MTIME]
                except OSError, e:
                    if e.errno != errno.ENOENT:
                        raise
                    tile_mtime = 0

                if tile_mtime < max_child_mtime:
                    # If any child was updated more recently than ourself, then
                    # we need rendering
                    yield path, None, True
                else:
                    # Nope.
                    yield path, max_child_mtime, False

    def _nuke_path(self, path):
        """Given a quadtree path, erase it from disk. This is called by
        _iterate_and_check_tiles() as a helper-method.

        """
        if len(path) == self.treedepth:
            # path referrs to a single tile
            tileobj = RenderTile.from_path(path)
            imgpath = tileobj.get_filepath(self.outputdir, self.imgextension)
            if os.path.exists(imgpath):
                # No need to catch ENOENT since this is only called from the
                # master process
                logging.debug("Found an image that shouldn't exist. Deleting it: %s", imgpath)
                os.remove(imgpath)
        else:
            # path referrs to a composite tile, and by extension a directory
            dirpath = os.path.join(self.outputdir, *(str(x) for x in path))
            imgpath = dirpath + "." + self.imgextension
            if os.path.exists(imgpath):
                logging.debug("Found an image that shouldn't exist. Deleting it: %s", imgpath)
                os.remove(imgpath)
            if os.path.exists(dirpath):
                logging.debug("Found a subtree that shouldn't exist. Deleting it: %s", dirpath)
                shutil.rmtree(dirpath)

##
## Functions for converting (x, z) to (col, row) and back
##

def convert_coords(chunkx, chunkz):
    """Takes a coordinate (chunkx, chunkz) where chunkx and chunkz are
    in the chunk coordinate system, and figures out the row and column
    in the image each one should be. Returns (col, row)."""

    # columns are determined by the sum of the chunk coords, rows are the
    # difference
    # change this function, and you MUST change unconvert_coords
    return (chunkx + chunkz, chunkz - chunkx)

def unconvert_coords(col, row):
    """Undoes what convert_coords does. Returns (chunkx, chunkz)."""

    # col + row = chunkz + chunkz => (col + row)/2 = chunkz
    # col - row = chunkx + chunkx => (col - row)/2 = chunkx
    return ((col - row) / 2, (col + row) / 2)

######################
# The following two functions define the mapping from chunks to tiles and back.
# The mapping from chunks to tiles (get_tiles_by_chunk()) is used during the
# chunk scan to determine which tiles need updating, while the mapping from a
# tile to chunks (get_chunks_by_tile()) is used by the tile rendering routines
# to get which chunks are needed.
def get_tiles_by_chunk(chunkcol, chunkrow):
    """For the given chunk, returns an iterator over Render Tiles that this
    chunk touches.  Iterates over (tilecol, tilerow)

    """
    # find tile coordinates. Remember tiles are identified by the
    # address of the chunk in their upper left corner.
    tilecol = chunkcol - chunkcol % 2
    tilerow = chunkrow - chunkrow % 4

    # If this chunk is in an /even/ column, then it spans two tiles.
    if chunkcol % 2 == 0:
        colrange = (tilecol-2, tilecol)
    else:
        colrange = (tilecol,)

    # If this chunk is in a row divisible by 4, then it touches the
    # tile above it as well. Also touches the next 4 tiles down (16
    # rows)
    if chunkrow % 4 == 0:
        rowrange = xrange(tilerow-4, tilerow+32+1, 4)
    else:
        rowrange = xrange(tilerow, tilerow+32+1, 4)

    return product(colrange, rowrange)

def get_chunks_by_tile(tile, regionset):
    """Get chunk sections that are relevant to the given render-tile. Only
    returns chunk sections that are in chunks that actually exist according to
    the given regionset object. (Does not check to see if the chunk section
    itself within the chunk exists)

    This function is expected to return the chunk sections in the correct order
    for rendering, i.e. back to front.

    Returns an iterator over chunks tuples where each item is
    (col, row, chunkx, chunky, chunkz, mtime)
    """

    # This is not a documented usage of this function and is used only for
    # debugging
    if regionset is None:
        get_mtime = lambda x,y: True
    else:
        get_mtime = regionset.get_chunk_mtime

    # Each tile has two even columns and an odd column of chunks.

    # First do the odd. For each chunk in the tile's odd column the tile
    # "passes through" three chunk sections.
    oddcol_sections = []
    for i, y in enumerate(reversed(xrange(16))):
        for row in xrange(tile.row + 3 - i*2, tile.row - 2 - i*2, -2):
            oddcol_sections.append((tile.col+1, row, y))

    evencol_sections = []
    for i, y in enumerate(reversed(xrange(16))):
        for row in xrange(tile.row + 4 - i*2, tile.row - 3 - i*2, -2):
            evencol_sections.append((tile.col+2, row, y))
            evencol_sections.append((tile.col, row, y))

    eveniter = reversed(evencol_sections)
    odditer = reversed(oddcol_sections)

    # There are 4 rows of chunk sections per Y value on even columns, but 3
    # rows on odd columns. This iteration order yields them in back-to-front
    # order appropriate for rendering
    for col, row, y in roundrobin((
            eveniter,eveniter,
            odditer,
            eveniter,eveniter,
            odditer,
            eveniter,eveniter,
            odditer,
            eveniter,eveniter,
            )):
        chunkx, chunkz = unconvert_coords(col, row)
        mtime = get_mtime(chunkx, chunkz)
        if mtime:
            yield (col, row, chunkx, y, chunkz, mtime)

class RendertileSet(object):
    """This object holds a set of render-tiles using a quadtree data structure.
    It is typically used to hold tiles that need rendering. This implementation
    collapses subtrees that are completely in or out of the set to save memory.

    An instance of this class holds a full tree.

    The instance knows its "level", which corresponds to the zoom level where 1
    is the inner-most (most zoomed in) tiles.

    Instances hold the state of their children (in or out of the set). Leaf
    nodes are images and do not physically exist in the tree as objects, but
    are represented as booleans held by the objects at the second-to-last
    level; level 1 nodes keep track of leaf image state. Level 2 nodes keep
    track of level 1 state, and so forth.

    """
    __slots__ = ("depth", "children", "num_tiles", "num_tiles_all")
    def __init__(self, depth):
        """Initialize a new tree with the specified depth. This actually
        initializes a node, which is the root of a subtree, with `depth` levels
        beneath it.

        """
        # Stores the depth of the tree according to this node. This is not the
        # depth of this node, but rather the number of levels below this node
        # (including this node).
        self.depth = depth

        # the self.children array holds the 4 children of this node. This
        # follows the same quadtree convention as elsewhere: children 0, 1, 2,
        # 3 are the upper-left, upper-right, lower-left, and lower-right
        # respectively
        # Values are:
        # False
        #   All children down this subtree are not in the set
        # True
        #   All children down this subtree are in the set
        # An array of the same format
        #   The array defines which children down that subtree are in the set
        # A node with depth=1 cannot have a RendertileSet instance in its
        # children since its children are leaves, representing images, not more
        # tree
        self.children = [False] * 4

        self.num_tiles     = 0
        self.num_tiles_all = 0

    def add(self, path):
        """Marks the requested leaf node as in this set

        Path is an iterable of integers representing the path to the leaf node
        that is to be added to the set

        """
        path = list(path)
        assert len(path) == self.depth

        if self.num_tiles == 0:
            # The first child is being added. A root composite tile will be
            # rendered.
            self.num_tiles_all += 1

        self._add_helper(self.children, list(reversed(path)))

    def _add_helper(self, children, path):
        """Recursive helper for add()
        """

        childnum = path.pop()

        if path:
            # We are not at the leaf, recurse.

            if children[childnum] == True:
                # The child is already in the tree.
                return
            elif children[childnum] == False:
                # Expand all-false.
                children[childnum] = [False]*4

                # This also means an additional composite tile.
                self.num_tiles_all += 1

            self._add_helper(children[childnum], path)

            if children[childnum] == [True]*4:
                # Collapse all-true.
                children[childnum] = True

        else:
            # We are at the leaf.
            if not children[childnum]:
                self.num_tiles     += 1
                self.num_tiles_all += 1

            children[childnum] = True

    def __iter__(self):
        return self.iterate()

    def iterate(self, level=None, robin=False, offset=(0,0)):
        """Returns an iterator over every tile in this set. Each item yielded
        is a sequence of integers representing the quadtree path to the tiles
        in the set. Yielded sequences are of length self.depth.

        If level is None, iterates over tiles of the highest level, i.e.
        worldtiles. If level is a value between 1 and the depth of this tree,
        this method iterates over tiles at that level. Zoom level 1 is zoomed
        all the way out, zoom level `depth` is all the way in.

        In other words, specifying level causes the tree to be iterated as if
        it was only that depth.

        If the `robin` parameter is True, recurses to the four top-level
        subtrees simultaneously in a round-robin manner.

        """
        if level is None:
            todepth = 1
        else:
            if not (level > 0 and level <= self.depth):
                raise ValueError("Level parameter must be between 1 and %s" % self.depth)
            todepth = self.depth - level + 1

        return (tuple(path) for path in self._iterate_helper([], self.children, self.depth, onlydepth=todepth, robin=robin, offset=offset))

    def posttraversal(self, robin=False, offset=(0,0)):
        """Returns an iterator over tile paths for every tile in the
        set, including the explictly marked render-tiles, as well as the
        implicitly marked ancestors of those render-tiles. Returns in
        post-traversal order, so that tiles with dependencies will always be
        yielded after their dependencies.

        If the `robin` parameter is True, recurses to the four top-level
        subtrees simultaneously in a round-robin manner.

        """
        return (tuple(path) for path in self._iterate_helper([], self.children, self.depth, robin=robin, offset=offset))

    def _iterate_helper(self, path, children, depth, onlydepth=None, robin=False, offset=(0,0)):
        """Returns an iterator over tile paths for every tile in the set."""

        # A variant of children with a collapsed False/True expanded to a list.
        children_list = [children] * 4 if isinstance(children, bool) else children

        targetdepth = 1 if onlydepth is None else onlydepth

        if depth == targetdepth:
            # Base case
            for (childnum, child), _ in distance_sort(enumerate(children_list), offset):
                if child:
                    yield path + [childnum]
        else:
            gens = []
            for (childnum_, child), childoffset_ in distance_sort(enumerate(children_list), offset):
                if child:
                    def go(childnum, childoffset):
                        for p in self._iterate_helper(path + [childnum], children_list[childnum], depth-1, onlydepth=onlydepth, offset=childoffset):
                            yield p
                    gens.append(go(childnum_, childoffset_))

            for p in roundrobin(gens) if robin else chain(*gens):
                yield p

        if onlydepth is None and any(children_list):
            yield path

    def query_path(self, path):
        """Queries for the state of the given tile in the tree.

        Returns True for items in the set, False otherwise. Works for
        rendertiles as well as upper tiles (which are True if they have a
        descendent that is in the set)

        """
        # Traverse the tree down the given path. If the tree has been
        # collapsed, then just return the stored boolean. Otherwise, if we find
        # the specific tree node requested, return its state using the
        # __nonzero__ call.
        treenode = self.children
        for pathelement in path:
            treenode = treenode[pathelement]
            if isinstance(treenode, bool):
                return treenode

        # If the method has not returned at this point, treenode is the
        # requested node, but it is an inner node. That will only happen if one
        # or more of the children down the tree are True.
        return True

    def __nonzero__(self):
        """Returns the boolean context of this particular node. If any
        descendent of this node is True return True. Otherwise, False.

        """
        # Any children that are True or are a list evaluate to True.
        return any(self.children)

    def count(self):
        """Returns the total number of render-tiles in this set.

        """
        # XXX There seems to be something wrong with the num_tiles calculation.
        # Calculate the number of tiles by iteration and emit a warning if it
        # does not match.
        from itertools import imap
        num = sum(imap(lambda _: 1, self.iterate()))
        if num != self.num_tiles:
            logging.error("Please report this to the developers: RendertileSet num_tiles=%r, count=%r, children=%r", self.num_tiles, num, self.children)
        return num

    def count_all(self):
        """Returns the total number of render-tiles plus implicitly marked
        upper-tiles in this set

        """
        # XXX There seems to be something wrong with the num_tiles calculation.
        # Calculate the number of tiles by iteration and emit a warning if it
        # does not match.
        from itertools import imap
        num = sum(imap(lambda _: 1, self.posttraversal()))
        if num != self.num_tiles_all:
            logging.error("Please report this to the developers: RendertileSet num_tiles_all=%r, count_all=%r, children=%r", self.num_tiles, num, self.children)
        return num

def distance_sort(children, (off_x, off_y)):
    order = []
    for child, (dx, dy) in izip(children, [(-1,-1), (1,-1), (-1,1), (1,1)]):
        x = off_x*2 + dx
        y = off_y*2 + dy
        order.append((child, (x,y)))

    return sorted(order, key=lambda (_, (x,y)): x*x + y*y)

class RenderTile(object):
    """A simple container class that represents a single render-tile.

    A render-tile is a tile that is rendered, not a tile composed of other
    tiles (composite-tile).

    """
    __slots__ = ("col", "row", "path")
    def __init__(self, col, row, path):
        """Initialize the tile obj with the given parameters. It's probably
        better to use one of the other constructors though

        """
        self.col = col
        self.row = row
        self.path = tuple(path)

    def __repr__(self):
        return "%s(%r,%r,%r)" % (self.__class__.__name__, self.col, self.row, self.path)

    def __eq__(self,other):
        return self.col == other.col and self.row == other.row and tuple(self.path) == tuple(other.path)

    def __ne__(self, other):
        return not self == other

    # To support pickling
    def __getstate__(self):
        return self.col, self.row, self.path
    def __setstate__(self, state):
        self.__init__(*state)

    def get_filepath(self, tiledir, imgformat):
        """Returns the path to this file given the directory to the tiles

        """
        # os.path.join would be the proper way to do this path concatenation,
        # but it is surprisingly slow, probably because it checks each path
        # element if it begins with a slash. Since we know these components are
        # all relative, just concatinate with os.path.sep
        pathcomponents = [tiledir]
        pathcomponents.extend(str(x) for x in self.path)
        path = os.path.sep.join(pathcomponents)
        imgpath = ".".join((path, imgformat))
        return imgpath


    @classmethod
    def from_path(cls, path):
        """Constructor that takes a path and computes the col,row address of
        the tile and constructs a new tile object.

        """
        path = tuple(path)

        depth = len(path)

        # Radius of the world in chunk cols/rows
        # (Diameter in X is 2**depth, divided by 2 for a radius, multiplied by
        # 2 for 2 chunks per tile. Similarly for Y)
        xradius = 2**depth
        yradius = 2*2**depth

        col = -xradius
        row = -yradius
        xsize = xradius
        ysize = yradius

        for p in path:
            if p in (1,3):
                col += xsize
            if p in (2,3):
                row += ysize
            xsize //= 2
            ysize //= 2

        return cls(col, row, path)

    @classmethod
    def compute_path(cls, col, row, depth):
        """Constructor that takes a col,row of a tile and computes the path.

        """
        assert col % 2 == 0
        assert row % 4 == 0

        xradius = 2**depth
        yradius = 2*2**depth

        colbounds = [-xradius, xradius]
        rowbounds = [-yradius, yradius]

        path = []

        for level in xrange(depth):
            # Strategy: Find the midpoint of this level, and determine which
            # quadrant this row/col is in. Then set the bounds to that level
            # and repeat

            xmid = (colbounds[1] + colbounds[0]) // 2
            ymid = (rowbounds[1] + rowbounds[0]) // 2

            if col < xmid:
                if row < ymid:
                    path.append(0)
                    colbounds[1] = xmid
                    rowbounds[1] = ymid
                else:
                    path.append(2)
                    colbounds[1] = xmid
                    rowbounds[0] = ymid
            else:
                if row < ymid:
                    path.append(1)
                    colbounds[0] = xmid
                    rowbounds[1] = ymid
                else:
                    path.append(3)
                    colbounds[0] = xmid
                    rowbounds[0] = ymid

        return cls(col, row, path)
