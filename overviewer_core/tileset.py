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

iterate_work_items(phase)
    Takes a phase number (a non-negative integer). This method should return an
    iterator over work items. The work items can be any pickelable object; they
    are treated as opaque by the Dispatcher. The work item objects are passed
    back in to the do_work() method (perhaps in a different, identically
    configured instance)

do_work(workobj)
    Does the work for a given work object. This method is not expected to
    return anything, so the results of its work should be reflected on the
    filesystem or by sending signals.


"""

class TileSet(object):
    """The TileSet object manages the work required to produce a set of tiles
    on disk. It calculates the work that needs to be done and tells the
    dipatcher (through the Worker interface) this information. The Dispatcher
    then tells this object when and where to do the work of rendering the tiles.

    """

    def __init__(self, regionsetobj, assetmanagerobj, options, outputdir):
        """Construct a new TileSet object with the given configuration options
        dictionary.

        options is a dictionary of configuration parameters (strings mapping to
        values) that are interpreted by the rendering engine.

        regionsetobj is the RegionSet object that is used to render the tiles.

        assetmanagerobj is the AssetManager object that represents the
        destination directory where we'll put our tiles.

        outputdir is the absolute path to the tile output directory where the
        tiles are saved.
        TODO: This should probably be relative to the asset manager's output
        directory to avoid redundancy.

        
        Current valid options for the options dictionary are shown below. All
        the options must be specified unless they are not relevant. If the
        given options do not conform to the specifications, behavior is
        undefined (this class does not do any error checking and assumes items
        are given in the correct form).

        bgcolor
            A hex string specifying the background color for jpeg output.
            e.g.: "#1A1A1A". Not relevant unless rendering jpeg.

        forcerender
            True to indicate every tile should be rendered regardless of any
            mtime checks. False otherwise.

        imgformat
            A string indicating the output format. Must be one of 'png' or
            'jpeg'

        imgquality
            An integer 1-100 indicating the quality of the jpeg output. Only
            relevant in jpeg mode.

        optimizeimg
            an integer indiating optimizations to perform on png outputs. 0
            indicates no optimizations. Only relevant in png mode.
            1 indicates pngcrush is run on all output images
            2 indicates pngcrush and advdef are run on all output images with advdef -z2
            3 indicates pngcrush and advdef are run on all output images with advdef -z4

        rendermode
            Perhaps the most important/relevant option: a string indicating the
            render mode to render. This rendermode must have already been
            registered with the C extension module. 

        rerender_prob
            A floating point number between 0 and 1 indicating the probability
            that a tile which is not marked for render by any mtime checks will
            be rendered anyways. 0 disables this option.

        """
        pass

    def do_preprocessing(self):
        """For the preprocessing step of the Worker interface, this does the
        chunk scan and stores the resulting tree as a private instance
        attribute for later use in iterate_work_items()

        """
        pass

    def get_num_phases(self):
        """Returns the number of levels in the quadtree, which is equal to the
        number of phases of work that need to be done.

        """
        pass

    def iterate_work_items(self, phase):
        """Iterates over the dirty tiles in the tree at level depth-phase. So
        the first phase iterates over the deepest tiles in the tree, and works
        its way to the root node of the tree.

        """
        pass

    def do_work(self, tileobj):
        """Renders the given tile.

        """

    def get_persistent_data(self):
        """Returns a dictionary representing the persistent data of this
        TileSet. Typically this is called by AssetManager

        """
        pass
