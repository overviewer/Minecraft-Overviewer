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

import logging

from PIL import Image

"""
This module has an alpha-over function that is used throughout
Overviewer. It defaults to the PIL paste function when the custom
alpha-over extension cannot be found.
"""

extension_alpha_over = None
try:
    from _composite import alpha_over as _extension_alpha_over
    extension_alpha_over = _extension_alpha_over
except ImportError:
    pass

def alpha_over(dest, src, pos_or_rect=(0, 0), mask=None):
    """Composite src over dest, using mask as the alpha channel (if
    given), otherwise using src's alpha channel. pos_or_rect can
    either be a position or a rectangle, specifying where on dest to
    put src. Falls back to dest.paste() if the alpha_over extension
    can't be found."""
    if mask == None:
        mask = src
    
    global extension_alpha_over
    if extension_alpha_over != None:
        # extension ALWAYS expects rects, so convert if needed
        if len(pos_or_rect) == 2:
            pos_or_rect = (pos_or_rect[0], pos_or_rect[1], src.size[0], src.size[1])
        extension_alpha_over(dest, src, pos_or_rect, mask)
    else:
        # fallback
        dest.paste(src, pos_or_rect, mask)

