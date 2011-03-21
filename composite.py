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

from c_overviewer import alpha_over as extension_alpha_over

def alpha_over(dest, src, pos_or_rect=(0, 0), mask=None):
    """Composite src over dest, using mask as the alpha channel (if
    given), otherwise using src's alpha channel. pos_or_rect can
    either be a position or a rectangle, specifying where on dest to
    put src. Falls back to dest.paste() if the alpha_over extension
    can't be found."""
    if mask is None:
        mask = src
    
    global extension_alpha_over
    return extension_alpha_over(dest, src, pos_or_rect, mask)
