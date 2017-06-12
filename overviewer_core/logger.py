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

import sys
import os
import logging
import platform
import ctypes
from cStringIO import StringIO

# Some cool code for colored logging:
# For background, add 40. For foreground, add 30
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

# Windows colors, taken from WinCon.h
FOREGROUND_BLUE   = 0x01
FOREGROUND_GREEN  = 0x02
FOREGROUND_RED    = 0x04
FOREGROUND_BOLD   = 0x08
FOREGROUND_WHITE  = FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED

BACKGROUND_BLACK  = 0x00
BACKGROUND_BLUE   = 0x10
BACKGROUND_GREEN  = 0x20
BACKGROUND_RED    = 0x40

COLORIZE = {
    #'INFO': WHITe,
    'DEBUG': CYAN,
}
HIGHLIGHT = {
    'CRITICAL': RED,
    'ERROR': RED,
    'WARNING': YELLOW,
}


class WindowsOutputStream(object):
    """A file-like object that proxies sys.stderr and interprets simple ANSI
    escape codes for color, translating them to the appropriate Windows calls.

    """
    def __init__(self, stream=None):
        assert platform.system() == 'Windows'
        self.stream = stream or sys.stderr

        # go go gadget ctypes 
        self.GetStdHandle = ctypes.windll.kernel32.GetStdHandle
        self.SetConsoleTextAttribute = ctypes.windll.kernel32.SetConsoleTextAttribute
        self.STD_OUTPUT_HANDLE = ctypes.c_int(0xFFFFFFF5)
        self.output_handle = self.GetStdHandle(self.STD_OUTPUT_HANDLE)
        if self.output_handle == 0xFFFFFFFF:
            raise Exception("Something failed in WindowsColorFormatter")


        # default is white text on a black background
        self.currentForeground = FOREGROUND_WHITE
        self.currentBackground = BACKGROUND_BLACK
        self.currentBold       = 0

    def updateWinColor(self, Fore=None, Back=None, Bold=False):
        if Fore != None: self.currentForeground = Fore
        if Back != None: self.currentBackground = Back
        if Bold: 
            self.currentBold = FOREGROUND_BOLD
        else:
            self.currentBold = 0

        self.SetConsoleTextAttribute(self.output_handle,
                ctypes.c_int(self.currentForeground | self.currentBackground | self.currentBold))

    def write(self, s):

        msg_strm = StringIO(s) 
    
        while (True):
            c = msg_strm.read(1)
            if c == '': break
            if c == '\033':
                c1 = msg_strm.read(1)
                if c1 != '[': # 
                    sys.stream.write(c + c1)
                    continue
                c2 = msg_strm.read(2)
                if c2 == "0m": # RESET_SEQ
                    self.updateWinColor(Fore=FOREGROUND_WHITE, Back=BACKGROUND_BLACK)

                elif c2 == "1;":
                    color = ""
                    while(True):
                        nc = msg_strm.read(1)
                        if nc == 'm': break
                        color += nc
                    color = int(color) 
                    if (color >= 40): # background
                        color = color - 40
                        if color == BLACK:
                            self.updateWinColor(Back=BACKGROUND_BLACK)
                        if color == RED:
                            self.updateWinColor(Back=BACKGROUND_RED)
                        elif color == GREEN:
                            self.updateWinColor(Back=BACKGROUND_GREEN)
                        elif color == YELLOW:
                            self.updateWinColor(Back=BACKGROUND_RED | BACKGROUND_GREEN)
                        elif color == BLUE:
                            self.updateWinColor(Back=BACKGROUND_BLUE)
                        elif color == MAGENTA:
                            self.updateWinColor(Back=BACKGROUND_RED | BACKGROUND_BLUE)
                        elif color == CYAN:
                            self.updateWinColor(Back=BACKGROUND_GREEN | BACKGROUND_BLUE)
                        elif color == WHITE:
                            self.updateWinColor(Back=BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_BLUE)
                    elif (color >= 30): # foreground
                        color = color - 30
                        if color == BLACK:
                            self.updateWinColor(Fore=FOREGROUND_BLACK)
                        if color == RED:
                            self.updateWinColor(Fore=FOREGROUND_RED)
                        elif color == GREEN:
                            self.updateWinColor(Fore=FOREGROUND_GREEN)
                        elif color == YELLOW:
                            self.updateWinColor(Fore=FOREGROUND_RED | FOREGROUND_GREEN)
                        elif color == BLUE:
                            self.updateWinColor(Fore=FOREGROUND_BLUE)
                        elif color == MAGENTA:
                            self.updateWinColor(Fore=FOREGROUND_RED | FOREGROUND_BLUE)
                        elif color == CYAN:
                            self.updateWinColor(Fore=FOREGROUND_GREEN | FOREGROUND_BLUE)
                        elif color == WHITE:
                            self.updateWinColor(Fore=FOREGROUND_WHITE)

                         
                    
                elif c2 == "1m": # BOLD_SEQ
                    pass
                
            else:
                self.stream.write(c)



    def flush(self):
        self.stream.flush()

class HighlightingFormatter(logging.Formatter):
    """Base class of our custom formatter
    
    """
    datefmt = "%Y-%m-%d %H:%M:%S"
    funcName_len = 15

    def __init__(self, verbose=False):
        if verbose:
            fmtstr = '%(fileandlineno)-18s %(pid)s %(asctime)s ' \
                    '%(levelname)-8s %(message)s'
        else:
            fmtstr = '%(asctime)s ' '%(shortlevelname)-1s%(message)s'

        logging.Formatter.__init__(self, fmtstr, self.datefmt)

    def format(self, record):
        """Add a few extra options to the record

        pid
            The process ID

        fileandlineno
            A combination filename:linenumber string, so it can be justified as
            one entry in a format string.

        funcName
            The function name truncated/padded to a fixed width characters

        shortlevelname
            The level name truncated to 1 character
        
        """

        record.shortlevelname = record.levelname[0] + ' ' 
        if record.levelname == 'INFO': record.shortlevelname = ''

        record.pid = os.getpid()
        record.fileandlineno = "%s:%s" % (record.filename, record.lineno)

        # Set the max length for the funcName field, and left justify
        l = self.funcName_len
        record.funcName = ("%-" + str(l) + 's') % record.funcName[:l]

        return self.highlight(record)

    def highlight(self, record):
        """This method applies any special effects such as colorization. It
        should modify the records in the record object, and should return the
        *formatted line*. This probably involves calling
        logging.Formatter.format()

        Override this in subclasses

        """
        return logging.Formatter.format(self, record)

class DumbFormatter(HighlightingFormatter):
    """Formatter for dumb terminals that don't support color, or log files.
    Prints a bunch of stars before a highlighted line.

    """
    def highlight(self, record):
        if record.levelname in HIGHLIGHT:
            line = logging.Formatter.format(self, record)
            line = "*" * min(79,len(line)) + "\n" + line
            return line
        else:
            return HighlightingFormatter.highlight(self, record)


class ANSIColorFormatter(HighlightingFormatter):
    """Uses ANSI escape sequences to enable GLORIOUS EXTRA-COLOR!

    """
    def highlight(self, record):
        if record.levelname in COLORIZE:
            # Colorize just the levelname
            # left justify again because the color sequence bumps the length up
            # above 8 chars
            levelname_color = COLOR_SEQ % (30 + COLORIZE[record.levelname]) + \
                    "%-8s" % record.levelname + RESET_SEQ
            record.levelname = levelname_color
            return logging.Formatter.format(self, record)

        elif record.levelname in HIGHLIGHT:
            # Colorize the entire line
            line = logging.Formatter.format(self, record)
            line = COLOR_SEQ % (40 + HIGHLIGHT[record.levelname]) + line + \
                    RESET_SEQ
            return line

        else:
            # No coloring if it's not to be highlighted or colored
            return logging.Formatter.format(self, record)

def configure(loglevel=logging.INFO, verbose=False, simple=False):
    """Configures the root logger to our liking

    For a non-standard loglevel, pass in the level with which to configure the handler.

    For a more verbose options line, pass in verbose=True

    This function may be called more than once.

    """

    logger = logging.getLogger()

    outstream = sys.stdout
    if simple:
        formatter = DumbFormatter(verbose)

    elif platform.system() == 'Windows':
        # Our custom output stream processor knows how to deal with select ANSI
        # color escape sequences
        outstream = WindowsOutputStream(outstream)
        formatter = ANSIColorFormatter(verbose)

    elif outstream.isatty():
        # terminal logging with ANSI color
        formatter = ANSIColorFormatter(verbose)

    else:
        # Let's not assume anything. Just text.
        formatter = DumbFormatter(verbose)

    if hasattr(logger, 'overviewerHandler'):
        # we have already set up logging so just replace the formatter
        # this time with the new values
        logger.overviewerHandler.setFormatter(formatter)
        logger.setLevel(loglevel)

    else:
        # Save our handler here so we can tell which handler was ours if the
        # function is called again
        logger.overviewerHandler = logging.StreamHandler(outstream)
        logger.overviewerHandler.setFormatter(formatter)
        logger.addHandler(logger.overviewerHandler)
        logger.setLevel(loglevel)

    # Make sure Pillow doesn't spam us in verbose mode
    logging.getLogger("PIL").setLevel(logging.WARNING)
