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
Misc utility routines used by multiple files that don't belong anywhere else
"""

import imp
import os
import os.path
import sys
import platform
from subprocess import Popen, PIPE
import logging
from cStringIO import StringIO
import ctypes
import platform
from itertools import cycle, islice, product
import shutil

def get_program_path():
    if hasattr(sys, "frozen") or imp.is_frozen("__main__"):
        return os.path.dirname(sys.executable)
    else:
        try:
            # normally, we're in ./overviewer_core/util.py
            # we want ./
            return os.path.dirname(os.path.dirname(__file__))
        except NameError:
            return os.path.dirname(sys.argv[0])


# does not require git, very likely to work everywhere
def findGitHash():
    this_dir = get_program_path()
    if os.path.exists(os.path.join(this_dir,".git")):
        with open(os.path.join(this_dir,".git","HEAD")) as f:
            data = f.read().strip()
        if data.startswith("ref: "):
            if not os.path.exists(os.path.join(this_dir, ".git", data[5:])):
                return data
            with open(os.path.join(this_dir, ".git", data[5:])) as g:
                return g.read().strip()
        else:
            return data
    else:
        try:
            import overviewer_version
            return overviewer_version.HASH
        except Exception:
            return "unknown"

def findGitVersion():
    try:
        p = Popen('git describe --tags', stdout=PIPE, stderr=PIPE, shell=True)
        p.stderr.close()
        line = p.stdout.readlines()[0]
        if line.startswith('release-'):
            line = line.split('-', 1)[1]
        if line.startswith('v'):
            line = line[1:]
        # turn 0.1.0-50-somehash into 0.1.50
        # and 0.1.0 into 0.1.0
        line = line.strip().replace('-', '.').split('.')
        if len(line) == 5:
            del line[4]
            del line[2]
        else:
            assert len(line) == 3
            line[2] = '0'
        line = '.'.join(line)
        return line
    except Exception:
        try:
            import overviewer_version
            return overviewer_version.VERSION
        except Exception:
            return "unknown"

def is_bare_console():
    """Returns true if Overviewer is running in a bare console in
    Windows, that is, if overviewer wasn't started in a cmd.exe
    session.
    """
    if platform.system() == 'Windows':
        try:
            import ctypes
            GetConsoleProcessList = ctypes.windll.kernel32.GetConsoleProcessList
            num = GetConsoleProcessList(ctypes.byref(ctypes.c_int(0)), ctypes.c_int(1))
            if (num == 1):
                return True
                
        except Exception:
            pass
    return False

def exit(ret=0):
    """Drop-in replacement for sys.exit that will automatically detect
    bare consoles and wait for user input before closing.
    """
    if ret and is_bare_console():
        print
        print "Press [Enter] to close this window."
        raw_input()
    sys.exit(ret)

# http://docs.python.org/library/itertools.html
def roundrobin(iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))

def iterate_base4(d):
    """Iterates over a base 4 number with d digits"""
    return product(xrange(4), repeat=d)
   

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

# Define a context manager to handle atomic renaming or "just forget it write
# straight to the file" depending on whether os.rename provides atomic
# overwrites.
# Detect whether os.rename will overwrite files
import tempfile
with tempfile.NamedTemporaryFile() as f1:
    with tempfile.NamedTemporaryFile() as f2:
        try:
            os.rename(f1.name,f2.name)
        except OSError:
            renameworks = False
        else:
            renameworks = True
            # re-make this file so it can be deleted without error
            open(f1.name, 'w').close()
del tempfile,f1,f2
doc = """This class acts as a context manager for files that are to be written
out overwriting an existing file.

The parameter is the destination filename. The value returned into the context
is the filename that should be used. On systems that support an atomic
os.rename(), the filename will actually be a temporary file, and it will be
atomically replaced over the destination file on exit.

On systems that don't support an atomic rename, the filename returned is the
filename given.

If an error is encountered, the file is attempted to be removed, and the error
is propagated.

Example:

with FileReplacer("config") as configname:
    with open(configout, 'w') as configout:
        configout.write(newconfig)
"""
if renameworks:
    class FileReplacer(object):
        __doc__ = doc
        def __init__(self, destname):
            self.destname = destname
            self.tmpname = destname + ".tmp"
        def __enter__(self):
            # rename works here. Return a temporary filename
            return self.tmpname
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                # error
                try:
                    os.remove(self.tmpname)
                except Exception, e:
                    logging.warning("An error was raised, so I was doing "
                            "some cleanup first, but I couldn't remove "
                            "'%s'!", self.tmpname)
            else:
                # atomic rename into place
                os.rename(self.tmpname, self.destname)
else:
    class FileReplacer(object):
        __doc__ = doc
        def __init__(self, destname):
            self.destname = destname
        def __enter__(self):
            return self.destname
        def __exit__(self, exc_type, exc_val, exc_tb):
            return
del renameworks

# Logging related classes are below

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


def mirror_dir(src, dst, entities=None):
    '''copies all of the entities from src to dst'''
    if not os.path.exists(dst):
        os.mkdir(dst)
    if entities and type(entities) != list: raise Exception("Expected a list, got a %r instead" % type(entities))
    
    # files which are problematic and should not be copied
    # usually, generated by the OS
    skip_files = ['Thumbs.db', '.DS_Store']
    
    for entry in os.listdir(src):
        if entry in skip_files:
            continue
        if entities and entry not in entities:
            continue
        
        if os.path.isdir(os.path.join(src,entry)):
            mirror_dir(os.path.join(src, entry), os.path.join(dst, entry))
        elif os.path.isfile(os.path.join(src,entry)):
            try:
                shutil.copy(os.path.join(src, entry), os.path.join(dst, entry))
            except IOError as outer: 
                try:
                    # maybe permission problems?
                    src_stat = os.stat(os.path.join(src, entry))
                    os.chmod(os.path.join(src, entry), src_stat.st_mode | stat.S_IRUSR)
                    dst_stat = os.stat(os.path.join(dst, entry))
                    os.chmod(os.path.join(dst, entry), dst_stat.st_mode | stat.S_IWUSR)
                except OSError: # we don't care if this fails
                    pass
                shutil.copy(os.path.join(src, entry), os.path.join(dst, entry))
                # if this stills throws an error, let it propagate up


def dict_subset(d, keys):
    "Return a new dictionary that is built from copying select keys from d"
    n = dict()
    for key in keys:
        if key in d:
            n[key] = d[key]
    return n

    
