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

import errno
import imp
import os.path
import platform
import sys
from itertools import cycle, islice, product
from string import hexdigits
from subprocess import PIPE, Popen


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


def findGitHash():
    try:
        p = Popen('git rev-parse HEAD', stdout=PIPE, stderr=PIPE, shell=True)
        p.stderr.close()
        line = p.stdout.readlines()[0].decode('utf-8').strip()
        if line and len(line) == 40 and all(c in hexdigits for c in line):
            return line
    except Exception:
        try:
            from . import overviewer_version
            return overviewer_version.HASH
        except Exception:
            pass
    return "unknown"


def findGitVersion():
    try:
        p = Popen('git describe --tags --match "v*.*.*"', stdout=PIPE, stderr=PIPE, shell=True)
        p.stderr.close()
        line = p.stdout.readlines()[0].decode('utf-8')
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
            from . import overviewer_version
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


def nice_exit(ret=0):
    """Drop-in replacement for sys.exit that will automatically detect
    bare consoles and wait for user input before closing.
    """
    if ret and is_bare_console():
        print("")
        print("Press [Enter] to close this window.")
        input()
    sys.exit(ret)


# http://docs.python.org/library/itertools.html
def roundrobin(iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


def dict_subset(d, keys):
    "Return a new dictionary that is built from copying select keys from d"
    n = dict()
    for key in keys:
        if key in d:
            n[key] = d[key]
    return n


def pid_exists(pid):    # http://stackoverflow.com/a/6940314/1318435
    """Check whether pid exists in the current process table."""
    if pid < 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno != errno.ESRCH
    else:
        return True
