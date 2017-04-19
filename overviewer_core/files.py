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

import os
import os.path
import tempfile
import shutil
import logging
import stat
import errno
import platform

on_windows = platform.system() == 'Windows'
default_caps = {"chmod_works": True, "rename_works": True}

def get_fs_caps(dir_to_test):
    return {"chmod_works": does_chmod_work(dir_to_test),
            "rename_works": does_rename_work(dir_to_test)
            }

def does_chmod_work(dir_to_test):
    "Detects if chmod works in a given directory"
    # a CIFS mounted FS is the only thing known to reliably not provide chmod

    if not os.path.isdir(dir_to_test):
        return True

    f1 = tempfile.NamedTemporaryFile(dir=dir_to_test)
    try:
        f1_stat = os.stat(f1.name)
        os.chmod(f1.name, f1_stat.st_mode | stat.S_IRUSR)
        chmod_works = True
        logging.debug("Detected that chmods work in %r" % dir_to_test)
    except OSError:
        chmod_works = False
        logging.debug("Detected that chmods do NOT work in %r" % dir_to_test)
    return chmod_works

def does_rename_work(dir_to_test):
    with tempfile.NamedTemporaryFile(dir=dir_to_test) as f1:
        with tempfile.NamedTemporaryFile(dir=dir_to_test) as f2:
            try:
                os.rename(f1.name,f2.name)
            except OSError:
                renameworks = False
                logging.debug("Detected that overwriting renames do NOT work in %r" % dir_to_test)
            else:
                renameworks = True
                logging.debug("Detected that overwriting renames work in %r" % dir_to_test)
                # re-make this file so it can be deleted without error
                open(f1.name, 'w').close()
    return renameworks

## useful recursive copy, that ignores common OS cruft
def mirror_dir(src, dst, entities=None, capabilities=default_caps):
    '''copies all of the entities from src to dst'''
    chmod_works = capabilities.get("chmod_works")
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
            mirror_dir(os.path.join(src, entry), os.path.join(dst, entry), capabilities=capabilities)
        elif os.path.isfile(os.path.join(src,entry)):
            try:
                if chmod_works:
                    shutil.copy(os.path.join(src, entry), os.path.join(dst, entry))
                else:
                    shutil.copyfile(os.path.join(src, entry), os.path.join(dst, entry))
            except IOError as outer: 
                try:
                    # maybe permission problems?
                    src_stat = os.stat(os.path.join(src, entry))
                    os.chmod(os.path.join(src, entry), src_stat.st_mode | stat.S_IRUSR)
                    dst_stat = os.stat(os.path.join(dst, entry))
                    os.chmod(os.path.join(dst, entry), dst_stat.st_mode | stat.S_IWUSR)
                except OSError: # we don't care if this fails
                    pass
                # try again; if this stills throws an error, let it propagate up
                if chmod_works:
                    shutil.copy(os.path.join(src, entry), os.path.join(dst, entry))
                else:
                    shutil.copyfile(os.path.join(src, entry), os.path.join(dst, entry))

# Define a context manager to handle atomic renaming or "just forget it write
# straight to the file" depending on whether os.rename provides atomic
# overwrites.
# Detect whether os.rename will overwrite files
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
class FileReplacer(object):
    __doc__ = doc
    def __init__(self, destname, capabilities=default_caps):
        self.caps = capabilities
        self.destname = destname
        if self.caps.get("rename_works"):
            self.tmpname = destname + ".tmp"
    def __enter__(self):
        if self.caps.get("rename_works"):
            # rename works here. Return a temporary filename
            return self.tmpname
        return self.destname
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.caps.get("rename_works"):
            if exc_type:
                # error
                try:
                    os.remove(self.tmpname)
                except Exception, e:
                    logging.warning("An error was raised, so I was doing "
                            "some cleanup first, but I couldn't remove "
                            "'%s'!", self.tmpname)
            else:
                # copy permission bits, if needed
                if self.caps.get("chmod_works") and os.path.exists(self.destname):
                    try:
                        shutil.copymode(self.destname, self.tmpname)
                    except OSError, e:
                        # Ignore errno ENOENT: file does not exist. Due to a race
                        # condition, two processes could conceivably try and update
                        # the same temp file at the same time
                        if e.errno != errno.ENOENT:
                            raise
                # atomic rename into place
                try:
                    if on_windows:
                        try:
                            os.remove(self.destname)
                        except OSError, e:
                            pass
                    os.rename(self.tmpname, self.destname)
                except OSError, e:
                    # Ignore errno ENOENT: file does not exist. Due to a race
                    # condition, two processes could conceivably try and update
                    # the same temp file at the same time
                    if e.errno != errno.ENOENT:
                        raise
