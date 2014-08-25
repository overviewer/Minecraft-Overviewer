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

import zipfile
import json
import os
from datetime import datetime

from . import path
from . import oil

class AssetPack:
    # should raise KeyError if path isn't present
    def open(self, *path):
        raise NotImplementedError("AssetPack.open")
    
    # everything below this line is implemented in terms of open()

    def read(self, *path, encoding=None):
        with self.open(*path) as f:
            d = f.read()
        if encoding:
            d = d.decode(encoding)
        return d
    
    def read_json(self, *path, encoding='utf-8'):
        return json.loads(self.read(*path, encoding=encoding))
    
    def open_texture(self, *path):
        metapath = list(path)
        metapath[-1] += ".mcmeta"
        
        with self.open(*path) as f:
            img = oil.Image.load(f)
        try:
            meta = self.read_json(*metapath)
        except KeyError:
            meta = {}
            
        # Textures may be animated, so un-animate them
        animation = meta.get('animation', {})
        wframes = animation.get('width', 1)
        
        w, h = img.size
        realw = w // wframes
        if realw != w or realw != h:
            dest = oil.Image(realw, realw)
            dest.composite(img, 255, 0, 0)
            img = dest
        
        return img

class CompositeAssetPack(AssetPack):
    def __init__(self, others):
        self.others = others
    
    def __repr__(self):
        return "<CompositeAssetPack: {0!r}>".format(self.others)
    
    def open(self, *path):
        for other in self.others:
            try:
                return other.open(*path)
            except KeyError:
                pass
        raise KeyError("could not find asset: " + repr(path))

class ZipAssetPack(AssetPack):
    def __init__(self, path_or_file, meta=None):
        self.path_or_file = path_or_file
        self.zip = zipfile.ZipFile(path_or_file, 'r')
        self.meta = meta
        if self.meta is None:
            try:
                self.meta = self.open_raw('pack.mcmeta').read().decode()
                self.meta = json.loads(self.meta)
                self.meta = self.meta['pack']
            except KeyError:
                pass
    
    def __repr__(self):
        return "<ZipAssetPack: {0!r}>".format(self.path_or_file)
    
    def __getstate__(self):
        return (self.path_or_file, self.meta)
    
    def __setstate__(self, d):
        path, meta = d
        self.__init__(path, meta=meta)
    
    def open(self, *path):
        return self.zip.open('/'.join(path))

# FIXME: don't use snapshots after 1.8 is out
def get_default(want_version='snapshot'):
    """Get the default assetpack, from an installed minecraft client. The
    version argument can be set to select a specific version, or you
    can use the special versions 'release' and 'snapshot' to select
    the latest available official release or snapshot release.

    """
    
    available = {}
    base = path.get_minecraft_path('versions')
    if not os.path.exists(base):
        raise InstalledMinecraftError('Could not open default asset pack, no Minecraft jars found in: ' + base)
    
    for version in os.listdir(base):
        full = os.path.join(base, version)
        if not os.path.isdir(full):
            continue
        data = os.path.join(full, version + '.jar')
        meta = os.path.join(full, version + '.json')
        if not all(os.path.isfile(p) for p in [data, meta]):
            continue
        
        with open(meta) as f:
            metadata = json.load(f)
        
        def parsetime(s):
            s = ''.join(s.rsplit(':', 1))
            dateformat = "%Y-%m-%dT%H:%M:%S%z"
            return datetime.strptime(s, dateformat)
        
        date = parsetime(metadata['releaseTime'])
        name = "Official Minecraft " + metadata['id']
        kind = metadata['type']
        pack = ZipAssetPack(data, meta=dict(name=name, date=date))
        
        for ver in [version, kind]:
            if ver in available:
                if available[ver].meta['date'] < date:
                    available[ver] = pack
            else:
                available[ver] = pack
        
    if not want_version in available:
        raise InstalledMinecraftError("could not find minecraft version: " + want_version)
    
    return available[want_version]
        
