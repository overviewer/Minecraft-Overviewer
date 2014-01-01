import wx
import sys
import os, os.path
import tempfile
import threading
import time
import types
import multiprocessing

modules = [
    'oil',
    'textures',
    'world',
    'isometricrenderer',
    'blockdefinitions',
    'blocks',
    'cache'
]

def reload_all(verbose=False):
    global modules
    for _ in range(2):
        for name in modules:
            name = 'overviewer.' + name
            if verbose:
                print "reloading", name
            if not name in sys.modules:
                exec "import " + name
            else:
                reload(sys.modules[name])
        verbose = False
    
    for name in sys.modules['overviewer.blocks'].block_modules:
        name = 'blocks.' + name
        if not name in modules:
            modules.append(name)
    
    for name, mod in sys.modules.items():
        if not isinstance(mod, types.ModuleType):
            continue
        if not name.startswith("overviewer."):
            continue
        name = name.split('.', 1)[1]
        if name in modules:
            continue
        modules.append(name)

# appease the pickle!
reload_all()
from overviewer import oil
from overviewer import textures
from overviewer import world
from overviewer import isometricrenderer
from overviewer import blockdefinitions
from overviewer import cache

BACKEND = oil.BACKEND_CPU

try:
    import pyinotify
    USE_INOTIFY = True
    print "found pyinotify, reloading enabled"
except ImportError:
    print "install pyinotify to get auto-code-reloading"
    USE_INOTIFY = False

def wait_for_code_change():
    if not USE_INOTIFY:
        while True:
            # if we have no inotify, don't even try
            time.sleep(10)
    
    wm = pyinotify.WatchManager()
    notifier = None
    done = []
    
    class OnModifyHandler(pyinotify.ProcessEvent):
        def process_IN_MODIFY(self, event):
            path = event.pathname
            exts = ['so', 'dll', 'py', 'dylib', 'obj', 'mtl', 'json']
            if any(path.endswith('.' + ext) for ext in exts):
                done.append(())
    
    watchpath = os.path.split(__file__)[0]
    handler = OnModifyHandler()
    
    notifier = pyinotify.Notifier(wm, default_proc_fun=handler, read_freq=1, timeout=10)
    notifier.coalesce_events()
    wm.add_watch(watchpath, pyinotify.ALL_EVENTS, rec=True, auto_add=True)
    
    while not done:
        notifier.process_events()
        while notifier.check_events():
            notifier.read_events()
            notifier.process_events()
        
    notifier.stop()

def render_tile(r, tile_size, x, y):
    im = oil.Image(tile_size, tile_size)
    r.render((tile_size * x, tile_size * y), im)
    
    _, f = tempfile.mkstemp(suffix='.png')
    im.save(f, oil.FORMAT_PNG)
    return f

def worker_initializer():
    oil.backend_set(BACKEND)

class MapWindow(wx.Window):
    tile_size = 256
    
    def __init__(self, rendererf, *args, **kwargs):
        super(MapWindow, self).__init__(*args, **kwargs)
        self.rendererf = rendererf
        self.cache = cache.LRUCache()
        self.oldcache = None
        self.pool = multiprocessing.Pool(initializer=worker_initializer)
        self.origin = (0, 0)
        self.zoom = 0
        self.scale = 1.0
        self.left_down = None
        self.left_down_origin = None
        
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_MOTION, self.OnMotion)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnMousewheel)
        self.Bind(wx.EVT_LEFT_DCLICK, self.OnDoubleClick)
    
    def get_tile(self, x, y):
        try:
            t = self.cache[(x, y)]
            if t is None and self.oldcache is not None:
                try:
                    return self.oldcache[(x, y)]
                except KeyError:
                    pass
            return t
        except KeyError:
            pass
        
        def callback(f):
            bmp = wx.Bitmap(f, wx.BITMAP_TYPE_PNG)
            os.unlink(f)
            self.cache[(x, y)] = bmp
            self.Refresh()
        
        def cbWrap(f):
            wx.CallAfter(callback, f)
        
        r = self.rendererf(self.scale)
        if r:
            self.pool.apply_async(render_tile, (r, self.tile_size, x, y), {}, cbWrap)
        self.cache[(x, y)] = None
        if self.oldcache is not None:
            try:
                return self.oldcache[(x, y)]
            except KeyError:
                return None
        return None
    
    def ResetMap(self):
        self.origin = (0, 0)
        self.zoom = 0
        self.scale = 1.0
        self.Refresh()
    
    def ClearCache(self, keep_old=False):
        if keep_old:
            self.oldcache = self.cache
        else:
            self.oldcache = None
        self.cache = cache.LRUCache()
        self.Refresh()
    
    def OnLeftDown(self, e):
        e.Skip()
        self.left_down = e.GetPositionTuple()
        self.left_down_origin = self.origin
    
    def OnMotion(self, e):
        e.Skip()
        if not e.LeftIsDown():
            return
        
        x, y = e.GetPositionTuple()
        sx, sy = self.left_down
        ox, oy = self.left_down_origin
        dx, dy = (sx - x, sy - y)
        nox, noy = (ox + dx, oy + dy)
        self.origin = (nox, noy)
        self.Refresh()
    
    def OnLeftUp(self, e):
        e.Skip()
        self.left_down = None
        self.left_down_origin = None
    
    def DoZoom(self, e, amount):
        oldscale = self.scale
        self.zoom += amount
        self.scale = 2.0 ** self.zoom
        
        cx, cy = e.GetPositionTuple()
        
        ox, oy = self.origin
        width, height = self.GetSize()
        ox -= width / 2
        oy -= height / 2
        
        x, y = (ox + cx, oy + cy)
        x *= self.scale / oldscale
        y *= self.scale / oldscale
        
        ox, oy = (int(x - cx), int(y - cy))
        ox += width / 2
        oy += height / 2
        self.origin = (ox, oy)
        
        self.ClearCache()
    
    def OnMousewheel(self, e):
        e.Skip()
        delta = e.GetWheelDelta()
        rotation = e.GetWheelRotation()
        rotation = rotation // delta
        if rotation == 0:
            return
        
        self.DoZoom(e, rotation)
    
    def OnDoubleClick(self, e):
        e.Skip()
        self.DoZoom(e, 1)
    
    def OnSize(self, e):
        self.Refresh()
    
    def OnPaint(self, e):
        dc = wx.BufferedPaintDC(self)
        dc.SetBackground(wx.Brush('#000000'))
        dc.Clear()
        
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.SetPen(wx.Pen('#FF0000'))
        
        width, height = self.GetSize()
        x, y = self.origin
        x -= width / 2
        y -= height / 2
        
        sx, sy = (x // self.tile_size, y // self.tile_size)
        ex, ey = (sx + width // self.tile_size, sy + height // self.tile_size)
        for tx in range(sx, ex + 2):
            for ty in range(sy, ey + 2):
                dx = tx * self.tile_size - x
                dy = ty * self.tile_size - y
                bmp = self.get_tile(tx, ty)
                if bmp is None:
                    continue
                dc.DrawBitmap(bmp, dx, dy)
                #dc.DrawRectangle(dx, dy, self.tile_size, self.tile_size)

class Explorer(wx.Frame):
    def __init__(self, *args, **kwargs):
        super(Explorer, self).__init__(*args, **kwargs)
        
        box = wx.BoxSizer(wx.HORIZONTAL)
        
        panel = wx.Panel(self)
        box.Add(panel, 0, wx.EXPAND | wx.ALL, 10)
        panelsizer = wx.GridBagSizer(10, 10)
        
        btn = wx.Button(panel, label="Load New World")
        btn.Bind(wx.EVT_BUTTON, self.OnOpenWorld)
        panelsizer.Add(btn, (0, 0))

        btn = wx.Button(panel, label="Reset View")
        btn.Bind(wx.EVT_BUTTON, lambda ev: self.map.ResetMap())
        panelsizer.Add(btn, (1, 0))
        
        btn = wx.RadioButton(panel, label="Isometric", style=wx.RB_GROUP)
        btn.Bind(wx.EVT_RADIOBUTTON, self.SetIsometric)
        panelsizer.Add(btn, (2, 0))
        btn = wx.RadioButton(panel, label="Top-Down")
        btn.Bind(wx.EVT_RADIOBUTTON, self.SetTopDown)
        panelsizer.Add(btn, (3, 0))

        panel.SetSizer(panelsizer)
        
        # rendererf stuff
        self.wpath = None
        self.RecreateObjects()
        self.matrix = self.isomatrix
        
        self.map = MapWindow(self.rendererf, self)
        box.Add(self.map, 1, wx.EXPAND)
        
        self.SetSizer(box)
    
    def OnReload(self):
        print "(reloading...)"
        reload_all()
        self.RecreateObjects()
        self.map.ClearCache(keep_old=True)
    
    def RecreateObjects(self):
        self.caches = [cache.LRUCache()]
        if self.wpath is None:
            self.world = None
            self.rset = None
        else:
            self.LoadWorld(self.wpath)
        self.tex = textures.Textures()
        self.bdefs = blockdefinitions.get_default()
        
        self.isomatrix = oil.Matrix().rotate(0.6154797, 0, 0).rotate(0, 0.7853982, 0).scale(17, 17, 17)
        self.tdmatrix = oil.Matrix().rotate(3.1415926 / 2, 0, 0).scale(17, 17, 17)
    
    def rendererf(self, zoom):
        if self.rset is None:
            return None
        matrix = oil.Matrix(self.matrix).scale(zoom, zoom, zoom)
        renderer = isometricrenderer.IsometricRenderer(self.rset, self.tex, self.bdefs, matrix)
        return renderer
    
    def SetMatrix(self, mat):
        inv = oil.Matrix(self.matrix).scale(self.map.scale, self.map.scale, self.map.scale)
        inv.invert()
        origin = inv.transform(*(self.map.origin + (0,)))
        self.matrix = mat
        mat = oil.Matrix(mat).scale(self.map.scale, self.map.scale, self.map.scale)
        self.map.origin = tuple(int(x) for x in mat.transform(*origin)[:2])
        self.map.ClearCache()
    
    def SetIsometric(self, ev):
        self.SetMatrix(self.isomatrix)

    def SetTopDown(self, ev):
        self.SetMatrix(self.tdmatrix)
    
    def LoadWorld(self, wpath):
        self.wpath = wpath
        self.world = world.World(wpath)
        self.rset = self.world.get_regionset(0)
        self.rset = world.CachedRegionSet(self.rset, self.caches)
    
    def OnOpenWorld(self, ev):
        dlg = wx.FileDialog(self, "Choose a world", "", "", "*.dat", wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            dirname = os.path.split(dlg.GetPath())[0]
            self.LoadWorld(dirname)
            self.map.ResetMap()
            self.map.ClearCache()
        dlg.Destroy()

def code_change_thread(frame):
    while True:
        wait_for_code_change()
        wx.CallAfter(frame.OnReload)

if __name__ == '__main__':
    if '--opengl' in sys.argv:
        BACKEND = oil.BACKEND_OPENGL
        print "using opengl backend"

    app = wx.App()
    frame = Explorer(None, -1, "wxExplorer")
    frame.Show()
    
    cthread = threading.Thread(target=code_change_thread, args=(frame,))
    cthread.daemon = True
    cthread.start()
    
    app.MainLoop()
