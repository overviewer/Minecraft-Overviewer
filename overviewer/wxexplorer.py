import wx
import sys
import os, os.path
import tempfile
import multiprocessing

from overviewer import oil
from overviewer import textures
from overviewer import world
from overviewer import isometricrenderer
from overviewer import blockdefinitions
from overviewer import cache


def render_tile(r, tile_size, x, y):
    im = oil.Image(tile_size, tile_size)
    r.render((tile_size * x, tile_size * y), im)
    
    _, f = tempfile.mkstemp(suffix='.png')
    im.save(f, oil.FORMAT_PNG)
    return f

class MapWindow(wx.Window):
    tile_size = 256
    
    def __init__(self, rendererf, *args, **kwargs):
        super(MapWindow, self).__init__(*args, **kwargs)
        self.rendererf = rendererf
        self.cache = cache.LRUCache()
        self.pool = multiprocessing.Pool()
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
    
    def get_tile(self, x, y):
        try:
            return self.cache[(x, y)]
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
        return None
    
    def ResetMap(self):
        self.origin = (0, 0)
        self.zoom = 0
        self.scale = 1.0
        self.Refresh()
    
    def ClearCache(self):
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
    
    def OnMousewheel(self, e):
        e.Skip()
        delta = e.GetWheelDelta()
        rotation = e.GetWheelRotation()
        rotation = rotation // delta
        if rotation == 0:
            return
        
        oldscale = self.scale
        self.zoom += rotation
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
        self.caches = [cache.LRUCache()]
        self.world = None
        self.rset = None
        self.tex = textures.Textures()
        self.bdefs = blockdefinitions.get_default()
        self.isomatrix = oil.Matrix().rotate(0.6154797, 0, 0).rotate(0, 0.7853982, 0).scale(17, 17, 17)
        self.tdmatrix = oil.Matrix().rotate(3.1415926 / 2, 0, 0).scale(17, 17, 17)
        self.matrix = self.isomatrix
        
        self.map = MapWindow(self.rendererf, self)
        box.Add(self.map, 1, wx.EXPAND)
        
        self.SetSizer(box)
    
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
    
    def OnOpenWorld(self, ev):
        dlg = wx.FileDialog(self, "Choose a world", "", "", "*.dat", wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            dirname = os.path.split(dlg.GetPath())[0]
            self.world = world.World(dirname)
            self.rset = self.world.get_regionset(0)
            self.rset = world.CachedRegionSet(self.rset, self.caches)
            self.map.ResetMap()
            self.map.ClearCache()
        dlg.Destroy()

if __name__ == '__main__':
    app = wx.App()
    frame = Explorer(None, -1, "wxExplorer")
    frame.Show()
    
    app.MainLoop()
