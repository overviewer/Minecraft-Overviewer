#!/usr/bin/env python2

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
    tile_size = 512
    
    def __init__(self, rendererf, *args, **kwargs):
        super(MapWindow, self).__init__(*args, **kwargs)
        self.rendererf = rendererf
        self.cache = cache.LRUCache()
        self.pool = multiprocessing.Pool()
        self.origin = (0, 0)
        self.left_down = None
        self.left_down_origin = None
        
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_MOTION, self.OnMotion)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
    
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
        
        r = self.rendererf()
        self.pool.apply_async(render_tile, (r, self.tile_size, x, y), {}, cbWrap)
        self.cache[(x, y)] = None
        return None

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
        nox, noy = (ox + sx - x, oy + sy - y)
        self.origin = (nox, noy)
        self.Refresh()
    
    def OnLeftUp(self, e):
        e.Skip()
        self.left_down = None
        self.left_down_origin = None
    
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

def make_renderer(wpath):
    caches = [cache.LRUCache()]
    w = world.World(wpath)
    rset = w.get_regionset(0)
    rset = world.CachedRegionSet(rset, caches)
    
    tex = textures.Textures()
    bdefs = blockdefinitions.get_default()
    matrix = oil.Matrix().rotate(0.6154797, 0, 0).rotate(0, 0.7853982, 0).scale(17, 17, 17)
    renderer = isometricrenderer.IsometricRenderer(rset, tex, bdefs, matrix)
    return renderer

if __name__ == '__main__':
    try:
        name, wpath = sys.argv
        name = os.path.split(name)[-1]
    except ValueError:
        print("usage: {} [worldpath]".format(sys.argv[0]))
        sys.exit(1)
    
    renderer = make_renderer(wpath)
    
    app = wx.App()
    frame = wx.Frame(None, -1, name)
    MapWindow(lambda: renderer, frame)
    frame.Show()
    
    app.MainLoop()
