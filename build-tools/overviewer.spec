# -*- mode: python -*-

block_cipher = None

import sys
sys.modules['FixTk'] = None

def get_overviewer_pkgname():
    from overviewer_core import overviewer_version
    return "overviewer-" + overviewer_version.VERSION

a = Analysis(['overviewer.py'],
             pathex=['Z:\\devel\\Minecraft-Overviewer'],
             binaries=None,
             datas=[("overviewer_core/data", "overviewer_core/data")],
             hiddenimports=['overviewer_core.aux_files.genPOI'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['FixTk', 'tcl', 'tk', '_tkinter', 'tkinter', 'Tkinter'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='overviewer',
          debug=False,
          strip=False,
          upx=False,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               name=get_overviewer_pkgname())
