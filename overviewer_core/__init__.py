# c_overviewer must be imported first, because it imports other
# modules; leaving this out can lead to bad dependency loops

try:
    import c_overviewer
except ImportError:
    pass
