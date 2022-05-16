# Provides an python wrapper for the ctypes wrapper for the c wrapper for leveldb.

import ctypes
import os.path as op
import sys

if sys.platform == "win32":
  if sys.maxsize > 2**32: # 64 bit python
    ldb = ctypes.cdll.LoadLibrary(op.join(op.dirname(op.realpath(__file__)), "LevelDB-MCPE-64.dll"))
  else: # 32 bit python
    ldb = ctypes.cdll.LoadLibrary(op.join(op.dirname(op.realpath(__file__)), "LevelDB-MCPE-32.dll"))
else: #linux, compile your own .so if this errors!
  ldb = ctypes.cdll.LoadLibrary(op.join(op.dirname(op.realpath(__file__)), "libleveldb.so")) # Load DLL

# Setup ctypes arguments and return types for all of the leveldb functions.
# Most of this pulled from Podshot/MCEdit-Unified
ldb.leveldb_filterpolicy_create_bloom.argtypes = [ctypes.c_int]
ldb.leveldb_filterpolicy_create_bloom.restype = ctypes.c_void_p
ldb.leveldb_filterpolicy_destroy.argtypes = [ctypes.c_void_p]
ldb.leveldb_filterpolicy_destroy.restype = None
ldb.leveldb_cache_create_lru.argtypes = [ctypes.c_size_t]
ldb.leveldb_cache_create_lru.restype = ctypes.c_void_p
ldb.leveldb_cache_destroy.argtypes = [ctypes.c_void_p]
ldb.leveldb_cache_destroy.restype = None

ldb.leveldb_options_create.argtypes = []
ldb.leveldb_options_create.restype = ctypes.c_void_p
ldb.leveldb_options_set_filter_policy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
ldb.leveldb_options_set_filter_policy.restype = None
ldb.leveldb_options_set_create_if_missing.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
ldb.leveldb_options_set_create_if_missing.restype = None
ldb.leveldb_options_set_error_if_exists.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
ldb.leveldb_options_set_error_if_exists.restype = None
ldb.leveldb_options_set_paranoid_checks.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
ldb.leveldb_options_set_paranoid_checks.restype = None
ldb.leveldb_options_set_write_buffer_size.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
ldb.leveldb_options_set_write_buffer_size.restype = None
ldb.leveldb_options_set_max_open_files.argtypes = [ctypes.c_void_p, ctypes.c_int]
ldb.leveldb_options_set_max_open_files.restype = None
ldb.leveldb_options_set_cache.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
ldb.leveldb_options_set_cache.restype = None
ldb.leveldb_options_set_block_size.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
ldb.leveldb_options_set_block_size.restype = None
ldb.leveldb_options_destroy.argtypes = [ctypes.c_void_p]
ldb.leveldb_options_destroy.restype = None

ldb.leveldb_options_set_compression.argtypes = [ctypes.c_void_p, ctypes.c_int]
ldb.leveldb_options_set_compression.restype = None

ldb.leveldb_open.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
ldb.leveldb_open.restype = ctypes.c_void_p
ldb.leveldb_close.argtypes = [ctypes.c_void_p]
ldb.leveldb_close.restype = None
ldb.leveldb_put.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
ldb.leveldb_put.restype = None
ldb.leveldb_delete.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
ldb.leveldb_delete.restype = None
ldb.leveldb_write.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
ldb.leveldb_write.restype = None
ldb.leveldb_get.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
ldb.leveldb_get.restype = ctypes.POINTER(ctypes.c_char)

ldb.leveldb_writeoptions_create.argtypes = []
ldb.leveldb_writeoptions_create.restype = ctypes.c_void_p
ldb.leveldb_writeoptions_destroy.argtypes = [ctypes.c_void_p]
ldb.leveldb_writeoptions_destroy.restype = None
ldb.leveldb_writeoptions_set_sync.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
ldb.leveldb_writeoptions_set_sync.restype = None

ldb.leveldb_readoptions_create.argtypes = []
ldb.leveldb_readoptions_create.restype = ctypes.c_void_p
ldb.leveldb_readoptions_destroy.argtypes = [ctypes.c_void_p]
ldb.leveldb_readoptions_destroy.restype = None
ldb.leveldb_readoptions_set_verify_checksums.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
ldb.leveldb_readoptions_set_verify_checksums.restype = None
ldb.leveldb_readoptions_set_fill_cache.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
ldb.leveldb_readoptions_set_fill_cache.restype = None
ldb.leveldb_readoptions_set_snapshot.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
ldb.leveldb_readoptions_set_snapshot.restype = None

ldb.leveldb_create_iterator.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
ldb.leveldb_create_iterator.restype = ctypes.c_void_p
ldb.leveldb_iter_destroy.argtypes = [ctypes.c_void_p]
ldb.leveldb_iter_destroy.restype = None
ldb.leveldb_iter_valid.argtypes = [ctypes.c_void_p]
ldb.leveldb_iter_valid.restype = ctypes.c_bool
ldb.leveldb_iter_key.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
ldb.leveldb_iter_key.restype = ctypes.c_void_p
ldb.leveldb_iter_value.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
ldb.leveldb_iter_value.restype = ctypes.c_void_p
ldb.leveldb_iter_next.argtypes = [ctypes.c_void_p]
ldb.leveldb_iter_next.restype = None
ldb.leveldb_iter_prev.argtypes = [ctypes.c_void_p]
ldb.leveldb_iter_prev.restype = None
ldb.leveldb_iter_seek_to_first.argtypes = [ctypes.c_void_p]
ldb.leveldb_iter_seek_to_first.restype = None
ldb.leveldb_iter_seek_to_last.argtypes = [ctypes.c_void_p]
ldb.leveldb_iter_seek_to_last.restype = None
ldb.leveldb_iter_seek.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
ldb.leveldb_iter_seek.restype = None
ldb.leveldb_iter_get_error.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
ldb.leveldb_iter_get_error.restype = None

ldb.leveldb_writebatch_create.argtypes = []
ldb.leveldb_writebatch_create.restype = ctypes.c_void_p
ldb.leveldb_writebatch_destroy.argtypes = [ctypes.c_void_p]
ldb.leveldb_writebatch_destroy.restype = None
ldb.leveldb_writebatch_clear.argtypes = [ctypes.c_void_p]
ldb.leveldb_writebatch_clear.restype = None

ldb.leveldb_writebatch_put.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
ldb.leveldb_writebatch_put.restype = None
ldb.leveldb_writebatch_delete.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
ldb.leveldb_writebatch_delete.restype = None

ldb.leveldb_approximate_sizes.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
ldb.leveldb_approximate_sizes.restype = None

ldb.leveldb_compact_range.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
ldb.leveldb_compact_range.restype = None

ldb.leveldb_create_snapshot.argtypes = [ctypes.c_void_p]
ldb.leveldb_create_snapshot.restype = ctypes.c_void_p
ldb.leveldb_release_snapshot.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
ldb.leveldb_release_snapshot.restype = None

ldb.leveldb_free.argtypes = [ctypes.c_void_p]
ldb.leveldb_free.restype = None

# Utility function for checking the error code returned by some leveldb functions.
def _checkError(err):
  if bool(err): # Not an empty null-terminated string
    message = ctypes.string_at(err)
    ldb.leveldb_free(ctypes.cast(err, ctypes.c_void_p))
    raise Exception(message)

def open(path):
  # Bloom filter: an efficient way to tell if something is in a cache.
  filter_policy = ldb.leveldb_filterpolicy_create_bloom(10)
  cache = ldb.leveldb_cache_create_lru(40 * 1024 * 1024)
  options = ldb.leveldb_options_create()
  # Many of these options were pulled from Podshot/MCEdit-Unified
  ldb.leveldb_options_set_compression(options, 4)
  ldb.leveldb_options_set_filter_policy(options, filter_policy)
  ldb.leveldb_options_set_create_if_missing(options, False)
  ldb.leveldb_options_set_write_buffer_size(options, 4 * 1024 * 1024)
  ldb.leveldb_options_set_cache(options, cache)
  ldb.leveldb_options_set_block_size(options, 163840)

  error = ctypes.POINTER(ctypes.c_char)()
  db = ldb.leveldb_open(options, path.encode("utf-8"), ctypes.byref(error))
  ldb.leveldb_options_destroy(options)
  _checkError(error)

  return db

def get(db, key):
  ro = ldb.leveldb_readoptions_create()
  size = ctypes.c_size_t(0)
  error = ctypes.POINTER(ctypes.c_char)()
  valPtr = ldb.leveldb_get(db, ro, key, len(key), ctypes.byref(size), ctypes.byref(error))
  ldb.leveldb_readoptions_destroy(ro)
  _checkError(error)
  if bool(valPtr):
    val = ctypes.string_at(valPtr, size.value)
    ldb.leveldb_free(ctypes.cast(valPtr, ctypes.c_void_p))
  else:
    raise KeyError("Key {} not found in database.".format(key))
  return val

def put(db, key, val):
  wo = ldb.leveldb_writeoptions_create()
  error = ctypes.POINTER(ctypes.c_char)()
  ldb.leveldb_put(db, wo, key, len(key), val, len(val), ctypes.byref(error))
  ldb.leveldb_writeoptions_destroy(wo)
  _checkError(error)

def putBatch(db, data):
  batch = ldb.leveldb_writebatch_create()
  for k, v in data.items():
    ldb.leveldb_writebatch_put(batch, k, len(k), v, len(v))
  wo = ldb.leveldb_writeoptions_create()
  error = ctypes.POINTER(ctypes.c_char)()
  ldb.leveldb_write(db, wo, batch, ctypes.byref(error))
  ldb.leveldb_writeoptions_destroy(wo)
  _checkError(error)

def delete(db, key):
  wo = ldb.leveldb_writeoptions_create()
  error = ctypes.POINTER(ctypes.c_char)()
  ldb.leveldb_delete(db, wo, key, len(key), ctypes.byref(error))
  ldb.leveldb_writeoptions_destroy(wo)
  _checkError(error)

def iterate(db, start=None, end=None):
  ro = ldb.leveldb_readoptions_create()
  it = ldb.leveldb_create_iterator(db, ro)
  ldb.leveldb_readoptions_destroy(ro)
  if start is None:
    ldb.leveldb_iter_seek_to_first(it)
  else:
    ldb.leveldb_iter_seek(it, start, len(start))
  try:
    while ldb.leveldb_iter_valid(it):
      size = ctypes.c_size_t(0)
      keyPtr = ldb.leveldb_iter_key(it, ctypes.byref(size))
      key = ctypes.string_at(keyPtr, size.value)
      if end is not None and key >= end:
        break
      valPtr = ldb.leveldb_iter_value(it, ctypes.byref(size))
      val = ctypes.string_at(valPtr, size.value)
      yield key, val
      ldb.leveldb_iter_next(it)
  finally:
    ldb.leveldb_iter_destroy(it)

def close(db):
  ldb.leveldb_close(db)
