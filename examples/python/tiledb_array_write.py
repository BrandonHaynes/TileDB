import sys
import random
#import constants
import tiledb
from tiledb import constants
import numpy as np

arrayname = 'array' + str(random.randint(0, sys.maxint))

conf = tiledb.Configuration('workspace', None, constants.TILEDB_IO_READ , constants.TILEDB_IO_WRITE)
print conf.home

context = tiledb.Context(conf)
#context.create_workspace('workspace/foo')
print context

path = "workspace/foo/" + arrayname
attrs = ["a1", "a2", "a3"]
dims = ["d1", "d2"]
domain = [(1, 4), (1, 4)]
#cell_counts = [1, constants.TILEDB_VAR_NUM, 2]
cell_counts = [1, 1, 2]
compression = [constants.TILEDB_NO_COMPRESSION, constants.TILEDB_NO_COMPRESSION, constants.TILEDB_NO_COMPRESSION, constants.TILEDB_NO_COMPRESSION]
extents = [2, 2]
types = [constants.TILEDB_INT32, constants.TILEDB_INT32, constants.TILEDB_INT32, constants.TILEDB_INT32]

schema = tiledb.Schema(path, attrs, 2, constants.TILEDB_ROW_MAJOR, cell_counts, 
	compression, 
	1, dims, domain, extents, constants.TILEDB_ROW_MAJOR, 
	types)

print schema
print schema.name
print schema.capacity
print schema.cell_order
print schema.tile_order
print schema.dense
print schema.attributes
print schema.values
print schema.compression
print schema.dimensions
print schema.types
print schema.domain
print schema.coordinate_type
print schema.coordinate_compression
print schema.extents

#context.create_array() #schema)

array = tiledb.Array(context, schema, path, constants.TILEDB_ARRAY_WRITE, None, None)
print array


array.write([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
#array.write([np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), [[0],[1,2,3],[4,5,6,7,8,9,10,11,12,13,14,15]], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
#array.write([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

array2 = tiledb.Array(context, path, constants.TILEDB_ARRAY_READ, None, None)
print array2

bufs = [np.zeros(256, dtype=np.int32), np.zeros(256, dtype=np.int32), np.zeros(256, dtype=np.int32)]
array2.read(bufs)

print bufs[0]


#Schema,      str,         list, int,  int, NoneType, list, int, list, list, list, int, list)
#api::object, std::string, list, long, int, list,     list, int, list, list, list, int, list)