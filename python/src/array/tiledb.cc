#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL TILEDB_ARRAY_API

#include <python2.7/Python.h>
#include <python2.7/numpy/ndarrayobject.h>
#include <boost/python/module.hpp>
#include <boost/python/class.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/def.hpp>
#include <boost/python/make_constructor.hpp>

#include "configuration.h"
#include "context.h"
#include "schema.h"
#include "array.h"

class Constants {};

BOOST_PYTHON_MODULE(tiledb)
{
	import_array();
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

	boost::python::class_<TileDB_Config>("Configuration", boost::python::no_init)
        .def("__init__", make_constructor(Configuration::create))
    	.def_readonly("home", &TileDB_Config::home_)
    	.def_readwrite("communicator", &TileDB_Config::mpi_comm_)
    	.def_readwrite("read_method", &TileDB_Config::read_method_)
    	.def_readwrite("write_method", &TileDB_Config::write_method_);

	boost::python::class_<Context, Context*>("Context", boost::python::no_init)
        .def("__init__", boost::python::make_constructor(Context::create))
        .def("create_workspace", &Context::create_workspace)
		.def("create_group", &Context::create_group)
		.def("create_array", &Context::create_array);

	boost::python::class_<Schema>("Schema", boost::python::no_init)
		.def("__init__", make_constructor(Schema::create))
		.def_readonly("name", &Schema::name)
		.def_readonly("capacity", &Schema::capacity)
		.def_readonly("cell_order", &Schema::cell_order)
		.def_readonly("tile_order", &Schema::tile_order)
		.def_readonly("dense", &Schema::dense)
		.def_readonly("attributes", &Schema::attributes)
		.def_readonly("values", &Schema::values)
		.def_readonly("compression", &Schema::compression)
		.def_readonly("dimensions", &Schema::dimensions)
		.def_readonly("domain", &Schema::domain)
		.def_readonly("extents", &Schema::extents)
		.def_readonly("types", &Schema::coordinate_type)
		.def_readonly("coordinate_type", &Schema::coordinate_type)
		.def_readonly("coordinate_compression", &Schema::coordinate_compression);

	boost::python::class_<Array>("Array", boost::python::no_init)
		.def("__init__", make_constructor(static_cast<std::shared_ptr<Array> (*)(const std::shared_ptr<Context>, const std::shared_ptr<Schema>,
			const std::string&, const int,
			const boost::python::object&, const boost::python::object&)>(Array::create)))
		.def("__init__", make_constructor(static_cast<std::shared_ptr<Array>(*)(const std::shared_ptr<Context>,
			const std::string&, const int,
			const boost::python::object&, const boost::python::object&)>(Array::create)))
		.def("write", &Array::write)
		.def("read", &Array::read);

	boost::python::scope constants = boost::python::class_<Constants>("constants");

	constants.attr("TILEDB_VERSION") = TILEDB_VERSION;

	constants.attr("TILEDB_OK") = TILEDB_OK;
	constants.attr("TILEDB_ERR") = TILEDB_ERR;

	constants.attr("TILEDB_ARRAY_READ") = TILEDB_ARRAY_READ;
	constants.attr("TILEDB_ARRAY_WRITE") = TILEDB_ARRAY_WRITE;
	constants.attr("TILEDB_ARRAY_WRITE_UNSORTED") = TILEDB_ARRAY_WRITE_UNSORTED;

	constants.attr("TILEDB_METADATA_READ") = TILEDB_METADATA_READ;
	constants.attr("TILEDB_METADATA_WRITE") = TILEDB_METADATA_WRITE;

	constants.attr("TILEDB_WORKSPACE") = TILEDB_WORKSPACE;
	constants.attr("TILEDB_GROUP") = TILEDB_GROUP;
	constants.attr("TILEDB_ARRAY") = TILEDB_ARRAY;
	constants.attr("TILEDB_METADATA") = TILEDB_METADATA;

	constants.attr("TILEDB_IO_MMAP") = TILEDB_IO_MMAP;
	constants.attr("TILEDB_IO_READ") = TILEDB_IO_READ;
	constants.attr("TILEDB_IO_MPI") = TILEDB_IO_MPI;
	constants.attr("TILEDB_IO_WRITE") = TILEDB_IO_WRITE;

	constants.attr("TILEDB_AIO_ERR") = TILEDB_AIO_ERR;
	constants.attr("TILEDB_AIO_COMPLETED") = TILEDB_AIO_COMPLETED;
	constants.attr("TILEDB_AIO_INPROGRESS") = TILEDB_AIO_INPROGRESS;
	constants.attr("TILEDB_AIO_OVERFLOW") = TILEDB_AIO_OVERFLOW;

	constants.attr("TILEDB_NAME_MAX_LEN") = TILEDB_NAME_MAX_LEN;

	constants.attr("TILEDB_CONSOLIDATION_BUFFER_SIZE") = TILEDB_CONSOLIDATION_BUFFER_SIZE;

	constants.attr("TILEDB_EMPTY_INT32") = TILEDB_EMPTY_INT32;
	constants.attr("TILEDB_EMPTY_INT64") = TILEDB_EMPTY_INT64;
	constants.attr("TILEDB_EMPTY_FLOAT32") = TILEDB_EMPTY_FLOAT32;
	constants.attr("TILEDB_EMPTY_FLOAT64") = TILEDB_EMPTY_FLOAT64;
	constants.attr("TILEDB_EMPTY_CHAR") = TILEDB_EMPTY_CHAR;

	constants.attr("TILEDB_VAR_NUM") = TILEDB_VAR_NUM;
	constants.attr("TILEDB_VAR_SIZE") = TILEDB_VAR_SIZE;

	constants.attr("TILEDB_INT32") = TILEDB_INT32;
	constants.attr("TILEDB_INT64") = TILEDB_INT64;
	constants.attr("TILEDB_FLOAT32") = TILEDB_FLOAT32;
	constants.attr("TILEDB_FLOAT64") = TILEDB_FLOAT64;
	constants.attr("TILEDB_CHAR") = TILEDB_CHAR;

	constants.attr("TILEDB_ROW_MAJOR") = TILEDB_ROW_MAJOR;
	constants.attr("TILEDB_COL_MAJOR") = TILEDB_COL_MAJOR;
	constants.attr("TILEDB_HILBERT") = TILEDB_HILBERT;

	constants.attr("TILEDB_NO_COMPRESSION") = TILEDB_NO_COMPRESSION;
	constants.attr("TILEDB_GZIP") = TILEDB_GZIP;

	constants.attr("TILEDB_COORDS") = TILEDB_COORDS;
	constants.attr("TILEDB_KEY") = TILEDB_KEY;

	constants.attr("TILEDB_FILE_SUFFIX") = TILEDB_FILE_SUFFIX;
	constants.attr("TILEDB_GZIP_SUFFIX") = TILEDB_GZIP_SUFFIX;

	constants.attr("TILEDB_GZIP_CHUNK_SIZE") = TILEDB_GZIP_CHUNK_SIZE;

	constants.attr("TILEDB_ARRAY_SCHEMA_FILENAME") = TILEDB_ARRAY_SCHEMA_FILENAME;
	constants.attr("TILEDB_METADATA_SCHEMA_FILENAME") = TILEDB_METADATA_SCHEMA_FILENAME;
	constants.attr("TILEDB_BOOK_KEEPING_FILENAME") = TILEDB_BOOK_KEEPING_FILENAME;
	constants.attr("TILEDB_FRAGMENT_FILENAME") = TILEDB_FRAGMENT_FILENAME;
	constants.attr("TILEDB_GROUP_FILENAME") = TILEDB_GROUP_FILENAME;
	constants.attr("TILEDB_WORKSPACE_FILENAME") = TILEDB_WORKSPACE_FILENAME;

	constants.attr("TILEDB_SORTED_BUFFER_SIZE") = TILEDB_SORTED_BUFFER_SIZE;
	constants.attr("TILEDB_SORTED_BUFFER_VAR_SIZE") = TILEDB_SORTED_BUFFER_VAR_SIZE;
}
