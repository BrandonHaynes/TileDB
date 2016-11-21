#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION //TODO why?

#include <stdint.h>
#include <map>
#include <string>
#include <python2.7/Python.h>
#include <python2.7/numpy/ndarrayobject.h>
#include <boost/make_shared.hpp>
#include <boost/python/import.hpp>
#include <boost/python/module.hpp>
#include <boost/python/class.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/def.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <mpich/mpi.h>
#include "c_api.h"
#include <vector>

namespace python = boost::python;

template<typename TValue, typename TCast = TValue>
static python::list make_list(const TValue* values, const size_t count)
{
	python::list list;

	for (auto i = 0; i < count; i++)
		list.append(static_cast<TCast>(values[i]));
	return list;
}

template<typename T>
static void extract_value(char** current, python::object value)
{
	*reinterpret_cast<T*>(*current) = python::extract<T>(value);
	*current += sizeof(T);
}

template<typename T, size_t Limit>
static void extract_indexable(char** current, python::object value)
{
	for (auto i = 0; i < python::len(value) && i < Limit; i++)
		extract_value<T>(current, value[i]);
}

static const std::map<const int, const size_t> attribute_sizes =
{
	{ TILEDB_INT32,   sizeof(int32_t) },
	{ TILEDB_INT64,   sizeof(int64_t) },
	{ TILEDB_FLOAT32, sizeof(float) },
	{ TILEDB_FLOAT64, sizeof(double) },
};

static const std::map<const int, const std::function<void(char**, python::object)>> extent_extractors =
{
	{ TILEDB_INT32,   extract_value<int32_t> },
	{ TILEDB_INT64,   extract_value<int64_t> },
	{ TILEDB_FLOAT32, extract_value<float> },
	{ TILEDB_FLOAT64, extract_value<double> },
};

static const std::map<const int, const std::function<void(char**, python::object)>> domain_extractors =
{
	{ TILEDB_INT32,   extract_indexable<int32_t, 2> },
	{ TILEDB_INT64,   extract_indexable<int64_t, 2> },
	{ TILEDB_FLOAT32, extract_indexable<float, 2> },
	{ TILEDB_FLOAT64, extract_indexable<double, 2> },
};

class Schema {
	friend class Context;

public:
	explicit Schema(TileDB_ArraySchema& schema) : 
		schema_(schema),
		variable_attributes_(std::count_if(schema_.cell_val_num_, schema_.cell_val_num_ + schema_.attribute_num_, [](int v) { return v == TILEDB_VAR_NUM; }))
	{ }
	~Schema() {
		if(tiledb_array_free_schema(&schema_) != TILEDB_OK)
			throw boost::enable_current_exception(std::runtime_error(tiledb_errmsg));
	}

	const TileDB_ArraySchema* handle() const { return &schema_; }
	std::string name() const { return std::string(schema_.array_name_); }
	long capacity() const { return schema_.capacity_; }
	int cell_order() const { return schema_.cell_order_; }
	int tile_order() const { return schema_.tile_order_; }
	int coordinate_type() const { return schema_.types_[schema_.attribute_num_]; }
	int coordinate_compression() const { return schema_.compression_[schema_.attribute_num_]; }
	bool dense() const { return schema_.dense_; }
	python::list attributes() const { return make_list<char*, const char*>(schema_.attributes_, schema_.attribute_num_); }
	python::list values() const { return make_list<int>(schema_.cell_val_num_, schema_.attribute_num_); }
	python::list compression() const { return make_list<int>(schema_.compression_, schema_.attribute_num_); }
	python::list dimensions() const { return make_list<char*, const char*>(schema_.dimensions_, schema_.dim_num_); }
	python::list types() const { return make_list<int>(schema_.types_, schema_.attribute_num_); }
	size_t write_buffers_required() const { return schema_.attribute_num_ + variable_attributes_; }
	python::list domain() const
	{
		python::list list;
		auto* current = static_cast<char*>(schema_.domain_);
		auto next_element = domain_factory_.at(coordinate_type());

		for (auto i = 0; i < schema_.dim_num_; i++)
			list.append(next_element(&current));

		return list;
	}

	python::list extents() const
	{
		python::list list;
		auto* current = static_cast<char*>(schema_.tile_extents_);
		auto next_element = extent_factory_.at(coordinate_type());

		for (auto i = 0; i < schema_.dim_num_; i++)
			list.append(next_element(&current));

		return list;
	}

private:
	const std::map<int, std::function<python::object(char**)>> domain_factory_ =
	{
		{ TILEDB_INT32,   next_tuple<int32_t> },
		{ TILEDB_INT64,   next_tuple<int64_t> },
		{ TILEDB_FLOAT32, next_tuple<float> },
		{ TILEDB_FLOAT64, next_tuple<double> },
	};
	const std::map<const int, const std::function<python::object(char**)>> extent_factory_ =
	{
		{ TILEDB_INT32,   next_value<int32_t> },
		{ TILEDB_INT64,   next_value<int64_t> },
		{ TILEDB_FLOAT32, next_value<float> },
		{ TILEDB_FLOAT64, next_value<double> },
	};

	template<typename T>
	static python::object next_value(char** data)
	{
		python::object object(*reinterpret_cast<T*>(*data));
		*data += sizeof(T);
		return object;
	}
	template<typename T>
	static python::tuple next_tuple(char** data)  { return python::make_tuple(next_value<T>(data), next_value<T>(data)); }

	TileDB_ArraySchema schema_;
	const size_t variable_attributes_;
};

class Context {
public:
	explicit Context(TileDB_CTX* context) : context_(context) {}
	~Context()
	{
		//TODO throwing in a destructor?
		if (context_ && tiledb_ctx_finalize(context_) != TILEDB_OK)
			throw boost::enable_current_exception(std::runtime_error(tiledb_errmsg));
	}

	TileDB_CTX* handle() const { return context_; }

	void create_workspace(const char* name) const
	{
		if (tiledb_workspace_create(context_, name) != TILEDB_OK)
			throw boost::enable_current_exception(std::runtime_error(tiledb_errmsg));
	}

	void create_group(const char* name) const {
		if (tiledb_group_create(context_, name) != TILEDB_OK)
			throw boost::enable_current_exception(std::runtime_error(tiledb_errmsg));
	}

	void create_array(const Schema* schema) const {
		if (tiledb_array_create(context_, schema->handle()) != TILEDB_OK)
			throw boost::enable_current_exception(std::runtime_error(tiledb_errmsg));
	}

private:
	TileDB_CTX* context_;
};

#define DEFAULT_BUFFER_SIZE 4096
class Array
{
public:
	//Array(Context& context, Schema& schema, std::string& directory, int mode) :
	//	Array(context, schema, directory, mode, python::object(), python::object())
	//	{ }

	Array(const boost::shared_ptr<Context> context, const std::string& directory, const int mode,
		  const python::object& subarray, const python::object& attributes) :
		schema_(load_schema(context, directory)),
	    array_(create_array(context, nullptr, directory, mode, subarray, attributes, false)),
		buffers_(create_buffers(*schema_)),
		sizes_(schema_->write_buffers_required(), DEFAULT_BUFFER_SIZE),
		mode_(mode)
	{ }

	Array(const boost::shared_ptr<Context> context, const boost::shared_ptr<Schema> schema, const std::string& directory, const int mode,
		const python::object& subarray, const python::object& attributes) :
		schema_(schema),
	    array_(create_array(context, schema, directory, mode, subarray, attributes, true)),
		buffers_(create_buffers(*schema_)),
		sizes_(schema_->write_buffers_required(), DEFAULT_BUFFER_SIZE),
		mode_(mode)
	{ }

	~Array()
	{
		for (auto i = 0; i < schema_->write_buffers_required(); i++)
			free(buffers_[i]);
		free(buffers_);

		if (array_ && tiledb_array_finalize(array_) != TILEDB_OK)
			throw boost::enable_current_exception(std::runtime_error(tiledb_errmsg));
	}

	const Schema& schema() const { return *schema_; }

	python::numeric::array read(python::list& arrays) const
	{
		if (schema_->write_buffers_required() != python::len(arrays))
			throw boost::enable_current_exception(std::runtime_error("Incorrect number of buffers provided"));
		else
		{
			//TODO why is this required?
			python::numeric::array::set_module_and_type("numpy", "ndarray");

			void* buffers[schema_->write_buffers_required()];
			size_t sizes[schema_->write_buffers_required()];
			sizes[0] = 999;
			for (auto i = 0; i < schema_->write_buffers_required(); i++)
			{
				const python::numeric::array& a = python::extract<python::numeric::array>(arrays[i]);
				//throw boost::enable_current_exception(std::runtime_error("foo" + std::to_string(python::extract<int32_t>(a[1]))));

				//auto *na = (PyArrayObject*)a.ptr(); //TODO PyArray_GETCONTIGUOUS((PyArrayObject*)a.ptr());
				auto *na = PyArray_GETCONTIGUOUS((PyArrayObject*)a.ptr());
				buffers[i] = PyArray_DATA(na); //TODO the cast shouldn't be required; if required use c++ cast
				sizes[i] = PyArray_NBYTES(na); // PyArray_MultiplyList(new npy_intp[3]{ 3, 3, 0 }, 1);
			}

			if (tiledb_array_read(array_, buffers, sizes) != TILEDB_OK)
				throw boost::enable_current_exception(std::runtime_error(tiledb_errmsg));
			//else
			///*return */ bool overflow = tiledb_array_overflow(array_, 0);

			npy_intp dims[1]{ (npy_intp)schema_->write_buffers_required() }; //TODO fix cast
			auto arr = PyArray_Return((PyArrayObject*)PyArray_SimpleNewFromData(1, dims, NPY_UINT64, sizes)); //TODO fix cast

			boost::python::handle<> handle(arr);
			boost::python::numeric::array farr(handle);
			return farr;


			//npy_intp dims[1]{ (npy_intp)schema_->write_buffers_required() }; //TODO fix cast
			//return PyArray_Return((PyArrayObject*)PyArray_SimpleNewFromData(1, dims, NPY_UINT32, sizes)); //TODO fix cast
			//return PyArray_Return(output = PyArray_FromDims(3, sizes, PyArray_LONG));
		}
	}

	//TODO optimize for ndarrays
	void write(python::list& values) const
	{
		auto count = schema_->write_buffers_required();
		size_t sizes[count];
		void **buffers;

		if(schema_->handle()->attribute_num_ != python::len(values))
			throw boost::enable_current_exception(std::runtime_error("Incorrect number of attribute lists provided"));
		else if (mode_ == TILEDB_ARRAY_WRITE)
			buffers = append(sizes, values);
		else if (mode_ == TILEDB_ARRAY_WRITE_UNSORTED)
			buffers = write_unsorted(sizes, values);
		else
			throw boost::enable_current_exception(std::runtime_error("Unsupported write mode."));

		if (tiledb_array_write(array_, const_cast<const void**>(buffers_), sizes) != TILEDB_OK) //TODO const_cast
			throw boost::enable_current_exception(std::runtime_error(tiledb_errmsg));

		//auto count = schema_->write_buffers_required();
		//const void *buffers[count];
		//size_t sizes[count];

		//assert(count == python::len(values));

		//for (auto i = 0, j = 0; i < python::len(values); i++, j++)
		//	if (schema_->handle()->cell_val_num_[i] == TILEDB_VAR_NUM)
		//		copy_variable_buffer(&j, schema_->handle()->types_[j], &sizes[j], values[i]);
		//	else
		//		copy_fixed_buffer(j, schema_->handle()->types_[j], &sizes[j], values[i]);

		//if (tiledb_array_write(array_, const_cast<const void**>(buffers_), sizes) != TILEDB_OK)
		//	throw boost::enable_current_exception(std::runtime_error(tiledb_errmsg));
	}

	void consolidate() const
	{
		if (tiledb_array_consolidate(context_->handle(), schema_->name().c_str()) != TILEDB_OK)
			throw boost::enable_current_exception(std::runtime_error(tiledb_errmsg));
	}

private:
	static boost::shared_ptr<Schema> load_schema(const boost::shared_ptr<Context> context, const std::string& directory)
	{
		auto schema = TileDB_ArraySchema();

		if (tiledb_array_load_schema(context->handle(), directory.c_str(), &schema) != TILEDB_OK)
			throw boost::enable_current_exception(std::runtime_error(tiledb_errmsg));
		else
			return boost::make_shared<Schema>(schema);
	}

	static TileDB_Array* create_array(const boost::shared_ptr<Context> context, const boost::shared_ptr<Schema> schema, const std::string& directory, const int mode,
									  const python::object& subarray, const python::object& attributes, const bool create)
	{
		//TODO throwing in a constructor?
		TileDB_Array* array_;
		void* subarray_ = nullptr;
		const char** attributes_ = nullptr;

		if (create && tiledb_array_create(context->handle(), schema->handle()) != TILEDB_OK)
			throw boost::enable_current_exception(std::runtime_error(tiledb_errmsg));
		//TODO expose attributes parameter
		else if(tiledb_array_init(context->handle(), &array_, directory.c_str(), mode, subarray_, attributes_, 0) != TILEDB_OK)
			throw boost::enable_current_exception(std::runtime_error(tiledb_errmsg));
		//array_(tiledb_array_create(context.handle(), schema.handle())) { }
		return array_;
	}

	void** append(size_t *sizes, python::list& values) const
	{
		for (auto i = 0, j = 0; i < python::len(values); i++, j++)
			if (schema_->handle()->cell_val_num_[i] == TILEDB_VAR_NUM)
				copy_variable_buffer(&j, schema_->handle()->types_[j], &sizes[j], values[i]);
			else
				copy_fixed_buffer(j, schema_->handle()->types_[j], &sizes[j], values[i]);

		return buffers_;
	}

	void** write_unsorted(size_t *sizes, python::list& values) const
	{
		throw boost::enable_current_exception(std::runtime_error("Not implemented."));
	}

	void copy_fixed_buffer(const size_t index, const int type, size_t *sizes, const python::object& values) const
	{
		auto length = python::len(values);
		*sizes = length * attribute_sizes.at(type);
		auto *current = get_buffer(index, *sizes);
		//auto *current = new char[*sizes];
		//*buffer = current;
		//throw boost::enable_current_exception(std::runtime_error("qwer" + std::to_string(*reinterpret_cast<int32_t*>(buffers_[0]))));
		for (auto j = 0; j < length; j++)
			extent_extractors.at(type)(&current, values[j]); // rename extent_extractors
															 //todo free
	}

	void copy_variable_buffer(int *index, const int type, size_t *sizes, const python::object& values) const
	{
		auto length = python::len(values);
		sizes[0] = length * attribute_sizes.at(TILEDB_INT32);
		//auto *offest = new int32_t[sizes[0]];
		auto *offset = get_buffer<int32_t*>(*index, sizes[0]);
		//buffer[0] = offset;

		auto total_length = 0;
		for (auto i = 0; i < length; i++) {
			auto current_length = python::len(values[i]);
			*offset++ = total_length * attribute_sizes.at(type);
			total_length += current_length;
		}

		sizes[1] = total_length * attribute_sizes.at(type);
		//auto current = new char[sizes[1]];
		auto *current = get_buffer(*index + 1, sizes[1]);
		//buffer[1] = current;

		for (auto i = 0; i < length; i++)
			for (auto j = 0; j < python::len(values[i]); j++)
				extent_extractors.at(type)(&current, values[i][j]); //TODO rename extent_extractors

		++*index; //TODO
	}

	static void** create_buffers(Schema& schema)
	{
		auto size = schema.write_buffers_required();
		auto buffers = new void*[size];

		for (auto i = 0; i < size; i++)
			buffers[i] = new char[DEFAULT_BUFFER_SIZE];

		return buffers;
	}

	template<typename T = char*>
	T get_buffer(const size_t index, const size_t size_required) const
	{
		return reinterpret_cast<T>(sizes_.at(index) >= size_required
			? buffers_[index]
			: realloc(buffers_[index], size_required));
	}

	boost::shared_ptr<Context> context_;
	boost::shared_ptr<Schema> schema_;
	TileDB_Array* array_ = nullptr;
	//TODO use vectors instead with a custom boost::python::extract converter
	mutable void **buffers_;
	const std::vector<size_t> sizes_;
	const int mode_;
};

static boost::shared_ptr<TileDB_Config> make_configuration(const char* home, boost::python::object communicator,
														   int read_method, int write_method)
{
    MPI_Comm* comm;

    //TODO switch ternary operator
    if(communicator.is_none())
    	comm = nullptr;
    else
    	//TODO test this
    	comm = python::extract<MPI_Comm*>(python::import("mpi4py.MPI").attr("_addressof")(communicator));

	return boost::make_shared<TileDB_Config>(TileDB_Config{home, comm, read_method, write_method});
}

static boost::shared_ptr<Context> make_context(TileDB_Config* configuration)
{
    TileDB_CTX* context;

    if(tiledb_ctx_init(&context, configuration) == TILEDB_OK)
    	return boost::make_shared<Context>(context);
    else
    	PyErr_SetString(PyExc_RuntimeError, "tiledb_ctx_init returned TILEDB_ERR");
}

//TODO accept none for cell_Value_num and other applicable fields
static boost::shared_ptr<Schema> make_schema(const std::string& array_name, 
	const python::list& attributes, const long capacity, 
	const int cell_order, const python::list& cell_val_num, 
	const python::list& compression, const int dense, 
	const python::list& dimensions, const python::list& domain, 
	const python::list& tile_extents, const int tile_order, 
	const python::list& types)
{
	//auto* schema = new TileDB_ArraySchema();
	TileDB_ArraySchema schema;
	const char* attributes_[python::len(attributes)];
	int cell_val_num_[python::len(cell_val_num)];
	int compression_[python::len(compression)];
	const char* dimensions_[python::len(dimensions)];
	int types_[python::len(types)];
	int64_t domain_[python::len(domain)];
	int64_t tile_extents_[python::len(tile_extents)];
	int coordinate_type = python::extract<int32_t>(types[python::len(types) - 1]);

	for(auto i = 0; i < python::len(attributes); i++)
		attributes_[i] = python::extract<const char*>(attributes[i]);

	for(auto i = 0; i < python::len(cell_val_num); i++)
		cell_val_num_[i] = python::extract<int>(cell_val_num[i]);

	for(auto i = 0; i < python::len(compression); i++)
		compression_[i] = python::extract<int>(compression[i]);

	for(auto i = 0; i < python::len(dimensions); i++)
		dimensions_[i] = python::extract<const char*>(dimensions[i]);

	for(auto i = 0; i < python::len(types); i++)
		types_[i] = python::extract<int>(types[i]);

	char* current_domain = reinterpret_cast<char*>(domain_);
	for(auto i = 0; i < python::len(domain); i++)
		domain_extractors.at(coordinate_type)(&current_domain, domain[i]);

	char* current_extent = reinterpret_cast<char*>(tile_extents_);
	for (auto i = 0; i < python::len(tile_extents); i++)
		extent_extractors.at(coordinate_type)(&current_extent, tile_extents[i]);

	if(tiledb_array_set_schema(&schema, array_name.c_str(), attributes_, python::len(attributes), capacity, cell_order, 
		cell_val_num_, compression_, dense, dimensions_, python::len(dimensions), domain_, current_domain - reinterpret_cast<char*>(domain_),
		tile_extents_, current_extent - reinterpret_cast<char*>(tile_extents_), tile_order, types_) != TILEDB_OK)
    	PyErr_SetString(PyExc_RuntimeError, "tiledb_array_set_schema returned TILEDB_ERR");
    else
		return boost::make_shared<Schema>(schema);
}


//TODO constify
//TODO create overloads; see BOOST_PYTHON_FUNCTION_OVERLOADS
static boost::shared_ptr<Array> make_array(const boost::shared_ptr<Context> context, const boost::shared_ptr<Schema> schema, const std::string& directory, const int mode,
	                                       const python::object& subarray, const python::object& attributes)
{
	return boost::make_shared<Array>(context, schema, directory, mode, subarray, attributes);
}


//TODO constify
static boost::shared_ptr<Array> make_array_load(const boost::shared_ptr<Context> context, const std::string& directory, const int mode,
	const python::object& subarray, const python::object& attributes)
{
	return boost::make_shared<Array>(context, directory, mode, subarray, attributes);
}


BOOST_PYTHON_MODULE(core)
{
	import_array(); //TODO move elsewhere?

	python::class_<TileDB_Config>("Configuration", python::no_init)
        .def("__init__", make_constructor(make_configuration))
    	.def_readonly("home", &TileDB_Config::home_)
    	.def_readwrite("communicator", &TileDB_Config::mpi_comm_)
    	.def_readwrite("read_method", &TileDB_Config::read_method_)
    	.def_readwrite("write_method", &TileDB_Config::write_method_);

    python::class_<Context, Context*>("Context", python::no_init)
        .def("__init__", python::make_constructor(make_context))
        .def("create_workspace", &Context::create_workspace)
		.def("create_group", &Context::create_group)
		.def("create_array", &Context::create_array);

	python::class_<Schema>("Schema", python::no_init)
		.def("__init__", make_constructor(make_schema))
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

	python::class_<Array>("Array", python::no_init)
		.def("__init__", make_constructor(make_array))
		.def("__init__", make_constructor(make_array_load))
		.def("write", &Array::write)
		.def("read", &Array::read);
}
