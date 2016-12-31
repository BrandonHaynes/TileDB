#include <map>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL TILEDB_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <python2.7/Python.h>
#include <python2.7/numpy/ndarrayobject.h>
#include <boost/python/numeric.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "array.h"
#include "extractors.h"

boost::python::numeric::array Array::read(boost::python::list& arrays) const
{
	if (schema_->write_buffers_required() != boost::python::len(arrays))
		throw std::runtime_error("Incorrect number of buffers provided");
	else
	{
		std::vector<void*> buffers(schema_->write_buffers_required());
		//void* buffers[schema_->write_buffers_required()];
		//size_t sizes[schema_->write_buffers_required()];
		std::vector<size_t> sizes(schema_->write_buffers_required());
		sizes[0] = 999;
		for (auto i = 0; i < schema_->write_buffers_required(); i++)
		{
			const boost::python::numeric::array& a = boost::python::extract<boost::python::numeric::array>(arrays[i]);

			auto *na = PyArray_GETCONTIGUOUS((PyArrayObject*)a.ptr());
			buffers[i] = PyArray_DATA(na);
			sizes[i] = PyArray_NBYTES(na);
		}

		if (tiledb_array_read(array_, buffers.data(), sizes.data()) != TILEDB_OK)
			throw std::runtime_error(tiledb_errmsg);
		//else
		///*return */ bool overflow = tiledb_array_overflow(array_, 0);

		npy_intp dims[1]{ static_cast<npy_intp>(schema_->write_buffers_required()) };
		auto arr = PyArray_Return(reinterpret_cast<PyArrayObject*>(PyArray_SimpleNewFromData(1, dims, NPY_UINT64, sizes.data())));

		boost::python::handle<> handle(arr);
		boost::python::numeric::array farr(handle);
		return farr;
	}
}

//TODO optimize for ndarrays
void Array::write(boost::python::list& values) const
{
	//auto count = schema_->write_buffers_required();
	std::vector<size_t> sizes(schema_->write_buffers_required());
	//size_t sizes[schema_->write_buffers_required()];
	//void **buffers;

	if (schema_->handle()->attribute_num_ != boost::python::len(values))
		throw std::runtime_error("Incorrect number of attribute lists provided");
	else if (mode_ == TILEDB_ARRAY_WRITE)
		append(sizes, values);
	else if (mode_ == TILEDB_ARRAY_WRITE_UNSORTED)
		write_unsorted(sizes, values);
	else
		throw std::runtime_error("Unsupported write mode.");

	if (tiledb_array_write(array_, const_cast<const void**>(buffers_), sizes.data()) != TILEDB_OK) //TODO const_cast
		throw std::runtime_error(tiledb_errmsg);

	//auto count = schema_->write_buffers_required();
	//const void *buffers[count];
	//size_t sizes[count];
	//
	//assert(count == boost::python::len(values));
	//
	//for (auto i = 0, j = 0; i < boost::python::len(values); i++, j++)
	//	if (schema_->handle()->cell_val_num_[i] == TILEDB_VAR_NUM)
	//		copy_variable_buffer(&j, schema_->handle()->types_[j], &sizes[j], values[i]);
	//	else
	//		copy_fixed_buffer(j, schema_->handle()->types_[j], &sizes[j], values[i]);
	//
	//if (tiledb_array_write(array_, const_cast<const void**>(buffers_), sizes) != TILEDB_OK)
	//	throw boost::enable_current_exception(std::runtime_error(tiledb_errmsg));
}


void Array::consolidate() const
{
	if (tiledb_array_consolidate(context_->handle(), schema_->name().c_str()) != TILEDB_OK)
		throw std::runtime_error(tiledb_errmsg);
}

std::shared_ptr<Schema> Array::load_schema(const std::shared_ptr<Context> context, const std::string& directory)
{
	auto schema = TileDB_ArraySchema();

	if (tiledb_array_load_schema(context->handle(), directory.c_str(), &schema) != TILEDB_OK)
		throw std::runtime_error(tiledb_errmsg);
	else
		return std::make_shared<Schema>(schema);
}


TileDB_Array* Array::create_array(const std::shared_ptr<Context> context, const std::shared_ptr<Schema> schema, const std::string& directory, const int mode,
	const boost::python::object& subarray, const boost::python::object& attributes, const bool create)
{
	TileDB_Array* array_;
	void* subarray_ = nullptr;
	const char** attributes_ = nullptr;

	if (create && tiledb_array_create(context->handle(), schema->handle()) != TILEDB_OK)
		throw std::runtime_error(tiledb_errmsg);
	//TODO expose attributes parameter
	else if (tiledb_array_init(context->handle(), &array_, directory.c_str(), mode, subarray_, attributes_, 0) != TILEDB_OK)
		throw std::runtime_error(tiledb_errmsg);
	//array_(tiledb_array_create(context.handle(), schema.handle())) { }
	return array_;
}

void** Array::append(std::vector<size_t>& sizes, boost::python::list& values) const
{
	for (auto i = 0, j = 0; i < boost::python::len(values); i++, j++)
		if (schema_->handle()->cell_val_num_[i] == TILEDB_VAR_NUM)
			copy_variable_buffer(&j, schema_->handle()->types_[j], &sizes[j], values[i]);
		else
			copy_fixed_buffer(j, schema_->handle()->types_[j], &sizes[j], values[i]);

	return buffers_;
}

void Array::copy_fixed_buffer(const size_t index, const int type, size_t *sizes, const boost::python::object& values) const
{
	auto length = boost::python::len(values);
	*sizes = length * attribute_sizes.at(type);
	auto *current = get_buffer(index, *sizes);
	//auto *current = new char[*sizes];
	//*buffer = current;
	//throw std::runtime_error("qwer" + std::to_string(*reinterpret_cast<int32_t*>(buffers_[0])));
	for (auto j = 0; j < length; j++)
		extent_extractors.at(type)(&current, values[j]); // rename extent_extractors
														 //todo free
}

void Array::copy_variable_buffer(int *index, const int type, size_t *sizes, const boost::python::object& values) const
{
	auto length = boost::python::len(values);
	sizes[0] = length * attribute_sizes.at(TILEDB_INT32);
	//auto *offest = new int32_t[sizes[0]];
	auto *offset = get_buffer<int32_t*>(*index, sizes[0]);
	//buffer[0] = offset;

	auto total_length = 0;
	for (auto i = 0; i < length; i++) {
		auto current_length = boost::python::len(values[i]);
		*offset++ = total_length * attribute_sizes.at(type);
		total_length += current_length;
	}

	sizes[1] = total_length * attribute_sizes.at(type);
	//auto current = new char[sizes[1]];
	auto *current = get_buffer(*index + 1, sizes[1]);
	//buffer[1] = current;

	for (auto i = 0; i < length; i++)
		for (auto j = 0; j < boost::python::len(values[i]); j++)
			extent_extractors.at(type)(&current, values[i][j]); //TODO rename extent_extractors

	++*index; //TODO
}
