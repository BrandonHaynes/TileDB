#ifndef __PYTHON_ARRAY_H__
#define __PYTHON_ARRAY_H__

#include <vector>

#include "context.h"
#include "schema.h"

class Array
{
public:
	static constexpr size_t DefaultBufferSize = 4096;

	static std::shared_ptr<Array> create(
		const std::shared_ptr<Context> context, const std::shared_ptr<Schema> schema,
		const std::string& directory, const int mode,
		const boost::python::object& subarray, const boost::python::object& attributes)
	{
		return std::make_shared<Array>(context, schema, directory, mode, subarray, attributes);
	}

	static std::shared_ptr<Array> create(
		const std::shared_ptr<Context> context, const std::string& directory, const int mode,
		const boost::python::object& subarray, const boost::python::object& attributes)
	{
		return std::make_shared<Array>(context, directory, mode, subarray, attributes);
	}

	Array(const std::shared_ptr<Context> context, const std::string& directory, const int mode,
		const boost::python::object& subarray, const boost::python::object& attributes) :
		schema_(load_schema(context, directory)),
		array_(create_array(context, nullptr, directory, mode, subarray, attributes, false)),
		buffers_(create_buffers(*schema_)),
		sizes_(schema_->write_buffers_required(), DefaultBufferSize),
		mode_(mode)
	{ }

	Array(const std::shared_ptr<Context> context, const std::shared_ptr<Schema> schema, const std::string& directory, const int mode,
		const boost::python::object& subarray, const boost::python::object& attributes) :
		schema_(schema),
		array_(create_array(context, schema, directory, mode, subarray, attributes, true)),
		buffers_(create_buffers(*schema_)),
		sizes_(schema_->write_buffers_required(), DefaultBufferSize),
		mode_(mode)
	{ }

	~Array()
	{
		for (auto i = 0; i < schema_->write_buffers_required(); i++)
			free(buffers_[i]);
		free(buffers_);

		if (array_ && tiledb_array_finalize(array_) != TILEDB_OK)
			throw std::runtime_error(tiledb_errmsg);
	}

	const Schema& schema() const { return *schema_; }

	boost::python::numeric::array read(boost::python::list& arrays) const;
	void write(boost::python::list& values) const;
	void consolidate() const;

private:
	static std::shared_ptr<Schema> load_schema(const std::shared_ptr<Context> context, const std::string& directory);

	static TileDB_Array* create_array(const std::shared_ptr<Context> context, const std::shared_ptr<Schema> schema, 
		const std::string& directory, const int mode,
		const boost::python::object& subarray, const boost::python::object& attributes, const bool create);

	void** append(std::vector<size_t>& sizes, boost::python::list& values) const;

	static void** write_unsorted(const std::vector<size_t>& sizes, boost::python::list& values) //const
	{
		throw std::runtime_error("Not implemented.");
	}

	void copy_fixed_buffer(const size_t index, const int type, size_t *sizes, const boost::python::object& values) const;

	void copy_variable_buffer(int *index, const int type, size_t *sizes, const boost::python::object& values) const;

	static void** create_buffers(Schema& schema)
	{
		auto size = schema.write_buffers_required();
		auto buffers = new void*[size];

		for (auto i = 0; i < size; i++)
			buffers[i] = new char[DefaultBufferSize];

		return buffers;
	}

	template<typename T = char*>
	T get_buffer(const size_t index, const size_t size_required) const
	{
		return reinterpret_cast<T>(sizes_.at(index) >= size_required
			? buffers_[index]
			: realloc(buffers_[index], size_required));
	}

	std::shared_ptr<Context> context_;
	const std::shared_ptr<Schema> schema_;
	TileDB_Array* array_ = nullptr;
	//TODO use vectors instead with a custom boost::python::extract converter
	mutable void **buffers_;
	const std::vector<size_t> sizes_;
	const int mode_;
};

#endif