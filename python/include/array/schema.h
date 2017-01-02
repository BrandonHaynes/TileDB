#ifndef __PYTHON_SCHEMA_H__
#define __PYTHON_SCHEMA_H__

#include <algorithm>
#include <map>

#include <boost/python/list.hpp>
#include <boost/python/tuple.hpp>

#include "c_api.h"

class Schema {
public:
	//TODO accept none for cell_Value_num and other applicable fields
	static std::shared_ptr<Schema> create(const std::string& array_name,
		const boost::python::object& attributes, const long capacity,
		const int cell_order, const boost::python::object& cell_val_num,
		const boost::python::object& compression, const int dense,
		const boost::python::object& dimensions, const boost::python::object& domain,
		const boost::python::object& tile_extents, const int tile_order,
		const boost::python::object& types);

	explicit Schema(TileDB_ArraySchema& schema) :
		schema_(schema),
		variable_attributes_(std::count_if(schema_.cell_val_num_, schema_.cell_val_num_ + schema_.attribute_num_, [](int v) { return v == TILEDB_VAR_NUM; }))
	{ }
	~Schema() {
		try { close(); }
		catch(...) {}
	}

	const TileDB_ArraySchema* handle() const { return &schema_; }

	void close()
	{
		if (!closed_ && tiledb_array_free_schema(&schema_) != TILEDB_OK)
			throw std::runtime_error(tiledb_errmsg);
		closed_ = true;
	}

	std::string name() const { return std::string(schema_.array_name_); }
	long capacity() const { return schema_.capacity_; }
	int cell_order() const { return schema_.cell_order_; }
	int tile_order() const { return schema_.tile_order_; }
	int coordinate_type() const { return schema_.types_[schema_.attribute_num_]; }
	int coordinate_compression() const { return schema_.compression_[schema_.attribute_num_]; }
	bool dense() const { return schema_.dense_; }
	boost::python::list attributes() const { return make_list<char*, const char*>(schema_.attributes_, schema_.attribute_num_); }
	boost::python::list values() const { return make_list<int>(schema_.cell_val_num_, schema_.attribute_num_); }
	boost::python::list compression() const { return make_list<int>(schema_.compression_, schema_.attribute_num_); }
	boost::python::list dimensions() const { return make_list<char*, const char*>(schema_.dimensions_, schema_.dim_num_); }
	boost::python::list types() const { return make_list<int>(schema_.types_, schema_.attribute_num_); }
	size_t write_buffers_required() const { return schema_.attribute_num_ + variable_attributes_; }
	boost::python::list domain() const
	{
		boost::python::list list;
		auto* current = static_cast<char*>(schema_.domain_);
		auto next_element = domain_factory_.at(coordinate_type());

		for (auto i = 0; i < schema_.dim_num_; i++)
			list.append(next_element(&current));

		return list;
	}

	boost::python::list extents() const
	{
		boost::python::list list;
		auto* current = static_cast<char*>(schema_.tile_extents_);
		auto next_element = extent_factory_.at(coordinate_type());

		for (auto i = 0; i < schema_.dim_num_; i++)
			list.append(next_element(&current));

		return list;
	}

private:
	template<typename T>
	static boost::python::tuple next_tuple(char** data) { return boost::python::make_tuple(next_value<T>(data), next_value<T>(data)); }

	template<typename T>
	static boost::python::object next_value(char** data)
	{
		boost::python::object object(*reinterpret_cast<T*>(*data));
		*data += sizeof(T);
		return object;
	}

	template<typename TValue, typename TCast = TValue>
	static boost::python::list make_list(const TValue* values, const size_t count)
	{
		boost::python::list list;

		for (auto i = 0; i < count; i++)
			list.append(static_cast<TCast>(values[i]));
		return list;
	}

	const std::map<int, std::function<boost::python::object(char**)>> domain_factory_ =
	{
		{ TILEDB_INT32,   next_tuple<int32_t> },
		{ TILEDB_INT64,   next_tuple<int64_t> },
		{ TILEDB_FLOAT32, next_tuple<float> },
		{ TILEDB_FLOAT64, next_tuple<double> },
	};
	const std::map<const int, const std::function<boost::python::object(char**)>> extent_factory_ =
	{
		{ TILEDB_INT32,   next_value<int32_t> },
		{ TILEDB_INT64,   next_value<int64_t> },
		{ TILEDB_FLOAT32, next_value<float> },
		{ TILEDB_FLOAT64, next_value<double> },
	};

	TileDB_ArraySchema schema_;
	const size_t variable_attributes_;
	bool closed_ = false;
};

#endif