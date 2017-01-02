#include <vector>

#include <boost/python/extract.hpp>

#include "schema.h"
#include "extractors.h"
#include <boost/python/stl_iterator.hpp>
#include <numeric>

using boost::python::stl_input_iterator;
using boost::python::object;

template<typename T>
std::vector<T> to_vector(const object& object)
{
	return std::vector<T>((stl_input_iterator<T>(object)), stl_input_iterator<T>());
}

std::vector<char> to_byte_vector(const object& source, 
	                             const std::function<char**(char**, object)> f)
{
	std::vector<char> vector(len(source) * sizeof(int64_t));
	auto end = vector.data();
	std::accumulate(stl_input_iterator<object>(source),
		            stl_input_iterator<object>(), 
				    &end,
		            f);
	vector.resize(end - vector.data());
	return vector;
}

std::shared_ptr<Schema> Schema::create(const std::string& array_name,
	const object& attributes, const long capacity,
	const int cell_order, const object& cell_val_num,
	const object& compression, const int dense,
	const object& dimensions, const object& domain,
	const object& tile_extents, const int tile_order,
	const object& types)
{
	TileDB_ArraySchema schema;
	int coordinate_type = boost::python::extract<int32_t>(types[len(types) - 1]);
	auto attributes_(to_vector<const char*>(attributes));
	auto cell_val_num_(to_vector<int>(cell_val_num));
	auto compression_(to_vector<int>(compression));
	auto dimensions_(to_vector<const char*>(dimensions));
	auto types_(to_vector<int>(types));
	auto domain_(to_byte_vector(domain, domain_extractors.at(coordinate_type)));
	auto tile_extents_(to_byte_vector(tile_extents, extent_extractors.at(coordinate_type)));

	if (tiledb_array_set_schema(&schema, array_name.c_str(), attributes_.data(), len(attributes), capacity, cell_order,
		cell_val_num_.data(), compression_.data(), dense, dimensions_.data(), len(dimensions), domain_.data(), domain_.size(),
		tile_extents_.data(), tile_extents_.size(), tile_order, types_.data()) != TILEDB_OK)
		throw std::runtime_error(tiledb_errmsg);
	else
		return std::make_shared<Schema>(schema);
}
