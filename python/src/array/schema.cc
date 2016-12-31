#include <vector>

#include <boost/python/extract.hpp>

#include "schema.h"
#include "extractors.h"

std::shared_ptr<Schema> Schema::create(const std::string& array_name,
	const boost::python::list& attributes, const long capacity,
	const int cell_order, const boost::python::list& cell_val_num,
	const boost::python::list& compression, const int dense,
	const boost::python::list& dimensions, const boost::python::list& domain,
	const boost::python::list& tile_extents, const int tile_order,
	const boost::python::list& types)
{
	//auto* schema = new TileDB_ArraySchema();
	TileDB_ArraySchema schema;
	std::vector<const char*> attributes_(boost::python::len(attributes));
	//const char* attributes_[python::len(attributes)];
	std::vector<int> cell_val_num_(boost::python::len(cell_val_num));
	//int cell_val_num_[python::len(cell_val_num)];
	std::vector<int> compression_(boost::python::len(compression));
	//int compression_[python::len(compression)];
	std::vector<const char*> dimensions_(boost::python::len(dimensions));
	//const char* dimensions_[python::len(dimensions)];
	std::vector<int> types_(boost::python::len(types));
	//int types_[python::len(types)];
	std::vector<int64_t> domain_(boost::python::len(domain));
	//int64_t domain_[python::len(domain)];
	std::vector<int64_t> tile_extents_(boost::python::len(tile_extents));
	//int64_t tile_extents_[python::len(tile_extents)];
	int coordinate_type = boost::python::extract<int32_t>(types[boost::python::len(types) - 1]);

	for (auto i = 0; i < boost::python::len(attributes); i++)
		attributes_[i] = boost::python::extract<const char*>(attributes[i]);

	for (auto i = 0; i < boost::python::len(cell_val_num); i++)
		cell_val_num_[i] = boost::python::extract<int>(cell_val_num[i]);

	for (auto i = 0; i < boost::python::len(compression); i++)
		compression_[i] = boost::python::extract<int>(compression[i]);

	for (auto i = 0; i < boost::python::len(dimensions); i++)
		dimensions_[i] = boost::python::extract<const char*>(dimensions[i]);

	for (auto i = 0; i < boost::python::len(types); i++)
		types_[i] = boost::python::extract<int>(types[i]);

	auto* current_domain = reinterpret_cast<char*>(domain_.data());
	for (auto i = 0; i < boost::python::len(domain); i++)
		domain_extractors.at(coordinate_type)(&current_domain, domain[i]);

	auto* current_extent = reinterpret_cast<char*>(tile_extents_.data());
	for (auto i = 0; i < boost::python::len(tile_extents); i++)
		extent_extractors.at(coordinate_type)(&current_extent, tile_extents[i]);

	if (tiledb_array_set_schema(&schema, array_name.c_str(), attributes_.data(), boost::python::len(attributes), capacity, cell_order,
		cell_val_num_.data(), compression_.data(), dense, dimensions_.data(), boost::python::len(dimensions), domain_.data(), current_domain - reinterpret_cast<char*>(domain_.data()),
		tile_extents_.data(), current_extent - reinterpret_cast<char*>(tile_extents_.data()), tile_order, types_.data()) != TILEDB_OK)
		throw std::runtime_error(tiledb_errmsg);
	else
		return std::make_shared<Schema>(schema);
}
