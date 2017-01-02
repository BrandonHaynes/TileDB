#include "extractors.h"

#include <boost/python/extract.hpp>

#include "c_api.h"

template<typename T>
static char** extract_value(char** current, boost::python::object value)
{
	*reinterpret_cast<T*>(*current) = boost::python::extract<T>(value);
	*current += sizeof(T);
	return current;
}

template<typename T, size_t Limit>
static char** extract_indexable(char** current, boost::python::object value)
{
	for (auto i = 0; i < boost::python::len(value) && i < Limit; i++)
		extract_value<T>(current, value[i]);
	return current;
}

const std::map<const int, const std::function<char**(char**, boost::python::object)>> domain_extractors =
{
	{ TILEDB_INT32,   extract_indexable<int32_t, 2> },
	{ TILEDB_INT64,   extract_indexable<int64_t, 2> },
	{ TILEDB_FLOAT32, extract_indexable<float, 2> },
	{ TILEDB_FLOAT64, extract_indexable<double, 2> },
};

const std::map<const int, const std::function<char**(char**, boost::python::object)>> extent_extractors =
{
	{ TILEDB_INT32,   extract_value<int32_t> },
	{ TILEDB_INT64,   extract_value<int64_t> },
	{ TILEDB_FLOAT32, extract_value<float> },
	{ TILEDB_FLOAT64, extract_value<double> },
};

const std::map<const int, const size_t> attribute_sizes =
{
	{ TILEDB_INT32,   sizeof(int32_t) },
	{ TILEDB_INT64,   sizeof(int64_t) },
	{ TILEDB_FLOAT32, sizeof(float) },
	{ TILEDB_FLOAT64, sizeof(double) },
};
