#ifndef __PYTHON_EXTRACTORS_H__
#define __PYTHON_EXTRACTORS_H__

#include <map>
#include <functional>

#include <boost/python/object.hpp>

extern const std::map<const int, const size_t> attribute_sizes;
extern const std::map<const int, const std::function<char**(char**, boost::python::object)>> domain_extractors;
extern const std::map<const int, const std::function<char**(char**, boost::python::object)>> extent_extractors;

#endif