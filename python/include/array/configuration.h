#ifndef __PYTHON_CONFIGURATION_H__
#define __PYTHON_CONFIGURATION_H__

#include <memory>

#include <boost/python/import.hpp>
#include <boost/python/extract.hpp>

#include "c_api.h"

class Configuration {
public:
	static std::shared_ptr<TileDB_Config> create(const char* home, boost::python::object communicator,
		int read_method, int write_method)
	{
		auto* comm = communicator.is_none()
			? nullptr
			: static_cast<MPI_Comm*>(boost::python::extract<MPI_Comm*>(boost::python::import("mpi4py.MPI").attr("_addressof")(communicator)));

		return std::make_shared<TileDB_Config>(TileDB_Config{ home, comm, read_method, write_method });
	}

private:
	Configuration() {}
};

#endif