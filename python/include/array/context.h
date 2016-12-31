#ifndef __PYTHON_CONTEXT_H__
#define __PYTHON_CONTEXT_H__

#include "schema.h"
#include "c_api.h"

class Context {
public:
	static std::shared_ptr<Context> create(const TileDB_Config* const configuration)
	{
		TileDB_CTX* context;

		if (tiledb_ctx_init(&context, configuration) == TILEDB_OK)
			return std::make_shared<Context>(context);
		else
			throw std::runtime_error(tiledb_errmsg);
	}

	explicit Context(TileDB_CTX* context) : context_(context) {}
	~Context()
	{
		try { close(); }
		catch (...) {}
	}

	void close()
	{
		if (context_ && tiledb_ctx_finalize(context_) != TILEDB_OK)
			throw std::runtime_error(tiledb_errmsg);
		context_ = nullptr;
	}

	void create_workspace(const char* name) const
	{
		if (tiledb_workspace_create(context_, name) != TILEDB_OK)
			throw std::runtime_error(tiledb_errmsg);
	}

	void create_group(const char* name) const {
		if (tiledb_group_create(context_, name) != TILEDB_OK)
			throw std::runtime_error(tiledb_errmsg);
	}

	void create_array(const Schema* schema) const {
		if (tiledb_array_create(context_, schema->handle()) != TILEDB_OK)
			throw std::runtime_error(tiledb_errmsg);
	}

	TileDB_CTX* handle() const { return context_; }

private:
	TileDB_CTX* context_;
};

#endif