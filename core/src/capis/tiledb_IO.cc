/**
 * @file   tiledb_IO.cc
 * @author Stavros Papadopoulos <stavrosp@csail.mit.edu>
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * Copyright (c) 2014 Stavros Papadopoulos <stavrosp@csail.mit.edu>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * @section DESCRIPTION
 *
 * This file implements the C APIs for basic Input/Output operations with
 * arrays.
 */

#include "loader.h"
#include "query_processor.h"
#include "storage_manager.h"
#include "tiledb_ctx.h"
#include "tiledb_IO.h"
#include "tiledb_error.h"

typedef struct TileDB_CTX {
  Loader* loader_;
  QueryProcessor* query_processor_;
  StorageManager* storage_manager_;
} TileDB_CTX;

int tiledb_close_array(
    TileDB_CTX* tiledb_ctx,
    int ad) {
  // TODO: Error messages here
  tiledb_ctx->storage_manager_->close_array(ad);

  return 0;
}

int tiledb_open_array(
    TileDB_CTX* tiledb_ctx,
    const char* array_name,
    const char* mode) {
  return tiledb_ctx->storage_manager_->open_array(array_name, mode);
}
