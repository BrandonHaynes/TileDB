/**
 * @file   tiledb_array_aio_write_dense.cc
 *
 * @section LICENSE
 *
 * The MIT License
 * 
 * @copyright Copyright (c) 2016 MIT and Intel Corporation
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
 * It shows how to write asynchronoulsy to a dense array. The case of sparse
 * array is similar.
 */

#include "c_api.h"
#include <cstring>

// Simply prints the input string to stdout
void *print_upon_completion(void* s) {
  printf("%s\n", (char*) s);

  return NULL;
}

int main() {
  // Initialize context with the default configuration parameters
  TileDB_CTX* tiledb_ctx;
  tiledb_ctx_init(&tiledb_ctx, NULL);

  // Initialize array
  TileDB_Array* tiledb_array;
  tiledb_array_init(
      tiledb_ctx,                                // Context 
      &tiledb_array,                             // Array object
      "my_workspace/dense_arrays/my_array_A",    // Array name
      TILEDB_ARRAY_WRITE,                        // Mode
      NULL,                                      // Entire domain
      NULL,                                      // All attributes
      0);                                        // Number of attributes

  // Prepare cell buffers
  int buffer_a1[] = 
  {
      0,  1,  2,  3,                                     // Upper left tile 
      4,  5,  6,  7,                                     // Upper right tile
      8,  9,  10, 11,                                    // Lower left tile
      12, 13, 14, 15                                     // Lower right tile
  };
  size_t buffer_a2[] = 
  {
      0,  1,  3,  6,                                     // Upper left tile
      10, 11, 13, 16,                                    // Upper right tile
      20, 21, 23, 26,                                    // Lower left tile
      30, 31, 33, 36                                     // Lower right tile
  };
  char buffer_var_a2[] =
      "abbcccdddd"                                       // Upper left tile
      "effggghhhh"                                       // Upper right tile
      "ijjkkkllll"                                       // Lower left tile
      "mnnooopppp";                                      // Lower right tile
  float buffer_a3[] = 
  {
      0.1,  0.2,  1.1,  1.2,  2.1,  2.2,  3.1,  3.2,     // Upper left tile
      4.1,  4.2,  5.1,  5.2,  6.1,  6.2,  7.1,  7.2,     // Upper right tile
      8.1,  8.2,  9.1,  9.2,  10.1, 10.2, 11.1, 11.2,    // Lower left tile
      12.1, 12.2, 13.1, 13.2, 14.1, 14.2, 15.1, 15.2,    // Lower right tile
  };
  void* buffers[] = { buffer_a1, buffer_a2, buffer_var_a2, buffer_a3 };
  size_t buffer_sizes[] = 
  { 
      sizeof(buffer_a1),  
      sizeof(buffer_a2),
      sizeof(buffer_var_a2)-1,  // No need to store the last '\0' character
      sizeof(buffer_a3)
  };

  // Prepare AIO request
  TileDB_AIO_Request tiledb_aio_request;
  // ALWAYS zero out the struct before populating it
  memset(&tiledb_aio_request, 0, sizeof(struct TileDB_AIO_Request));  
  tiledb_aio_request.buffers_ = buffers;
  tiledb_aio_request.buffer_sizes_ = buffer_sizes;
  tiledb_aio_request.completion_handle_ = print_upon_completion;
  char s[100] = "AIO request completed";
  tiledb_aio_request.completion_data_ = s; 

  // Write to array
  tiledb_array_aio_write(tiledb_array, &tiledb_aio_request); 

  // Wait for AIO to complete
  printf("AIO in progress\n");
  while(tiledb_aio_request.status_ != TILEDB_AIO_COMPLETED); 

  // Finalize array
  tiledb_array_finalize(tiledb_array);

  // Finalize context
  tiledb_ctx_finalize(tiledb_ctx);

  return 0;
}
