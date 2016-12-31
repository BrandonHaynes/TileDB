# **************** # 
#     Macros       #
# **************** # 

OS := $(shell uname)
ifneq ($(shell gcc -v 2>&1 | grep -c "clang"),0)
  COMPILER := clang
else
  COMPILER := gcc
endif

# --- Configuration flags --- #
CPPFLAGS = -std=gnu++11 -fPIC -fvisibility=hidden \
      -D_FILE_OFFSET_BITS=64 

# For the Travis integration
ifdef TRAVIS
  CPPFLAGS += --coverage
endif

# --- Support for OpenMP --- #
OPENMP_FLAG =
ifeq ($(COMPILER), gcc)
  CPPFLAGS += -DOPENMP
  OPENMP_FLAG = -fopenmp
endif

# --- Debug/Release mode handler --- #
BUILD =
ifeq ($(BUILD),)
  BUILD = release
endif
 
ifeq ($(BUILD),release)
  CPPFLAGS += -DNDEBUG -O3 
endif

ifeq ($(BUILD),debug)
  CPPFLAGS += -DDEBUG -gdwarf-3 -g3 -Wall 
endif

# --- Verbose mode handler --- #
VERBOSE =
ifeq ($(VERBOSE),1)
  CPPFLAGS += -DNVERBOSE
endif

# --- Compilers --- #
CXX = g++   

# --- Directories --- #
CORE_INCLUDE_DIR = core/include
CORE_INCLUDE_SUBDIRS = $(wildcard core/include/*)
CORE_SRC_DIR = core/src
CORE_SRC_SUBDIRS = $(wildcard core/src/*)
CORE_OBJ_DEB_DIR = core/obj/debug
CORE_BIN_DEB_DIR = core/bin/debug
PYTHON_SRC_DIR = python/src
PYTHON_SRC_SUBDIRS = $(wildcard python/src/*)
PYTHON_INCLUDE_SUBDIRS = $(wildcard python/include/*)
PYTHON_OBJ_DEB_DIR = python/obj/debug
ifeq ($(BUILD),debug)
  CORE_OBJ_DIR = $(CORE_OBJ_DEB_DIR)
  CORE_BIN_DIR = $(CORE_BIN_DEB_DIR)
  PYTHON_OBJ_DIR = $(PYTHON_OBJ_DEB_DIR)
endif
CORE_OBJ_REL_DIR = core/obj/release
CORE_BIN_REL_DIR = core/bin/release
PYTHON_OBJ_REL_DIR = python/obj/release
ifeq ($(BUILD),release)
  CORE_OBJ_DIR = $(CORE_OBJ_REL_DIR)
  CORE_BIN_DIR = $(CORE_BIN_REL_DIR)
  PYTHON_OBJ_DIR = $(PYTHON_OBJ_REL_DIR)
endif
CORE_LIB_DEB_DIR = core/lib/debug
PYTHON_LIB_DEB_DIR = python/lib/debug
ifeq ($(BUILD),debug)
  CORE_LIB_DIR = $(CORE_LIB_DEB_DIR)
  PYTHON_LIB_DIR = $(PYTHON_LIB_DEB_DIR)
endif
CORE_LIB_REL_DIR = core/lib/release
PYTHON_LIB_REL_DIR = python/lib/release
ifeq ($(BUILD),release)
  CORE_LIB_DIR = $(CORE_LIB_REL_DIR)
  PYTHON_LIB_DIR = $(PYTHON_LIB_REL_DIR)
endif
EXAMPLES_INCLUDE_DIR = examples/include
EXAMPLES_SRC_DIR = examples/src
EXAMPLES_OBJ_DEB_DIR = examples/obj/debug
EXAMPLES_BIN_DEB_DIR = examples/bin/debug
ifeq ($(BUILD),debug)
  EXAMPLES_OBJ_DIR = $(EXAMPLES_OBJ_DEB_DIR)
  EXAMPLES_BIN_DIR = $(EXAMPLES_BIN_DEB_DIR)
endif
EXAMPLES_OBJ_REL_DIR = examples/obj/release
EXAMPLES_BIN_REL_DIR = examples/bin/release
ifeq ($(BUILD),release)
  EXAMPLES_OBJ_DIR = $(EXAMPLES_OBJ_REL_DIR)
  EXAMPLES_BIN_DIR = $(EXAMPLES_BIN_REL_DIR)
endif
TEST_SRC_SUBDIRS = $(wildcard test/src/*)
TEST_SRC_DIR = test/src
TEST_OBJ_DIR = test/obj
TEST_BIN_DIR = test/bin
DOXYGEN_DIR = doxygen
DOXYGEN_MAINPAGE = $(DOXYGEN_DIR)/mainpage.dox

# --- Paths --- #
INCLUDE_PATHS = 
CORE_INCLUDE_PATHS = $(addprefix -I, $(CORE_INCLUDE_SUBDIRS))
PYTHON_INCLUDE_PATHS = $(addprefix -I, $(PYTHON_INCLUDE_SUBDIRS))
EXAMPLES_INCLUDE_PATHS = -I$(EXAMPLES_INCLUDE_DIR)
TEST_INCLUDE_PATHS = $(addprefix -I, $(CORE_INCLUDE_SUBDIRS))
LIBRARY_PATHS =

ifdef TRAVIS
  LIBRARY_PATHS += --coverage
endif

# --- Libraries --- #
ZLIB = -lz
OPENSSLLIB = -lcrypto
GTESTLIB = -lgtest -lgtest_main
MPILIB = -lmpi
PYTHONLIB = -L$(CORE_LIB_DIR) -ltiledb -lpython2.7 -lboost_python

# --- For the TileDB dynamic library --- #
ifeq ($(OS), Darwin)
  SHLIB_EXT = dylib
else
  SHLIB_EXT = so
endif

# --- Files --- #
CORE_INCLUDE := $(foreach D,$(CORE_INCLUDE_SUBDIRS),$D/*.h) 
CORE_SRC := $(wildcard $(foreach D,$(CORE_SRC_SUBDIRS),$D/*.cc))
CORE_OBJ := $(patsubst $(CORE_SRC_DIR)/%.cc, $(CORE_OBJ_DIR)/%.o, $(CORE_SRC))
PYTHON_SRC := $(wildcard $(foreach D,$(PYTHON_SRC_SUBDIRS),$D/*.cc))
PYTHON_OBJ := $(patsubst $(PYTHON_SRC_DIR)/%.cc, $(PYTHON_OBJ_DIR)/%.o, $(PYTHON_SRC))
#PYTHON_LIB := $(patsubst $(PYTHON_SRC_DIR)/%.cc, $(PYTHON_LIB_DIR)/%.$(SHLIB_EXT), $(PYTHON_SRC))
PYTHON_LIB := $(PYTHON_LIB_DIR)/tiledb.$(SHLIB_EXT)
EXAMPLES_INCLUDE := $(wildcard $(EXAMPLES_INCLUDE_DIR)/*.h)
EXAMPLES_SRC := $(wildcard $(EXAMPLES_SRC_DIR)/*.cc)
EXAMPLES_OBJ := $(patsubst $(EXAMPLES_SRC_DIR)/%.cc,\
                             $(EXAMPLES_OBJ_DIR)/%.o, $(EXAMPLES_SRC))
EXAMPLES_BIN := $(patsubst $(EXAMPLES_SRC_DIR)/%.cc,\
                             $(EXAMPLES_BIN_DIR)/%, $(EXAMPLES_SRC)) 
TEST_SRC := $(wildcard $(foreach D,$(TEST_SRC_SUBDIRS),$D/*.cc))
TEST_OBJ := $(patsubst $(TEST_SRC_DIR)/%.cc, $(TEST_OBJ_DIR)/%.o, $(TEST_SRC))

# **************** # 
# General Targets  #
# **************** # 

.PHONY: core python examples test doc clean_core \
        clean_test clean_doc clean_examples clean

all: core libtiledb 

core: $(CORE_OBJ) 

python: $(PYTHON_LIB)
#python: $(PYTHON_OBJ) $(PYTHON_LIB)

libtiledb: core $(CORE_LIB_DIR)/libtiledb.$(SHLIB_EXT) \
                $(CORE_LIB_DIR)/libtiledb.a

examples: libtiledb $(EXAMPLES_OBJ) $(EXAMPLES_BIN)

doc: doxyfile.inc 

test: libtiledb $(TEST_BIN_DIR)/tiledb_test
	@echo "Running TileDB tests"
	@$(TEST_BIN_DIR)/tiledb_test

clean: clean_core clean_libtiledb \
       clean_test clean_doc clean_examples 

# **************** # 
#       Core       #
# **************** # 

# --- Compilation and dependency genration --- #

-include $(CORE_OBJ:%.o=%.d)

$(CORE_OBJ_DIR)/%.o: $(CORE_SRC_DIR)/%.cc
	@mkdir -p $(dir $@) 
	@echo "Compiling $<"
	@$(CXX) $(CPPFLAGS) $(INCLUDE_PATHS) $(CORE_INCLUDE_PATHS) -c $< -o $@ 
	@$(CXX) -MM $(CORE_INCLUDE_PATHS) $(INCLUDE_PATHS) $< > $(@:.o=.d)
	@mv -f $(@:.o=.d) $(@:.o=.d.tmp)
	@sed 's|.*:|$@:|' < $(@:.o=.d.tmp) > $(@:.o=.d)
	@rm -f $(@:.o=.d.tmp)

# --- Cleaning --- #

clean_core: 
	@echo 'Cleaning core'
	@rm -rf $(CORE_OBJ_DEB_DIR)/* $(CORE_OBJ_REL_DIR)/* \
                $(CORE_BIN_DEB_DIR)/* $(CORE_BIN_REL_DIR)/*

# **************** # 
#     libtiledb    #
# **************** # 

-include $(CORE_OBJ:%.o=%.d)

# --- Linking --- #

ifeq ($(0S), Darwin)
  SHLIB_FLAGS = -dynamiclib
else
  SHLIB_FLAGS = -shared
endif

ifeq ($(SHLIB_EXT), so)
  SONAME = -Wl,-soname=libtiledb.so
else
  SONAME =
endif

$(CORE_LIB_DIR)/libtiledb.$(SHLIB_EXT): $(CORE_OBJ)
	@mkdir -p $(CORE_LIB_DIR)
	@echo "Creating dynamic library libtiledb.$(SHLIB_EXT)"
	@$(CXX) $(SHLIB_FLAGS) $(SONAME) -o $@ $^ $(LIBRARY_PATHS) $(MPILIB) \
		$(ZLIB) $(OPENSSLLIB) $(OPENMP_FLAG)

$(CORE_LIB_DIR)/libtiledb.a: $(CORE_OBJ)
	@mkdir -p $(CORE_LIB_DIR)
	@echo "Creating static library libtiledb.a"
	@ar rcs $(CORE_LIB_DIR)/libtiledb.a $^

# --- Cleaning --- #

clean_libtiledb:
	@echo "Cleaning libtiledb"
	@rm -rf $(CORE_LIB_DEB_DIR)/* $(CORE_LIB_REL_DIR)/*

# **************** # 
#      Python      #
# **************** # 

-include $(PYTHON_OBJ:%.o=%.d)

$(PYTHON_OBJ_DIR)/%.o: $(PYTHON_SRC_DIR)/%.cc
	@mkdir -p $(dir $@) 
	@echo "Compiling $<"
	@$(CXX) $(CPPFLAGS) $(INCLUDE_PATHS) $(PYTHON_INCLUDE_PATHS) $(CORE_INCLUDE_PATHS) -c $< -o $@ 
	@$(CXX) -MM $(PYTHON_INCLUDE_PATHS) $(CORE_INCLUDE_PATHS) $(INCLUDE_PATHS) $< > $(@:.o=.d)
	@mv -f $(@:.o=.d) $(@:.o=.d.tmp)
	@sed 's|.*:|$@:|' < $(@:.o=.d.tmp) > $(@:.o=.d)
	@rm -f $(@:.o=.d.tmp)

$(PYTHON_LIB): $(PYTHON_OBJ)
	@mkdir -p $(dir $@)
	@echo "Creating dynamic library tiledb.$(SHLIB_EXT) $^"
	@$(CXX) $(SHLIB_FLAGS) -o $@ $^ $(LIBRARY_PATHS) $(PYTHONLIB) $(MPILIB) $(ZLIB) $(OPENSSLLIB) $(OPENMP_FLAG)
	@touch $(dir $@)/__init__.py

# --- Cleaning --- #

clean_python: 
	@echo 'Cleaning Python'
	@rm -rf $(PYTHON_OBJ_DEB_DIR)/* $(PYTHON_OBJ_REL_DIR)/* \
            $(PYTHON_LIB_DEB_DIR)/* $(PYTHON_LIB_REL_DIR)/*

# **************** # 
#     Examples     #
# **************** # 

# --- Compilation and dependency genration --- #

-include $(EXAMPLES_OBJ:.o=.d)

$(EXAMPLES_OBJ_DIR)/%.o: $(EXAMPLES_SRC_DIR)/%.cc
	@mkdir -p $(EXAMPLES_OBJ_DIR)
	@echo "Compiling $<"
	@$(CXX) $(CPPFLAGS) $(OPENMP_FLAG) $(INCLUDE_PATHS) \
                $(EXAMPLES_INCLUDE_PATHS) \
		$(CORE_INCLUDE_PATHS) -c $< -o $@
	@$(CXX) -MM $(EXAMPLES_INCLUDE_PATHS) \
                    $(CORE_INCLUDE_PATHS) $(INCLUDE_PATHS) $< > $(@:.o=.d)
	@mv -f $(@:.o=.d) $(@:.o=.d.tmp)
	@sed 's|.*:|$@:|' < $(@:.o=.d.tmp) > $(@:.o=.d)
	@rm -f $(@:.o=.d.tmp)

# --- Linking --- #

$(EXAMPLES_BIN_DIR)/%: $(EXAMPLES_OBJ_DIR)/%.o $(CORE_LIB_DIR)/libtiledb.a
	@mkdir -p $(EXAMPLES_BIN_DIR)
	@echo "Creating $@"
	@$(CXX) -std=gnu++11 -o $@ $^ $(LIBRARY_PATHS) $(MPILIB) $(ZLIB) \
		 $(OPENSSLLIB) $(OPENMP_FLAG) 

# --- Cleaning --- #

clean_examples:
	@echo 'Cleaning examples'
	@rm -f $(EXAMPLES_OBJ_DEB_DIR)/* $(EXAMPLES_OBJ_REL_DIR)/* \
               $(EXAMPLES_BIN_DEB_DIR)/* $(EXAMPLES_BIN_REL_DIR)/*

# **************** # 
#       Test       #
# **************** # 

# --- Compilation and dependency genration --- #

-include $(TEST_OBJ:.o=.d)

$(TEST_OBJ_DIR)/%.o: $(TEST_SRC_DIR)/%.cc
	@mkdir -p $(dir $@) 
	@echo "Compiling $<"
	@$(CXX) $(CPPFLAGS) $(OPENMP_FLAG) $(TEST_INCLUDE_PATHS) \
		$(INCLUDE_PATHS) -c $< -o $@
	@$(CXX) -MM $(TEST_INCLUDE_PATHS) \
                $(CORE_INCLUDE_PATHS) $(INCLUDE_PATHS) $< > $(@:.o=.d)
	@mv -f $(@:.o=.d) $(@:.o=.d.tmp)
	@sed 's|.*:|$@:|' < $(@:.o=.d.tmp) > $(@:.o=.d)
	@rm -f $(@:.o=.d.tmp)

# --- Linking --- #

$(TEST_BIN_DIR)/tiledb_test: $(TEST_OBJ) $(CORE_LIB_DIR)/libtiledb.a
	@mkdir -p $(TEST_BIN_DIR)
	@echo "Creating test_cmd"
	@$(CXX) -std=gnu++11 -o $@ $^ $(LIBRARY_PATHS) $(MPILIB) $(ZLIB) \
		$(OPENSSLLIB) $(GTESTLIB) $(OPENMP_FLAG) 

# --- Cleaning --- #

clean_test:
	@echo "Cleaning test"
	@rm -rf $(TEST_OBJ_DIR) $(TEST_BIN_DIR)
	

# **************** # 
#   Documentation  #
# **************** # 

doxyfile.inc: $(CORE_INCLUDE) $(DOXYGEN_MAINPAGE)
	@echo 'Creating Doxygen documentation'
	@echo INPUT = $(DOXYGEN_DIR)/mainpage.dox $(CORE_INCLUDE) > doxyfile.inc
	@echo FILE_PATTERNS = *.h >> doxyfile.inc
	@doxygen Doxyfile.mk > Doxyfile.log 2>&1

# --- Cleaning --- #

clean_doc:
	@echo "Cleaning documentation"
	@rm -f doxyfile.inc

