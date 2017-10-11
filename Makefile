default: release

.PHONY: default release debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

CXX_FLAGS += -pedantic

CXX_FLAGS += -Idll/etl/include -Idll/etl/lib/include -Idll/nice_svm/include -Idll/include
LD_FLAGS  += -pthread -lopencv_core -lopencv_imgproc -lopencv_highgui -lsvm

RELEASE_DEBUG_CXX_FLAGS=-DNAN_DEBUG

ifneq (,$(findstring g++,$(CXX)))
ifneq (,$(GCC_LD_LIBRARY_PATH))
LD_FLAGS += -L$(GCC_LD_LIBRARY_PATH)
endif
endif

# Performance Flags (ETL)
CXX_FLAGS += -DETL_VECTORIZE_FULL -DETL_PARALLEL -DETL_CONV4_PREFER_BLAS

# Tune GCC warnings
ifeq (,$(findstring clang,$(CXX)))
ifneq (,$(findstring g++,$(CXX)))
CXX_FLAGS += -Wno-ignored-attributes -Wno-misleading-indentation
endif
endif

# Configure HMM
ifneq (,$(SPOTTER_NO_HMM))
CXX_FLAGS += -DSPOTTER_NO_MLPACK
else
CXX_FLAGS += -I/usr/include/armadillo_bits/
LD_FLAGS  += -lmlpack -larmadillo -lboost_serialization -lboost_program_options
endif

# Activate BLAS mode on demand
ifneq (,$(SPOTTER_MKL_THREADS))
CXX_FLAGS += -DETL_MKL_MODE $(shell pkg-config --cflags mkl-threads)
LD_FLAGS += $(shell pkg-config --libs mkl-threads)
ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-tautological-compare
endif
else
ifneq (,$(SPOTTER_MKL))
CXX_FLAGS += -DETL_MKL_MODE $(shell pkg-config --cflags mkl)
LD_FLAGS += $(shell pkg-config --libs mkl)
ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-tautological-compare
endif
else
ifneq (,$(SPOTTER_BLAS))
CXX_FLAGS += -DETL_BLAS_MODE $(shell pkg-config --cflags cblas)
LD_FLAGS += $(shell pkg-config --libs cblas)
endif
endif
endif

ifneq (,$(SPOTTER_CUDNN))
CXX_FLAGS += -DETL_CUDNN_MODE $(shell pkg-config --cflags cudnn)
LD_FLAGS += $(shell pkg-config --libs cudnn)
endif

ifneq (,$(SPOTTER_OLD))
CXX_FLAGS += -DOPENCV_23
endif

ifneq (,$(SPOTTER_MEMORY))
CXX_FLAGS += -DMEMORY_DEBUG
endif

# Uncomment the next line for better template error debugging
#CXX_FLAGS += -ftemplate-backtrace-limit=0

$(eval $(call auto_folder_compile,src))
$(eval $(call auto_add_executable,spotter))

release: release_spotter
release_debug: release_debug_spotter
debug: debug_spotter

all: release release_debug debug

cppcheck:
	cppcheck --force -I include/ --platform=unix64 --suppress=missingIncludeSystem --enable=all --std=c++11 src/*.cpp include/*.hpp

CLANG_FORMAT ?= clang-format-3.7
CLANG_MODERNIZE ?= clang-modernize-3.7
CLANG_TIDY ?= clang-tidy-3.7

format:
	git ls-files "*.hpp" "*.cpp" | xargs ${CLANG_FORMAT} -i -style=file

# Note: This way of doing this is ugly as hell and prevent parallelism, but it seems to be the only way to modernize both sources and headers
modernize:
	git ls-files "*.hpp" "*.cpp" > kws_file_list
	${CLANG_MODERNIZE} -add-override -loop-convert -pass-by-value -use-auto -use-nullptr -p ${PWD} -include-from=kws_file_list
	rm kws_file_list

# clang-tidy with some false positive checks removed
tidy:
	${CLANG_TIDY} -checks='*,-llvm-include-order,-clang-analyzer-alpha.core.PointerArithm,-clang-analyzer-alpha.deadcode.UnreachableCode,-clang-analyzer-alpha.core.IdenticalExpr' -p ${PWD} src/*.cpp -header-filter='include/*' &> tidy_report_light
	echo "The report from clang-tidy is availabe in tidy_report_light"

# clang-tidy with all the checks
tidy_all:
	${CLANG_TIDY} -checks='*' -p ${PWD} src/*.cpp -header-filter='include/*' &> tidy_report_all
	echo "The report from clang-tidy is availabe in tidy_report_all"

clean: base_clean

include make-utils/cpp-utils-finalize.mk
