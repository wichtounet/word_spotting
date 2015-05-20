default: release

.PHONY: default release debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

CXX_FLAGS += -pedantic

CXX_FLAGS += -Idll/etl/include -Idll/etl/lib/include -Idll/nice_svm/include -Idll/include
LD_FLAGS  += -pthread -lopencv_core -lopencv_imgproc -lopencv_highgui -lsvm

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -stdlib=libc++
else
ifneq (,$(findstring g++,$(CXX)))
ifneq (,$(GCC_LD_LIBRARY_PATH))
LD_FLAGS += -L$(GCC_LD_LIBRARY_PATH)
endif
endif
endif

# Vectorization
CXX_FLAGS += -DETL_VECTORIZE_FULL

# Activate BLAS mode on demand
ifneq (,$(SPOTTER_MKL))
CXX_FLAGS += -DETL_MKL_MODE $(shell pkg-config --cflags cblas)
LD_FLAGS += $(shell pkg-config --libs cblas)
CXX_FLAGS += -Wno-tautological-compare
else
ifneq (,$(SPOTTER_BLAS))
CXX_FLAGS += -DETL_BLAS_MODE $(shell pkg-config --cflags cblas)
LD_FLAGS += $(shell pkg-config --libs cblas)
endif
endif

$(eval $(call auto_folder_compile,src))
$(eval $(call auto_add_executable,spotter))

release: release_spotter
release_debug: release_debug_spotter
debug: debug_spotter

all: release release_debug debug

cppcheck:
	cppcheck --enable=all --std=c++11 -I include src

clean: base_clean

include make-utils/cpp-utils-finalize.mk
