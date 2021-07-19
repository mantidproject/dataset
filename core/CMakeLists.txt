# ~~~
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# ~~~
set(TARGET_NAME "scipp-core")
set(INC_FILES
    include/scipp/core/aligned_allocator.h
    include/scipp/core/dimensions.h
    include/scipp/core/dtype.h
    include/scipp/core/element_array.h
    include/scipp/core/element_array_view.h
    include/scipp/core/histogram.h
    include/scipp/core/memory_pool.h
    include/scipp/core/multi_index.h
    include/scipp/core/parallel-fallback.h
    include/scipp/core/parallel-tbb.h
    include/scipp/core/slice.h
    include/scipp/core/tag_util.h
    include/scipp/core/transform_common.h
    include/scipp/core/value_and_variance.h
    include/scipp/core/values_and_variances.h
    include/scipp/core/view_index.h
    include/scipp/core/element/arg_list.h
    include/scipp/core/element/arithmetic.h
    include/scipp/core/element/comparison.h
    include/scipp/core/element/event_operations.h
    include/scipp/core/element/geometric_operations.h
    include/scipp/core/element/histogram.h
    include/scipp/core/element/logical.h
    include/scipp/core/element/math.h
    include/scipp/core/element/rebin.h
    include/scipp/core/element/sort.h
    include/scipp/core/element/special_values.h
    include/scipp/core/element/trigonometry.h
    include/scipp/core/element/util.h
)

set(SRC_FILES
    dimensions.cpp
    dtype.cpp
    element_array_view.cpp
    except.cpp
    multi_index.cpp
    sizes.cpp
    slice.cpp
    strides.cpp
    string.cpp
    subbin_sizes.cpp
    view_index.cpp
)

set(LINK_TYPE "STATIC")
if(DYNAMIC_LIB)
  set(LINK_TYPE "SHARED")
endif(DYNAMIC_LIB)

add_library(${TARGET_NAME} ${LINK_TYPE} ${INC_FILES} ${SRC_FILES})
generate_export_header(${TARGET_NAME})
target_link_libraries(
  ${TARGET_NAME} PUBLIC scipp-common scipp-units Boost::Boost Eigen3::Eigen
)
if(TBB_FOUND AND NOT DISABLE_MULTI_THREADING)
  target_link_libraries(${TARGET_NAME} PUBLIC TBB::tbb)
endif()

# Include tcb/span as system header to avoid compiler warnings.
target_include_directories(
  ${TARGET_NAME}
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME core)

if(PRECOMPILED_HEADERS)
  target_precompile_headers(
    ${TARGET_NAME} PUBLIC include/scipp/core/element_array.h
    include/scipp/core/element_array_view.h include/scipp/core/multi_index.h
    ${CMAKE_CURRENT_BINARY_DIR}/include/scipp/core/parallel.h
  )
endif()

add_subdirectory(test)

scipp_install_component(TARGET ${TARGET_NAME} INSTALL_GENERATED)
