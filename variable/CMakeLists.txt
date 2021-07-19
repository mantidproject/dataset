# ~~~
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# ~~~
set(TARGET_NAME "scipp-variable")
set(INC_FILES
    ${variable_INC_FILES}
    include/scipp/variable/astype.h
    include/scipp/variable/arithmetic.h
    include/scipp/variable/bins.h
    include/scipp/variable/bin_util.h
    include/scipp/variable/comparison.h
    include/scipp/variable/except.h
    include/scipp/variable/logical.h
    include/scipp/variable/math.h
    include/scipp/variable/misc_operations.h
    include/scipp/variable/operations.h
    include/scipp/variable/rebin.h
    include/scipp/variable/reduction.h
    include/scipp/variable/shape.h
    include/scipp/variable/slice.h
    include/scipp/variable/sort.h
    include/scipp/variable/special_values.h
    include/scipp/variable/string.h
    include/scipp/variable/structures.h
    include/scipp/variable/subspan_view.h
    include/scipp/variable/transform.h
    include/scipp/variable/transform_subspan.h
    include/scipp/variable/trigonometry.h
    include/scipp/variable/util.h
    include/scipp/variable/variable_concept.h
    include/scipp/variable/variable_factory.h
    include/scipp/variable/variable.h
    include/scipp/variable/variable_keyword_arg_constructor.h
    include/scipp/variable/variable.tcc
    include/scipp/variable/visit.h
)

set(SRC_FILES
    ${variable_SRC_FILES}
    astype.cpp
    bins.cpp
    bin_array_variable.cpp
    bin_detail.cpp
    bin_util.cpp
    comparison.cpp
    convolution.cpp
    creation.cpp
    cumulative.cpp
    except.cpp
    operations.cpp
    rebin.cpp
    reduction.cpp
    shape.cpp
    slice.cpp
    sort.cpp
    special_values.cpp
    string.cpp
    structures.cpp
    subspan_view.cpp
    to_unit.cpp
    util.cpp
    variable_concept.cpp
    variable_factory.cpp
    variable.cpp
    variable_instantiate_basic.cpp
    variable_instantiate_bin_elements.cpp
    variable_instantiate_linalg.cpp
    variable_instantiate_map_elements.cpp
    variable_instantiate_view_elements.cpp
)

set(LINK_TYPE "STATIC")
if(DYNAMIC_LIB)
  set(LINK_TYPE "SHARED")
endif(DYNAMIC_LIB)

add_library(${TARGET_NAME} ${LINK_TYPE} ${INC_FILES} ${SRC_FILES})
generate_export_header(${TARGET_NAME})
# Macros such as INSTANTIATE_VARIABLE are used for several targets (in
# particular also from dataset/ and python/). Therefore, it cannot use
# SCIPP_VARIABLE_EXPORT but has to point to the export macro of the current
# target. It therefore uses SCIPP_EXPORT, which can be pointed to the current
# applicable export macro, as in the following:
target_compile_definitions(
  ${TARGET_NAME} PRIVATE SCIPP_EXPORT=SCIPP_VARIABLE_EXPORT
)
target_link_libraries(${TARGET_NAME} PUBLIC scipp-core)
if(TBB_FOUND AND NOT DISABLE_MULTI_THREADING)
  target_link_libraries(${TARGET_NAME} PUBLIC TBB::tbb)
endif()

target_include_directories(
  ${TARGET_NAME}
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME variable)

if(PRECOMPILED_HEADERS)
  target_precompile_headers(
    ${TARGET_NAME} PUBLIC include/scipp/variable/transform.h
    include/scipp/variable/variable.h
  )
endif()

add_subdirectory(test)

scipp_install_component(TARGET ${TARGET_NAME} INSTALL_GENERATED)
