// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/common/index.h"
#include "scipp/common/index_composition.h"

TEST(IndexTest, size) { ASSERT_EQ(sizeof(scipp::index), 8); }
TEST(IndexTest, sign) { ASSERT_EQ(scipp::index{-1}, int64_t(-1)); }

TEST(IndexCompositionTest, flat_index_from_strides_0d) {
  const std::array<scipp::index, 0> strides{};
  const std::array<scipp::index, 0> indices{};
  EXPECT_EQ(scipp::flat_index_from_strides(strides.begin(), strides.end(),
                                           indices.begin()),
            0);
}

TEST(IndexCompositionTest, flat_index_from_strides_1d) {
  for (scipp::index stride = -5; stride < 6; ++stride) {
    const std::array<scipp::index, 1> strides{stride};
    for (scipp::index index = 0; index < 10; ++index) {
      const std::array<scipp::index, 1> indices{index};
      EXPECT_EQ(scipp::flat_index_from_strides(strides.begin(), strides.end(),
                                               indices.begin()),
                index * stride);
    }
  }
}

TEST(IndexCompositionTest, flat_index_from_strides_2d) {
  for (scipp::index stride0 = -5; stride0 < 6; ++stride0) {
    for (scipp::index stride1 = -5; stride1 < 6; ++stride1) {
      const std::array<scipp::index, 2> strides{stride0, stride1};
      for (scipp::index index0 = 0; index0 < 10; ++index0) {
        for (scipp::index index1 = 0; index1 < 10; ++index1) {
          const std::array<scipp::index, 2> indices{index0, index1};
          EXPECT_EQ(scipp::flat_index_from_strides(
                        strides.begin(), strides.end(), indices.begin()),
                    index0 * stride0 + index1 * stride1);
        }
      }
    }
  }
}

TEST(IndexCompositionTest, flat_index_from_strides_3d) {
  for (scipp::index stride0 = -5; stride0 < 6; ++stride0) {
    for (scipp::index stride1 = -5; stride1 < 6; ++stride1) {
      for (scipp::index stride2 = -5; stride2 < 6; ++stride2) {
        const std::array<scipp::index, 3> strides{stride0, stride1, stride2};
        for (scipp::index index0 = 0; index0 < 10; ++index0) {
          for (scipp::index index1 = 0; index1 < 10; ++index1) {
            for (scipp::index index2 = 0; index2 < 10; ++index2) {
              const std::array<scipp::index, 3> indices{index0, index1, index2};
              EXPECT_EQ(scipp::flat_index_from_strides(
                            strides.begin(), strides.end(), indices.begin()),
                        index0 * stride0 + index1 * stride1 + index2 * stride2);
            }
          }
        }
      }
    }
  }
}

TEST(IndexCompositionTest, extract_indices_0d) {
  const std::array<scipp::index, 3> shape{-1, -2, -3};
  std::array<scipp::index, 3> indices{-1, -2, -3};
  scipp::extract_indices(0, shape.begin(), shape.begin() + 0, indices.begin());
  EXPECT_EQ(indices[0], 0);
  scipp::extract_indices(1, shape.begin(), shape.begin() + 0, indices.begin());
  EXPECT_EQ(indices[0], 1);
}

TEST(IndexCompositionTest, extract_indices_1d) {
  for (scipp::index size : {0, 1, 2, 5}) {
    for (scipp::index total_index = 0; total_index < size; ++total_index) {
      const std::array<scipp::index, 3> shape{size, -2, -3};
      std::array<scipp::index, 3> indices{-1, -2, -3};
      scipp::extract_indices(total_index, shape.begin(), shape.begin() + 1,
                             indices.begin());
      EXPECT_EQ(indices[0], total_index);
    }
  }
}

TEST(IndexCompositionTest, extract_indices_1d_size0) {
  const std::array<scipp::index, 3> shape{0, -2, -3};
  std::array<scipp::index, 3> indices{-1, -2, -3};
  scipp::extract_indices(0, shape.begin(), shape.begin() + 1, indices.begin());
  EXPECT_EQ(indices[0], 0);
  scipp::extract_indices(1, shape.begin(), shape.begin() + 1, indices.begin());
  EXPECT_EQ(indices[0], 1);
}

TEST(IndexCompositionTest, extract_indices_2d) {
  const std::array<scipp::index, 3> shape{2, 3, -1};
  for (scipp::index i = 0; i < shape[0] * shape[1]; ++i) {
    std::array<scipp::index, 3> indices{-1, -2, -3};
    scipp::extract_indices(i, shape.begin(), shape.begin() + 2,
                           indices.begin());
    EXPECT_EQ(indices[0] + shape[0] * indices[1], i);
  }
}

TEST(IndexCompositionTest, extract_indices_2d_end) {
  const std::array<scipp::index, 3> shape{2, 3, -1};
  std::array<scipp::index, 3> indices{-1, -2, -3};
  const std::array<scipp::index, 3> expected{0, 3, -3};
  scipp::extract_indices(2 * 3, shape.begin(), shape.begin() + 2,
                         indices.begin());
  EXPECT_EQ(indices, expected);
}

TEST(IndexCompositionTest, extract_indices_2d_size0) {
  std::array<scipp::index, 3> shape{0, 1, -3};
  std::array<scipp::index, 3> indices{-1, -2, -3};
  scipp::extract_indices(0, shape.begin(), shape.begin() + 2, indices.begin());
  EXPECT_EQ(indices[0], 0);
  EXPECT_EQ(indices[1], 0);
  scipp::extract_indices(1, shape.begin(), shape.begin() + 2, indices.begin());
  EXPECT_EQ(indices[0], 0);
  EXPECT_EQ(indices[1], 1);

  shape = {2, 0, -3};
  scipp::extract_indices(0, shape.begin(), shape.begin() + 2, indices.begin());
  EXPECT_EQ(indices[0], 0);
  EXPECT_EQ(indices[1], 0);
  scipp::extract_indices(1, shape.begin(), shape.begin() + 2, indices.begin());
  EXPECT_EQ(indices[0], 1);
  EXPECT_EQ(indices[1], 0);

  shape = {0, 0, -3};
  scipp::extract_indices(0, shape.begin(), shape.begin() + 2, indices.begin());
  EXPECT_EQ(indices[0], 0);
  EXPECT_EQ(indices[1], 0);
  scipp::extract_indices(1, shape.begin(), shape.begin() + 2, indices.begin());
  EXPECT_EQ(indices[0], 0);
  EXPECT_EQ(indices[1], 1);
}

TEST(IndexCompositionTest, extract_indices_3d) {
  const std::array<scipp::index, 3> shape{4, 5, 2};
  for (scipp::index i = 0; i < shape[0] * shape[1] * shape[2]; ++i) {
    std::array<scipp::index, 3> indices{-1, -2, -3};
    scipp::extract_indices(i, shape.begin(), shape.begin() + 3,
                           indices.begin());
    EXPECT_EQ(indices[0] + shape[0] * (indices[1] + shape[1] * indices[2]), i);
  }
}

TEST(IndexCompositionTest, extract_indices_3d_end) {
  const std::array<scipp::index, 3> shape{2, 3, 5};
  std::array<scipp::index, 3> indices{-1, -2, -3};
  const std::array<scipp::index, 3> expected{0, 0, 5};
  scipp::extract_indices(2 * 3 * 5, shape.begin(), shape.begin() + 3,
                         indices.begin());
  EXPECT_EQ(indices, expected);
}
