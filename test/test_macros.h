// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include <algorithm>
#include <initializer_list>

#define EXPECT_THROW_MSG(TRY_BLOCK, EXCEPTION_TYPE, MESSAGE)                   \
  EXPECT_THROW(                                                                \
      {                                                                        \
        try {                                                                  \
          TRY_BLOCK;                                                           \
        } catch (const EXCEPTION_TYPE &e) {                                    \
          EXPECT_STREQ(MESSAGE, e.what());                                     \
          throw;                                                               \
        }                                                                      \
      },                                                                       \
      EXCEPTION_TYPE);

#define EXPECT_THROW_MSG_SUBSTR(TRY_BLOCK, EXCEPTION_TYPE, SUBSTR)             \
  EXPECT_THROW(                                                                \
      {                                                                        \
        try {                                                                  \
          TRY_BLOCK;                                                           \
        } catch (const EXCEPTION_TYPE &e) {                                    \
          EXPECT_TRUE(std::string(e.what()).find(SUBSTR) !=                    \
                      std::string::npos);                                      \
          throw;                                                               \
        }                                                                      \
      },                                                                       \
      EXCEPTION_TYPE);

template <class T1, class T2>
bool equals(const T1 &a, const std::initializer_list<T2> &b) {
  return std::equal(a.begin(), a.end(), b.begin(), b.end());
}
