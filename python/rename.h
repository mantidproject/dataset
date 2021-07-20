// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include "scipp/dataset/dataset.h"

template <class T> T rename_dims(T &self, const std::map<Dim, Dim> &name_dict) {
  auto out(self);
  for (const auto &[from, to] : name_dict)
    out.rename(from, to);
  return out;
}
