#pragma once

#include <array>
#include <iostream>
#include <sstream>
#include <vector>

// Use namespace std to make the operators discoverable through ADL.
namespace std { // NOLINT
template <typename T, size_t N>
std::ostream &operator<<(std::ostream &os, const std::array<T, N> &a) {
  os << '[';
  for (size_t i = 0; i + 1 < std::size(a); ++i) {
    os << a[i] << ", ";
  }
  if (std::size(a) > 0) {
    os << a.back();
  }
  os << ']';
  return os;
}

template <typename T, size_t N>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  os << '[';
  for (size_t i = 0; i + 1 < std::size(v); ++i) {
    os << v[i] << ", ";
  }
  if (std::size(v) > 0) {
    os << v.back();
  }
  os << ']';
  return os;
}
} // namespace std