/**
 * @file
 */
#pragma once

#include "numbirch/array/external.hpp"

namespace numbirch {
template<class T, int D> class Array;

/**
 * Shape and layout of Array.
 * 
 * @ingroup array
 *
 * @tparam D Number of dimensions.
 */
template<int D>
class ArrayShape {
  static_assert(0 <= D && D <= 2, "Array supports up to two dimensions");
};

/**
 * Shape and layout of a scalar (zero-dimensional array).
 * 
 * @ingroup array
 */
template<>
class ArrayShape<0> {
public:
  static constexpr int64_t size() {
    return 1;
  }

  static constexpr int64_t volume() {
    return 1;
  }

  static constexpr int rows() {
    return 1;
  }

  static constexpr int columns() {
    return 1;
  }

  static constexpr int width() {
    return 1;
  }

  static constexpr int height() {
    return 1;
  }

  static constexpr int stride() {
    return 0;
  }

  static constexpr bool conforms(const ArrayShape<0>& o) {
    return true;
  }

  template<int D>
  static constexpr bool conforms(const ArrayShape<D>& o) {
    return false;
  }

  static constexpr ArrayShape<0> compact() {
    return ArrayShape<0>();
  }

  static constexpr int64_t offset(const int64_t s) {
    return s;
  }

  static constexpr int64_t transpose(const int64_t t) {
    return t;
  }
};

/**
 * Make a scalar shape.
 * 
 * @ingroup array
 */
inline ArrayShape<0> make_shape() {
  return ArrayShape<0>();
}

/**
 * Shape and layout of a vector (one-dimensional array).
 * 
 * @ingroup array
 */
template<>
class ArrayShape<1> {
public:
  explicit ArrayShape(const int n = 0, const int inc = 1) :
      n(n),
      inc(inc) {
    //
  }

  int64_t size() const {
    return n;
  }

  int64_t volume() const {
    return int64_t(inc)*n;
  }

  int rows() const {
    return n;
  }

  static constexpr int columns() {
    return 1;
  }

  static constexpr int width() {
    return 1;
  }

  int height() const {
    return n;
  }

  int stride() const {
    return inc;
  }

  bool conforms(const ArrayShape<1>& o) const {
    return n == o.n;
  }

  template<int D>
  static constexpr bool conforms(const ArrayShape<D>& o) {
    return false;
  }

  ArrayShape<1> compact() const {
    return ArrayShape<1>(n);
  }

  template<class T>
  auto slice(const T* buffer, const std::pair<int,int>& range) const {
    assert(1 <= range.first && range.first <= n && "start of range out of bounds");
    assert(range.second <= n && "end of range out of bounds");
    int i = range.first;
    int len = std::max(0, range.second - range.first + 1);
    auto shp = ArrayShape<1>(len, inc);
    auto buf = shp.volume() ? buffer + (i - 1)*int64_t(inc) : nullptr;
    return Array(buf, shp);
  }

  template<class T>
  auto slice(const T* buffer, const int i) const {
    assert(1 <= i && i <= n && "index out of bounds");
    auto buf = buffer + (i - 1)*int64_t(inc);
    auto shp = ArrayShape<0>();
    return Array(buf, shp);
  }

  template<class T>
  auto& dice(T* buffer, const int i) {
    assert(1 <= i && i <= n && "index out of bounds");
    return buffer[(i - 1)*int64_t(inc)];
  }

  template<class T>
  const auto& dice(T* buffer, const int i) const {
    assert(1 <= i && i <= n && "index out of bounds");
    return buffer[(i - 1)*int64_t(inc)];
  }

  /**
   * Compute the 0-based offset to the @p s th element in serial storage
   * order.
   *
   * @param s Serial index, 0-based.
   */
  int64_t offset(const int64_t s) const {
    return s*inc;
  }

  /**
   * Compute the 0-based offset to the @p t th element in transpose storage
   * order.
   *
   * @param t Transpose index, 0-based.
   */
  int64_t transpose(const int64_t t) const {
    return t*inc;
  }

private:
  /**
   * Number of elements.
   */
  int n;

  /**
   * Stride between elements.
   */
  int inc;
};

/**
 * Make a vector shape.
 * 
 * @ingroup array
 * 
 * @param n Length.
 * 
 * @return Shape.
 */
inline ArrayShape<1> make_shape(const int n) {
  return ArrayShape<1>(n);
}

/**
 * Shape and layout of a matrix (two-dimensional array) in column-major order.
 * 
 * @ingroup array
 */
template<>
class ArrayShape<2> {
public:
  ArrayShape() :
      m(0),
      n(0),
      ld(0) {
    //
  }

  explicit ArrayShape(const int m, const int n) :
      m(m),
      n(n),
      ld(m) {
    //
  }

  explicit ArrayShape(const int m, const int n, const int ld) :
      m(m),
      n(n),
      ld(ld) {
    //
  }

  int64_t size() const {
    return int64_t(m)*n;
  }

  int64_t volume() const {
    return int64_t(ld)*n;
  }

  int rows() const {
    return m;
  }

  int columns() const {
    return n;
  }

  int width() const {
    return m;
  }

  int height() const {
    return n;
  }

  int stride() const {
    return ld;
  }

  bool conforms(const ArrayShape<2>& o) const {
    return m == o.m && n == o.n;
  }

  template<int D>
  static constexpr bool conforms(const ArrayShape<D>& o) {
    return false;
  }

  ArrayShape<2> compact() const {
    return ArrayShape<2>(m, n);
  }

  /**
   * Compute the 0-based offset to the @p s th element in serial storage
   * order.
   *
   * @param s Serial index, 0-based.
   */
  int64_t offset(const int64_t s) const {
    int64_t j = s/m;
    int64_t i = s - j*m;
    return i + ld*j;
  }

  /**
   * Compute the 0-based offset to the @p t th element in transpose storage
   * order.
   *
   * @param t Transpose index, 0-based.
   */
  int64_t transpose(const int64_t t) const {
    int64_t i = t/n;
    int64_t j = t - i*n;
    return i + ld*j;
  }

  template<class T>
  auto diagonal(T* buffer) const {
    using U = typename std::remove_const<T>::type;
    U* buf = const_cast<U*>(buffer);
    return Array<U,1>(buf, ArrayShape<1>(std::min(m, n), ld + 1));
  }

  template<class T>
  auto slice(const T* buffer, const std::pair<int,int>& rows,
     const std::pair<int,int>& cols) const {
    assert(1 <= rows.first && rows.first <= m && "start of row range out of bounds");
    assert(rows.second <= m && "end of row range out of bounds");
    assert(1 <= cols.first && cols.first <= n && "start of column range out of bounds");
    assert(cols.second <= n && "end of column range out of bounds");
    int i = rows.first;
    int r = std::max(0, rows.second - rows.first + 1);
    int j = cols.first;
    int c = std::max(0, cols.second - cols.first + 1);
    auto shp = ArrayShape<2>(r, c, ld);
    auto buf = shp.volume() ? buffer + (i - 1) + int64_t(ld)*(j - 1) : nullptr;
    return Array(buf, shp);
  }

  template<class T>
  auto slice(const T* buffer, const std::pair<int,int>& rows, const int j) const {
    assert(1 <= rows.first && rows.first <= m && "start of row range out of bounds");
    assert(rows.second <= m && "end of row range out of bounds");
    assert(1 <= j && j <= n && "column index out of bounds");
    int i = rows.first;
    int r = std::max(0, rows.second - rows.first + 1);
    auto shp = ArrayShape<1>(r);
    auto buf = shp.volume() ? buffer + (i - 1) + int64_t(ld)*(j - 1) : nullptr;
    return Array(buf, shp);
  }

  template<class T>
  auto slice(T* buffer, const int i, const std::pair<int,int>& cols) const {
    assert(1 <= i && i <= m && "row index out of bounds");
    assert(1 <= cols.first && cols.first <= n && "start of column range out of bounds");
    assert(cols.second <= n && "end of column range out of bounds");
    int j = cols.first;
    int c = std::max(0, cols.second - cols.first + 1);
    auto shp = ArrayShape<1>(c, ld);
    auto buf = shp.volume() ? buffer + (i - 1) + int64_t(ld)*(j - 1) : nullptr;
    return Array(buf, shp);
  }

  template<class T>
  auto slice(T* buffer, const int i, const int j) const {
    assert(1 <= i && i <= m && "row index out of bounds");
    assert(1 <= j && j <= n && "column index out of bounds");
    auto buf = buffer + (i - 1) + int64_t(ld)*(j - 1);
    auto shp = ArrayShape<0>();
    return Array(buf, shp);
  }

  template<class T>
  auto& dice(T* buffer, const int i, const int j) {
    assert(1 <= i && i <= m && "row index out of bounds");
    assert(1 <= j && j <= n && "column index out of bounds");
    return buffer[(i - 1) + int64_t(ld)*(j - 1)];
  }

  template<class T>
  const auto& dice(T* buffer, const int i, const int j) const {
    assert(1 <= i && i <= m && "row index out of bounds");
    assert(1 <= j && j <= n && "column index out of bounds");
    return buffer[(i - 1) + int64_t(ld)*(j - 1)];
  }

private:
  /**
   * Number of rows.
   */
  int m;

  /**
   * Number of columns.
   */
  int n;

  /**
   * Stride between columns.
   */
  int ld;
};

/**
 * Make a matrix shape.
 * 
 * @ingroup array
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * 
 * @return Shape.
 */
inline ArrayShape<2> make_shape(const int m, const int n) {
  return ArrayShape<2>(m, n);
}

/**
 * Make a scalar, vector or matrix shape.
 * 
 * @ingroup array
 * 
 * @tparam D Number of dimensions.
 * 
 * @param m Width. For a scalar or vector this is 1, for a matrix this is its
 * number of rows.
 * @param n Height. For a scalar this is 1, for a vector this is its length,
 * for a matrix this is its number of columns.
 * 
 * @return Shape.
 * 
 * @see width(), height()
 */
template<int D>
ArrayShape<D> make_shape(const int m, const int n) {
  if constexpr (D == 0) {
    assert(m == 1);
    assert(n == 1);
    return make_shape();
  } else if constexpr (D == 1) {
    assert(m == 1);
    return make_shape(n);
  } else {
    return make_shape(m, n);
  }
}

}
