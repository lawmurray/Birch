/**
 * @file
 */
#pragma once

#include <utility>
#include <algorithm>

#include <cassert>
#include <cstdint>

namespace numbirch {
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
  ArrayShape(const int64_t k = 0) : k(k) {
    assert(k >= 0);
  }

  static constexpr int dims() {
    return 0;
  }

  int64_t offset() const {
    return k;
  }

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

  void compact() {
    k = 0;
  }

  void clear() {
    k = 0;
  }

  int64_t serial(const int64_t s = 0) const {
    return s;
  }

  int64_t transpose(const int64_t t = 0) const {
    return t;
  }

private:
  /**
   * Offset.
   */
  int64_t k;
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
  explicit ArrayShape(const int64_t k = 0, const int n = 0,
      const int inc = 1) :
      k(k),
      n(n),
      inc(inc) {
    assert(k >= 0);
    assert(n >= 0);
    assert(inc >= 1);
  }

  static constexpr int dims() {
    return 1;
  }

  int64_t offset() const {
    return k;
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

  void compact() {
    k = 0;
    inc = 1;
  }

  void clear() {
    k = 0;
    n = 0;
    inc = 1;
  }

  void extend(const int extra) {
    n += extra;
  }

  ArrayShape<1> slice(const std::pair<int,int>& range) const {
    assert(1 <= range.first && range.first <= n &&
        "start of range out of bounds");
    assert(range.second <= n && "end of range out of bounds");
    int64_t off = k + (range.first - 1)*int64_t(inc);
    int len = std::max(0, range.second - range.first + 1);
    return ArrayShape<1>(off, len, inc);
  }

  ArrayShape<0> slice(const int i) const {
    assert(1 <= i && i <= n && "index out of bounds");
    int64_t off = k + (i - 1)*int64_t(inc);
    return ArrayShape<0>(off);
  }

  int64_t dice(const int i) const {
    assert(1 <= i && i <= n && "index out of bounds");
    return (i - 1)*int64_t(inc);
  }

  /**
   * Compute the 0-based offset to the @p s th element in serial storage
   * order.
   *
   * @param s Serial index, 0-based.
   */
  int64_t serial(const int64_t s = 0) const {
    return s*inc;
  }

  /**
   * Compute the 0-based offset to the @p t th element in transpose storage
   * order.
   *
   * @param t Transpose index, 0-based.
   */
  int64_t transpose(const int64_t t = 0) const {
    return t*inc;
  }

private:
  /**
   * Offset.
   */
  int64_t k;

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
  return ArrayShape<1>(0, n);
}

/**
 * Shape and layout of a matrix (two-dimensional array) in column-major order.
 * 
 * @ingroup array
 */
template<>
class ArrayShape<2> {
public:
  ArrayShape(const int64_t k = 0, const int m = 0, const int n = 0,
      const int ld = 0) :
      k(k),
      m(m),
      n(n),
      ld(ld) {
    assert(k >= 0);
    assert(m >= 0);
    assert(n >= 0);
    assert(ld >= m);
  }

  static constexpr int dims() {
    return 2;
  }

  int64_t offset() const {
    return k;
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

  void compact() {
    k = 0;
    ld = m;
  }

  void clear() {
    k = 0;
    m = 0;
    n = 0;
    ld = 0;
  }

  /**
   * Compute the 0-based offset to the @p s th element in serial storage
   * order.
   *
   * @param s Serial index, 0-based.
   */
  int64_t serial(const int64_t s = 0) const {
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
  int64_t transpose(const int64_t t = 0) const {
    int64_t i = t/n;
    int64_t j = t - i*n;
    return i + ld*j;
  }

  ArrayShape<1> diagonal() const {
    return ArrayShape<1>(k, std::min(m, n), ld + 1);
  }

  ArrayShape<2> slice(const std::pair<int,int>& rows,
     const std::pair<int,int>& cols) const {
    assert(1 <= rows.first && rows.first <= m &&
        "start of row range out of bounds");
    assert(rows.second <= m && "end of row range out of bounds");
    assert(1 <= cols.first && cols.first <= n &&
        "start of column range out of bounds");
    assert(cols.second <= n && "end of column range out of bounds");
    int r = std::max(0, rows.second - rows.first + 1);
    int c = std::max(0, cols.second - cols.first + 1);
    int64_t off = k + (rows.first - 1) + int64_t(ld)*(cols.first - 1);
    return ArrayShape<2>(off, r, c, ld);
  }

  ArrayShape<1> slice(const std::pair<int,int>& rows, const int j) const {
    assert(1 <= rows.first && rows.first <= m &&
        "start of row range out of bounds");
    assert(rows.second <= m && "end of row range out of bounds");
    assert(1 <= j && j <= n && "column index out of bounds");
    int r = std::max(0, rows.second - rows.first + 1);
    int64_t off = k + (rows.first - 1) + int64_t(ld)*(j - 1);
    return ArrayShape<1>(off, r);
  }

  ArrayShape<1> slice(const int i, const std::pair<int,int>& cols) const {
    assert(1 <= i && i <= m && "row index out of bounds");
    assert(1 <= cols.first && cols.first <= n &&
        "start of column range out of bounds");
    assert(cols.second <= n && "end of column range out of bounds");
    int c = std::max(0, cols.second - cols.first + 1);
    int64_t off = k + (i - 1) + int64_t(ld)*(cols.first - 1);
    return ArrayShape<1>(off, c, ld);
  }

  ArrayShape<0> slice(const int i, const int j) const {
    assert(1 <= i && i <= m && "row index out of bounds");
    assert(1 <= j && j <= n && "column index out of bounds");
    int64_t off = k + (i - 1) + int64_t(ld)*(j - 1);
    return ArrayShape<0>(off);
  }

  int64_t dice(const int i, const int j) const {
    assert(1 <= i && i <= m && "row index out of bounds");
    assert(1 <= j && j <= n && "column index out of bounds");
    return (i - 1) + int64_t(ld)*(j - 1);
  }

private:
  /**
   * Offset.
   */
  int64_t k;

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
  return ArrayShape<2>(0, m, n, m);
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
