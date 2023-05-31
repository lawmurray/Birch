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
class Shape {
  static_assert(0 <= D && D <= 2, "Array supports up to two dimensions");
};

/**
 * Shape and layout of a scalar (zero-dimensional array).
 * 
 * @ingroup array
 */
template<>
class Shape<0> {
public:
  static constexpr int dims() {
    return 0;
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

  static constexpr bool conforms(const Shape<0>& o) {
    return true;
  }

  template<int D>
  static constexpr bool conforms(const Shape<D>& o) {
    return false;
  }

  static constexpr bool contiguous() {
    return true;
  }

  Shape<0> compact() const {
    return *this;
  }

  int64_t serial(const int64_t s = 0) const {
    return s;
  }

  int64_t transpose(const int64_t t = 0) const {
    return t;
  }
};

/**
 * Make a scalar shape.
 * 
 * @ingroup array
 */
inline Shape<0> make_shape() {
  return Shape<0>();
}

/**
 * Shape and layout of a vector (one-dimensional array).
 * 
 * @ingroup array
 */
template<>
class Shape<1> {
public:
  explicit Shape(const int n = 0, const int inc = 1) :
      n(n),
      inc(inc) {
    assert(n >= 0);
    assert(inc >= 1);
  }

  static constexpr int dims() {
    return 1;
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

  bool conforms(const Shape<1>& o) const {
    return n == o.n;
  }

  template<int D>
  static constexpr bool conforms(const Shape<D>& o) {
    return false;
  }

  bool contiguous() const {
    return inc == 1;
  }

  Shape<1> compact() const {
    Shape<1> o(*this);
    o.inc = 1;
    return o;
  }

  void extend(const int extra) {
    n += extra;
  }

  Shape<1> range(const std::pair<int,int>& range) const {
    assert(1 <= range.first && range.first <= n &&
        "start of range out of bounds");
    assert(range.second <= n && "end of range out of bounds");
    int len = std::max(0, range.second - range.first + 1);
    return Shape<1>(len, inc);
  }

  Shape<0> range(const int i) const {
    return Shape<0>();
  }

  int64_t offset(const std::pair<int,int>& range) const {
    return offset(range.first);
  }

  int64_t offset(const int i) const {
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
inline Shape<1> make_shape(const int n) {
  return Shape<1>(n);
}

/**
 * Shape and layout of a matrix (two-dimensional array) in column-major order.
 * 
 * @ingroup array
 */
template<>
class Shape<2> {
public:
  explicit Shape(const int m = 0, const int n = 0, const int ld = 0) :
      m(m),
      n(n),
      ld(ld) {
    assert(m >= 0);
    assert(n >= 0);
    assert(ld >= m);
  }

  static constexpr int dims() {
    return 2;
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

  bool conforms(const Shape<2>& o) const {
    return m == o.m && n == o.n;
  }

  template<int D>
  static constexpr bool conforms(const Shape<D>& o) {
    return false;
  }

  bool contiguous() const {
    return m == ld;
  }

  Shape<2> compact() const {
    Shape<2> o(*this);
    o.ld = o.m;
    return o;
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

  Shape<1> diagonal() const {
    return Shape<1>(std::min(m, n), ld + 1);
  }

  Shape<2> range(const std::pair<int,int>& rows,
     const std::pair<int,int>& cols) const {
    assert(1 <= rows.first && rows.first <= m &&
        "start of row range out of bounds");
    assert(rows.second <= m && "end of row range out of bounds");
    assert(1 <= cols.first && cols.first <= n &&
        "start of column range out of bounds");
    assert(cols.second <= n && "end of column range out of bounds");
    int r = std::max(0, rows.second - rows.first + 1);
    int c = std::max(0, cols.second - cols.first + 1);
    return Shape<2>(r, c, ld);
  }

  Shape<1> range(const std::pair<int,int>& rows, const int j) const {
    assert(1 <= rows.first && rows.first <= m &&
        "start of row range out of bounds");
    assert(rows.second <= m && "end of row range out of bounds");
    assert(1 <= j && j <= n && "column index out of bounds");
    int r = std::max(0, rows.second - rows.first + 1);
    return Shape<1>(r);
  }

  Shape<1> range(const int i, const std::pair<int,int>& cols) const {
    assert(1 <= i && i <= m && "row index out of bounds");
    assert(1 <= cols.first && cols.first <= n &&
        "start of column range out of bounds");
    assert(cols.second <= n && "end of column range out of bounds");
    int c = std::max(0, cols.second - cols.first + 1);
    return Shape<1>(c, ld);
  }

  Shape<0> range(const int i, const int j) const {
    assert(1 <= i && i <= m && "row index out of bounds");
    assert(1 <= j && j <= n && "column index out of bounds");
    return Shape<0>();
  }

  int64_t offset(const std::pair<int,int>& rows,
     const std::pair<int,int>& cols) const {
    return offset(rows.first, cols.first);
  }

  int64_t offset(const std::pair<int,int>& rows, const int j) const {
    return offset(rows.first, j);
  }

  int64_t offset(const int i, const std::pair<int,int>& cols) const {
    return offset(i, cols.first);
  }

  int64_t offset(const int i, const int j) const {
    assert(1 <= i && i <= m && "row index out of bounds");
    assert(1 <= j && j <= n && "column index out of bounds");
    return (i - 1) + int64_t(ld)*(j - 1);
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
inline Shape<2> make_shape(const int m, const int n) {
  return Shape<2>(m, n, m);
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
Shape<D> make_shape(const int m, const int n) {
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
