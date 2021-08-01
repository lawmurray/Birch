/**
 * @file
 */
#pragma once

namespace libbirch {
/**
 * Shape and layout of an array.
 *
 * @ingroup libbirch
 * 
 * @tparam D Number of dimensions.
 */
template<int D>
class ArrayShape {
  static_assert(D <= 2, "arrays support only up to two dimensions");
};

/**
 * Shape and layout of a vector (one-dimensional array).
 */
template<>
class ArrayShape<1> {
public:
  ArrayShape(const int n = 0, const int inc = 1) :
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

  int stride() const {
    return inc;
  }

  int width() const {
    return 1;
  }

  int height() const {
    return n;
  }

  bool conforms(const ArrayShape<1>& o) const {
    return n == o.n;
  }

  void compact() {
    inc = 1;
  }

  template<class T>
  auto slice(T* buffer, const std::pair<int,int>& range) const {
    assert(1 <= range.first && range.first <= n && "start of range out of bounds");
    assert(range.second <= n && "end of range out of bounds");

    using U = typename std::remove_const<T>::type;
    U* buf = const_cast<U*>(buffer);

    int i = range.first;
    int len = std::max(0, range.second - range.first + 1);

    return Array<U,1>(buf + (i - 1)*int64_t(inc), ArrayShape<1>(len, inc));
  }

  template<class T>
  T& slice(T* buffer, const int i) const {
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
 * Shape and layout of a matrix (two-dimensional array) in column-major order.
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

  ArrayShape(const int m, const int n) :
      m(m),
      n(n),
      ld(m) {
    //
  }

  ArrayShape(const int m, const int n, const int ld) :
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

  int stride() const {
    return ld;
  }

  int width() const {
    return m;
  }

  int height() const {
    return n;
  }

  bool conforms(const ArrayShape<2>& o) const {
    return m == o.m && n == o.n;
  }

  void compact() {
    ld = m;
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
  auto slice(T* buffer, const std::pair<int,int>& rows,
     const std::pair<int,int>& cols) const {
    assert(1 <= rows.first && rows.first <= m && "start of row range out of bounds");
    assert(rows.second <= m && "end of row range out of bounds");
    assert(1 <= cols.first && cols.first <= n && "start of column range out of bounds");
    assert(cols.second <= n && "end of column range out of bounds");

    using U = typename std::remove_const<T>::type;
    U* buf = const_cast<U*>(buffer);

    int i = rows.first;
    int r = std::max(0, rows.second - rows.first + 1);
    int j = cols.first;
    int c = std::max(0, cols.second - cols.first + 1);

    return Array<U,2>(buf + (i - 1) + int64_t(ld)*(j - 1),
        ArrayShape<2>(r, c, ld));
  }

  template<class T>
  auto slice(T* buffer, const std::pair<int,int>& rows, const int j) const {
    assert(1 <= rows.first && rows.first <= m && "start of row range out of bounds");
    assert(rows.second <= m && "end of row range out of bounds");
    assert(1 <= j && j <= n && "column index out of bounds");

    using U = typename std::remove_const<T>::type;
    U* buf = const_cast<U*>(buffer);

    int i = rows.first;
    int r = std::max(0, rows.second - rows.first + 1);

    return Array<U,1>(buf + (i - 1) + int64_t(ld)*(j - 1),
        ArrayShape<1>(r, 1));
  }

  template<class T>
  auto slice(T* buffer, const int i, const std::pair<int,int>& cols) const {
    assert(1 <= i && i <= m && "row index out of bounds");
    assert(1 <= cols.first && cols.first <= n && "start of column range out of bounds");
    assert(cols.second <= n && "end of column range out of bounds");

    using U = typename std::remove_const<T>::type;
    U* buf = const_cast<U*>(buffer);

    int j = cols.first;
    int c = std::max(0, cols.second - cols.first + 1);

    return Array<U,1>(buf + (i - 1) + int64_t(ld)*(j - 1),
        ArrayShape<1>(c, ld));
  }

  template<class T>
  T& slice(T* buffer, const int i, const int j) const {
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
}
