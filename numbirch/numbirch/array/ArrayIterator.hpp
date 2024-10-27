/**
 * @file
 */
#pragma once

namespace numbirch {
/**
 * Iterator over Array.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam D Number of dimensions.
 */
template<class T, int D>
class ArrayIterator {
public:
  using value_type = std::remove_const_t<T>;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using iterator_category = std::random_access_iterator_tag;

  ArrayIterator() : buf(nullptr), shp(), pos(0) {
    //
  }

  explicit ArrayIterator(T* buf, const ArrayShape<D> shp,
      const difference_type pos) :
      buf(buf),
      shp(shp),
      pos(pos) {
    //
  }

  reference operator[](const difference_type i) {
    return buf[shp.serial(pos + i)];
  }

  const_reference operator[](const difference_type i) const {
    return buf[shp.serial(pos + i)];
  }

  difference_type operator-(const ArrayIterator& o) const {
    return pos - o.pos;
  }

  reference operator*() {
    return *get();
  }

  const_reference operator*() const {
    return *get();
  }

  pointer operator->() {
    return get();
  }

  const_pointer operator->() const {
    return get();
  }

  bool operator==(const ArrayIterator& o) const {
    return get() == o.get();
  }

  bool operator!=(const ArrayIterator& o) const {
    return get() != o.get();
  }

  bool operator<=(const ArrayIterator& o) const {
    return get() <= o.get();
  }

  bool operator<(const ArrayIterator& o) const {
    return get() < o.get();
  }

  bool operator>=(const ArrayIterator& o) const {
    return get() >= o.get();
  }

  bool operator>(const ArrayIterator& o) const {
    return get() > o.get();
  }

  ArrayIterator& operator+=(const difference_type i) {
    pos += i;
    return *this;
  }

  ArrayIterator operator+(const difference_type i) const {
    ArrayIterator result(*this);
    result += i;
    return result;
  }

  ArrayIterator& operator-=(const difference_type i) {
    pos -= i;
    return *this;
  }

  ArrayIterator operator-(const difference_type i) const {
    ArrayIterator result(*this);
    result -= i;
    return result;
  }

  ArrayIterator& operator++() {
    ++pos;
    return *this;
  }

  ArrayIterator operator++(int) {
    ArrayIterator result(*this);
    ++pos;
    return result;
  }

  ArrayIterator& operator--() {
    --pos;
    return *this;
  }

  ArrayIterator operator--(int) {
    ArrayIterator result(*this);
    --pos;
    return result;
  }

private:
  /**
   * Raw pointer for the current position.
   */
  pointer get() {
    return buf + shp.serial(pos);
  }

  /**
   * Raw pointer for the current position.
   */
  const_pointer get() const {
    return buf + shp.serial(pos);
  }

  /**
   * Buffer.
   */
  T* buf;

  /**
   * Shape.
   */
  ArrayShape<D> shp;

  /**
   * Position.
   */
  difference_type pos;
};
}
