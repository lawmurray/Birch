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
class Iterator {
public:
  using value_type = std::remove_const_t<T>;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using iterator_category = std::random_access_iterator_tag;

  explicit Iterator(T* buf, const Shape<D> shp, const difference_type pos) :
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

  difference_type operator-(const Iterator& o) const {
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

  bool operator==(const Iterator& o) const {
    return get() == o.get();
  }

  bool operator!=(const Iterator& o) const {
    return get() != o.get();
  }

  bool operator<=(const Iterator& o) const {
    return get() <= o.get();
  }

  bool operator<(const Iterator& o) const {
    return get() < o.get();
  }

  bool operator>=(const Iterator& o) const {
    return get() >= o.get();
  }

  bool operator>(const Iterator& o) const {
    return get() > o.get();
  }

  Iterator& operator+=(const difference_type i) {
    pos += i;
    return *this;
  }

  Iterator operator+(const difference_type i) const {
    Iterator result(*this);
    result += i;
    return result;
  }

  Iterator& operator-=(const difference_type i) {
    pos -= i;
    return *this;
  }

  Iterator operator-(const difference_type i) const {
    Iterator result(*this);
    result -= i;
    return result;
  }

  Iterator& operator++() {
    ++pos;
    return *this;
  }

  Iterator operator++(int) {
    Iterator result(*this);
    ++pos;
    return result;
  }

  Iterator& operator--() {
    --pos;
    return *this;
  }

  Iterator operator--(int) {
    Iterator result(*this);
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
  Shape<D> shp;

  /**
   * Position.
   */
  difference_type pos;
};
}
