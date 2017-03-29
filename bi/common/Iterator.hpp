/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Iterator over list.
 *
 * @ingroup compiler_common
 */
template<class T>
class Iterator {
public:
  /**
   * Constructor.
   *
   * @param o The list, `nullptr` gives a one-past-end iterator.
   */
  Iterator(const T* o = nullptr);

  Iterator<T>& operator++();
  Iterator<T> operator++(int);
  const T* operator*();
  bool operator==(const Iterator<T>& o) const;
  bool operator!=(const Iterator<T>& o) const;

private:
  /**
   * The list.
   */
  const T* o;
};
}
