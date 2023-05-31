/**
 * @file
 */
#pragma once

#include "numbirch/memory.hpp"
#include "numbirch/utility.hpp"
#include "numbirch/array/Shape.hpp"
#include "numbirch/array/Iterator.hpp"
#include "numbirch/array/Atomic.hpp"
#include "numbirch/array/Array.hpp"

#include <algorithm>
#include <numeric>
#include <utility>
#include <initializer_list>
#include <memory>
#include <atomic>

#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <cstring>

namespace numbirch::array {
using namespace numbirch;
}
namespace numbirch {
using namespace numbirch::array;
}

namespace numbirch::array {
/**
 * Multidimensional array with copy-on-write.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam D Number of dimensions, where `0 <= D <= 2`.
 * 
 * An Array supports the operations slice() and dice():
 * 
 * @li A slice() returns an Array of `D` dimensions or fewer. This includes a
 * `Array<T,0>` of one element.
 * @li A dice() returns a reference of type `const T&` or `T&` to an
 * individual element.
 * 
 * The use of dice() may trigger synchronization with the device to ensure
 * that all device reads and writes have concluded before the host can access
 * an individual element.
 */
template<class T, int D>
class ArrayCOW {
  template<class U, int E> friend class ArrayCOW;
public:
  static_assert(std::is_arithmetic_v<T>, "ArrayCOW is only for arithmetic types");
  static_assert(!std::is_const_v<T>, "ArrayCOW cannot have a const value type");
  static_assert(!std::is_reference_v<T>, "ArrayCOW cannot have a reference value type");

  using value_type = T;
  using shape_type = Shape<D>;

  /**
   * Number of dimensions.
   */
  static constexpr int ndims = D;

  /**
   * Constructor.
   */
  ArrayCOW() :
      ArrayCOW(make_shape<D>()) {
    //
  }

  /**
   * Constructor.
   *
   * @param shp Shape.
   */
  ArrayCOW(const Shape<D>& shp) :
      arr(shp.volume() > 0 ? new Array<T,D>(shp, 1) : nullptr) {
    //
  }

  /**
   * Constructor.
   *
   * @param value Fill value.
   * @param shp Shape.
   */
  ArrayCOW(const T& value, const Shape<D>& shp) :
      arr(shp.volume() > 0 ? new Array<T,D>(value, shp, 1) : nullptr) {
    //
  }

  /**
   * Constructor.
   *
   * @param value Fill value.
   * @param shp Shape.
   */
  template<class U>
  ArrayCOW(const ArrayCOW<U,0>& value, const Shape<D>& shp) :
      arr(shp.volume() > 0 ? new Array<T,D>(value, shp, 1) : nullptr) {
    //
  }

  /**
   * Constructor.
   *
   * @param l Lambda called to construct each element. Argument is a 1-based
   * serial index.
   * @param shp Shape.
   */
  template<class L, std::enable_if_t<std::is_invocable_r_v<T,L,int>,int> = 0>
  ArrayCOW(const L& l, const Shape<D>& shp) :
      arr(shp.volume() > 0 ? new Array<T,D>(l, shp, 1) : nullptr) {
    //
  }

  /**
   * Constructor.
   * 
   * @param slice Array.
   */
  ArrayCOW(const Array<T,D>& arr) :
      arr(arr.volume() > 0 ? new Array<T,D>(arr, 1) : nullptr) {
    //
  }

  /**
   * Constructor.
   * 
   * @param value Fill value.
   * 
   * Constructs an array of the given number of dimensions, with a single
   * element set to @p value.
   */
  ArrayCOW(const T& value) :
      arr(new Array<T,D>(value, 1)) {
    //
  }

  /**
   * Constructor.
   * 
   * @param value Fill value.
   * 
   * Constructs an array of the given number of dimensions, with a single
   * element set to @p value.
   */
  template<class U>
  ArrayCOW(const Array<U,0>& value) :
      arr(new Array<T,D>(value, 1)) {
    //
  }

  /**
   * Constructor.
   *
   * @param values Values.
   */
  template<int E = D, std::enable_if_t<E == 1,int> = 0>
  ArrayCOW(const std::initializer_list<T>& values) :
      arr(values.size() > 0 ? new Array<T,D>(
          make_shape(values.size()), 1) : nullptr) {
    //
  }

  /**
   * Constructor.
   *
   * @param values Values.
   */
  template<int E = D, std::enable_if_t<E == 2,int> = 0>
  ArrayCOW(const std::initializer_list<std::initializer_list<T>>& values) :
      arr(values.size() > 0 ? new Array<T,D>(
          make_shape(values.size(), values.begin()->size()), 1) : nullptr) {
    //
  }

  /**
   * Copy constructor.
   * 
   * @param o Source object.
   */
  ArrayCOW(const ArrayCOW& o) {
    auto s = o.arr.load();
    if (s) {
      s->incShared();
    }
    arr.store(s);  ///@todo Can we avoid atomic store on construction here?
  }

  /**
   * Move constructor.
   */
  ArrayCOW(ArrayCOW&& o) :
      arr(o.arr.exchange(nullptr)) {
    //
  }

  /**
   * Destructor.
   */
  ~ArrayCOW() {
    auto s = arr.load();
    if (s && s->decShared() == 0) {
      delete s;
    }
  }

  /**
   * Copy assignment.
   */
  ArrayCOW& operator=(const ArrayCOW& o) {
    auto s = o.arr.load();
    auto t = arr.exchange(s);
    if (s) {
      s->incShared();
    }
    if (t && t->decShared() == 0) {
      delete t;
    }
    return *this;
  }

  /**
   * Move assignment.
   */
  ArrayCOW& operator=(ArrayCOW&& o) {
    auto s = o.arr.exchange(nullptr);
    auto t = arr.exchange(s);
    if (t && t->decShared() == 0) {
      delete t;
    }
    return *this;
  }

  /**
   * Array copy assignment.
   */
  template<class U>
  ArrayCOW& operator=(const Array<U,D>& o) {
    *get() = o;
    return *this;
  }

  /**
   * Array move assignment.
   */
  ArrayCOW& operator=(Array<T,D>&& o) {
    if (o.bytes) {
      /* can claim ownership of buffer in new slice */
      auto s = new Array<T,D>(std::move(o), 1);
      auto t = arr.exchange(s);
      if (t && t->decShared() == 0) {
        delete t;
      }
      o.buf = nullptr;
      o.bytes = 0;
    } else {
      *get(false, false) = o;
    }
    return *this;
  }

  /**
   * Value assignment (fill).
   */
  ArrayCOW& operator=(const T& value) {
    *get(true, false) = value;
    return *this;
  }

  /**
   * Value assignment (fill).
   */
  template<class U, int E = D, std::enable_if_t<E != 0,int> = 0>
  ArrayCOW& operator=(const Array<U,0>& value) {
    *get(true, false) = value;
    return *this;
  }

  /**
   * @copydoc slice()
   */
  template<class... Args, std::enable_if_t<sizeof...(Args) == D && D != 0 &&
      !all_integral_v<Args...>,int> = 0>
  auto operator()(const Args... args) {
    return slice(args...);
  }

  /**
   * @copydoc slice()
   */
  template<class... Args, std::enable_if_t<sizeof...(Args) == D && D != 0 &&
      !all_integral_v<Args...>,int> = 0>
  const auto operator()(const Args... args) const {
    return slice(args...);
  }

  /**
   * @copydoc dice()
   */
  template<class... Args, std::enable_if_t<sizeof...(Args) == D && D != 0 &&
      all_integral_v<Args...>,int> = 0>
  T& operator()(const Args... args) {
    return dice(args...);
  }

  /**
   * @copydoc dice()
   */
  template<class... Args, std::enable_if_t<sizeof...(Args) == D && D != 0 &&
      all_integral_v<Args...>,int> = 0>
  const T& operator()(const Args... args) const {
    return dice(args...);
  }

  /**
   * Array conversion.
   */
  operator Array<T,D>() {
    return slice();
  }

  /**
   * Array conversion.
   */
  operator Array<T,D>() const {
    return slice();
  }

  /**
   * Value conversion (scalar only).
   */
  template<class U = T, int E = D, std::enable_if_t<
      std::is_arithmetic_v<U> && E == 0,int> = 0>
  operator U() const {
    return value();
  }

  /**
   * @copydoc value()
   */
  auto operator*() const {
    return value();
  }

  /**
   * For a scalar, the value; for a vector or matrix, `*this`.
   */
  auto value() const {
    if constexpr (D == 0) {
      return *dice();
    } else {
      return *this;
    }
  }

  /**
   * Whole array.
   */
  Array<T,D> slice() {
    auto s = get();
    return Array<T,D>(s->buf, s->streamAlloc, s->stream, s->shp);
  }

  /**
   * Whole array.
   */
  Array<T,D> slice() const {
    auto s = get();
    return Array<T,D>(s->buf, s->streamAlloc, s->stream, s->shp);
  }

  /**
   * Array.
   *
   * @tparam Args Argument types.
   * 
   * @param args Ranges or indices defining a slice. An index should be of
   * type `int`, and a range of type `std::pair` giving the first and last
   * indices of the range of elements to select. Indices and ranges are
   * 1-based.
   *
   * @return ArrayCOW, giving a view of the selected elements of the original
   * array.
   * 
   * The number of dimensions of the returned ArrayCOW is `D` minus the number of
   * indices among @p args. In particular, if @p args are all indices, the
   * return value will be of type `Array<T,0>` (c.f. dice()).
   */
  template<class... Args, std::enable_if_t<sizeof...(Args) == D &&
      D != 0,int> = 0>
  auto slice(const Args... args) {
    return get()->slice(args...);
  }

  /**
   * @copydoc slice()
   */
  template<class... Args, std::enable_if_t<sizeof...(Args) == D &&
      D != 0,int> = 0>
  const auto slice(const Args... args) const {
    return get()->slice(args...);
  }

  /**
   * Dice.
   * 
   * @param args Indices defining the element to select. The indices are
   * 1-based.
   *
   * @return Reference to the selected element.
   * 
   * @see slice(), which returns an `Array<T,0>` rather than `T&` for the same
   * arguments.
   */
  template<class... Args, std::enable_if_t<sizeof...(Args) == D && D != 0 &&
      all_integral_v<Args...>,int> = 0>
  T& dice(const Args... args) {
    return get()->dice(args...);
  }

  /**
   * @copydoc dice()
   */
  template<class... Args, std::enable_if_t<sizeof...(Args) == D && D != 0 &&
      all_integral_v<Args...>,int> = 0>
  const T& dice(const Args... args) const {
    return get()->dice(args...);
  }

  /**
   * Whole array as scalar.
   */
  Array<T,0> scal() const {
    return get()->scal();
  }

  /**
   * Whole array as vector.
   */
  Array<T,1> vec() const {
    return get()->vec();
  }

  /**
   * Whole array as matrix.
   * 
   * @param n Number of columns.
   */
  Array<T,2> mat(const int n) const {
    return get()->mat(n);
  }

  /**
   * Diagonal of matrix as vector.
   */
  template<int E = D, std::enable_if_t<E == 2,int> = 0>
  Array<T,1> diagonal() {
    return get()->diagonal();
  }

  /**
   * Diagonal of matrix as vector.
   */
  template<int E = D, std::enable_if_t<E == 2,int> = 0>
  Array<T,1> diagonal() const {
    return get()->diagonal();
  }

  /**
   * Shape.
   */
  Shape<D> shape() const {
    return get()->shp;
  }

  /**
   * Number of elements.
   */
  int64_t size() const {
    return get()->size();
  }

  /**
   * Number of elements allocated.
   */
  int64_t volume() const {
    return get()->volume();
  }

  /**
   * Length. For a scalar this is 1, for a vector its length, for a matrix its
   * number of rows. Same as rows().
   */
  int length() const {
    return get()->rows();
  }

  /**
   * Number of rows. For a scalar this is 1, for a vector its length, for a
   * matrix its number of rows. Same as length().
   */
  int rows() const {
    return get()->rows();
  }

  /**
   * Number of columns. For a scalar or vector this is 1, for a matrix its
   * number of columns.
   */
  int columns() const {
    return get()->columns();
  }

  /**
   * Width, in number of elements. This refers to the 2d memory layout of the
   * array, where the width is the number of elements in each contiguous
   * block. For a scalar or vector it is 1, for a matrix it is the number of
   * rows.
   */
  int width() const {
    return get()->width();
  }

  /**
   * Height, in number of elements. This refers to the 2d memory layout of the
   * array, where the height is the number of contiguous blocks. For a scalar
   * it is 1, for a vector it is the length, for a matrix it is the number of
   * columns.
   */
  int height() const {
    return get()->height();
  }

  /**
   * Stride, in number of elements. This refers to the 2d memory layout of the
   * array, where the stride is the number of elements between the first
   * element of each contiguous block. For a scalar it is 0, for a vector it
   * is the stride between elements, for a matrix it is the stride between
   * columns.
   */
  int stride() const {
    return get()->stride();
  }

  /**
   * Does the shape of this array conform to that of another? Two shapes
   * conform if they have the same number of dimensions and lengths along
   * those dimensions. Strides may differ.
   */
  template<class U, int E>
  bool conforms(const ArrayCOW<U,E>& o) const {
    return get()->conforms(o.shape());
  }

  /**
   * Is the array storage contiguous?
   */
  bool contiguous() const {
    return get()->contiguous();
  }

  /**
   * Iterator to the first element.
   */
  Iterator<T,D> begin() {
    return get()->begin();
  }

  /**
   * @copydoc begin()
   */
  Iterator<const T,D> begin() const {
    return get()->begin();
  }

  /**
   * Iterator to one past the last element.
   */
  Iterator<T,D> end() {
    return get()->end();
  }

  /**
   * @copydoc end()
   */
  Iterator<const T,D> end() const {
    return get()->end();
  }

  /**
   * Push a value onto the end of a vector, resizing it if necessary.
   * 
   * @param value The value.
   * 
   * push() is typically used when initializing vectors of unknown length on
   * host. It works by extending the array by one with a realloc(), then
   * pushing the new value.
   */
  template<int E = D, std::enable_if_t<E == 1,int> = 0>
  void push(const T& value) {
    auto s = get();
    if (s) {
      s->push(value);
    } else {
      arr.store(new Array<T,1>(value));
    }
  }

private:
  /**
   * Get the shared array.
   * 
   * @param alloc If a copy-on-write is triggered, allocate a new buffer?
   * @param copy If a copy-on-write is triggered, copy over the contents of
   * the current buffer?
   * 
   * @return Pointer to the shared array.
   * 
   * The arguments @p alloc and @p copy may be used as an optimization
   * according to the write operation, e.g. where copying over the contents
   * of the current buffer would be wasteful, as they will just be overwritten
   * again in an assignment.
   */
  Array<T,D>* get(const bool alloc = true, const bool copy = true) {
    auto s = arr.load();
    auto t = s;
    bool success = true;
    do {
      if (s->numShared() > 1) {
        /* copy on write */
        if (alloc) {
          if (copy) {
            t = new Array<T,D>(*s, 1);
          } else {
            t = new Array<T,D>(s->shp, 1);
          }
        } else {
          t = new Array<T,D>(Shape<D>(), 1);
        }
        success = arr.compare_exchange_strong(s, t);
        if (success) {
          if (s->decShared() == 0) {
            delete s;
          }
        } else {
          /* another thread performed an update in the meantime, try again */
          if (t->decShared() == 0) {
            delete t;
          }
        }
      } else {
        success = true;
      }
    } while (!success);
    return t;
  }

  /**
   * Get the shared array.
   */
  Array<T,D>* get() const {
    return arr.load();
  }

  /**
   * Shared array.
   */
  std::atomic<Array<T,D>*> arr;
};

template<class T>
ArrayCOW(const std::initializer_list<std::initializer_list<T>>&) ->
    ArrayCOW<std::decay_t<T>,2>;

template<class T>
ArrayCOW(const std::initializer_list<T>&) ->
    ArrayCOW<std::decay_t<T>,1>;

template<class T>
ArrayCOW(const T& value) ->
    ArrayCOW<std::decay_t<T>,0>;

template<class T, int D>
ArrayCOW(const T& value, const Shape<D>& shape) ->
    ArrayCOW<std::decay_t<T>,D>;

}
