/**
 * @file
 */
#pragma once

#include "numbirch/memory.hpp"
#include "numbirch/utility.hpp"
#include "numbirch/array/Shape.hpp"
#include "numbirch/array/Iterator.hpp"

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

namespace numbirch::disable_adl {
using namespace numbirch;
}
namespace numbirch {
using namespace numbirch::disable_adl;
}

namespace numbirch::disable_adl {
/**
 * Multidimensional array.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam D Number of dimensions, where `0 <= D <= 2`.
 */
template<class T, int D>
class Array {
  template<class U, int E> friend class Array;
  template<class U, int E> friend class ArrayCOW;
public:
  static_assert(std::is_arithmetic_v<T>, "Array is only for arithmetic types");
  static_assert(!std::is_const_v<T>, "Array cannot have a const value type");
  static_assert(!std::is_reference_v<T>, "Array cannot have a reference value type");

  using value_type = T;
  using shape_type = Shape<D>;

  /**
   * Number of dimensions.
   */
  static constexpr int ndims = D;

  /**
   * Default constructor.
   */
  Array() :
      Array(Shape<D>()) {
    //
  }

  /**
   * Constructor with reference count initialization.
   * 
   * @param shp Shape.
   */
  Array(const Shape<D>& shp) :
      buf(shp.volume() > 0 ? static_cast<T*>(malloc(shp.volume()*sizeof(T))) : nullptr),
      bytes(shp.volume()*sizeof(T)),
      streamAlloc(stream_get()),
      stream(stream_get()),
      shp(shp),
      own(true) {
    //
  }

  /**
   * Constructor.
   *
   * @param value Fill value.
   * @param shp Shape.
   */
  Array(const T& value, const Shape<D>& shp) :
      Array(shp) {
    stream_join(stream);
    memset(buf, stride(), value, width(), height());
    stream = stream_get();
  }

  /**
   * Constructor.
   *
   * @param value Fill value.
   * @param shp Shape.
   */
  template<class U>
  Array(const Array<U,0>& value, const Shape<D>& shp) :
      Array(shp) {
    stream_join(stream);
    memset(buf, stride(), value.buffer(), width(), height());
    stream = stream_get();
  }

  /**
   * Constructor.
   *
   * @param l Lambda called to construct each element. Argument is a 1-based
   * serial index.
   * @param shp Shape.
   */
  template<class L, std::enable_if_t<std::is_invocable_r_v<T,L,int>,int> = 0>
  Array(const L& l, const Shape<D>& shp) :
      Array(shp) {
    for (int64_t i = 0; i < size(); ++i) {
      buf[shp.serial(i)] = l(i);
    }
  }

  /**
   * Constructor.
   * 
   * @param value Fill value.
   * 
   * Constructs an array of the given number of dimensions, with a single
   * element set to @p value.
   */
  Array(const T& value) :
      Array(value, make_shape<D>(1, 1)) {
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
  Array(const Array<U,0>& value) :
      Array(value, make_shape<D>(1, 1)) {
    //
  }

  /**
   * Constructor.
   *
   * @param values Values.
   */
  template<int E = D, std::enable_if_t<E == 1,int> = 0>
  Array(const std::initializer_list<T>& values) :
      Array(make_shape(values.size())) {
    int i = 0;
    for (auto x : values) {
      buf[shp.serial(i++)] = x;
    }
  }

  /**
   * Constructor.
   *
   * @param values Values.
   */
  template<int E = D, std::enable_if_t<E == 2,int> = 0>
  Array(const std::initializer_list<std::initializer_list<T>>& values) :
      Array(make_shape(values.size(), values.begin()->size())) {
    int i = 0;
    for (auto row : values) {
      for (auto x : row) {
        buf[shp.transpose(i++)] = x;
      }
    }
  }

  /**
   * Copy constructor.
   * 
   * @param o Source object.
   * @param r Reference count. Typically zero, unless this is being used by a
   * shared array.
   */
  Array(const Array& o) :
      Array(o.shp) {
    stream_join(stream);
    stream_join(o.stream);
    memcpy(buf, stride(), o.buf, o.stride(), width(), height());
    stream = stream_get();
    const_cast<Array&>(o).stream = stream_get();
  }

  /**
   * Move constructor.
   * 
   * @param o Source object.
   * @param r Reference count. Typically zero, unless this is being used by a
   * shared array.
   */
  Array(Array&& o) :
      buf(nullptr),
      bytes(0),
      streamAlloc(stream_get()),
      stream(stream_get()),
      shp(),
      own(true) {
    if (o.own) {
      /* transfer ownership */
      std::swap(buf, o.buf);
      std::swap(bytes, o.bytes);
      std::swap(streamAlloc, o.streamAlloc);
      std::swap(stream, o.stream);
      std::swap(shp, o.shp);
    } else {
      /* copy */
      bytes = o.bytes;
      shp = o.shp;
      if (bytes) {
        buf = static_cast<T*>(malloc(bytes));
        stream_join(stream);
        stream_join(o.stream);
        memcpy(buf, stride(), o.buf, o.stride(), width(), height());
        stream = stream_get();
        const_cast<Array&>(o).stream = stream_get();
      }
    }
  }

  /**
   * Destructor.
   */
  ~Array() {
    if (own) {
      stream_finish(streamAlloc, stream);
      free(buf, bytes);
    }
  }

  /**
   * Copy assignment.
   */
  Array& operator=(const Array& o) {
    assert(own || conforms(o));
    if (own && !conforms(o)) {
      /* as this owns the buffer, a resize is allowed */
      stream_finish(streamAlloc, stream);
      buf = static_cast<T*>(realloc(buf, bytes, o.bytes));
      bytes = o.bytes;
      shp = o.shp;
    }
    stream_join(stream);
    stream_join(o.stream);
    memcpy(buf, stride(), o.buf, o.stride(), width(), height());
    stream = stream_get();
    const_cast<Array&>(o).stream = stream_get();
    return *this;
  }

  /**
   * Copy assignment.
   */
  template<class U>
  Array& operator=(const Array<U,D>& o) {
    assert(own || conforms(o));
    if (own && !conforms(o)) {
      /* as this owns the buffer, a resize is allowed */
      stream_finish(streamAlloc, stream);
      buf = static_cast<T*>(realloc(buf, bytes, o.bytes*sizeof(T)/sizeof(U)));
      bytes = o.bytes*sizeof(T)/sizeof(U);
      shp = o.shp;
    }
    stream_join(stream);
    stream_join(o.stream);
    memcpy(buf, stride(), o.buf, o.stride(), width(), height());
    stream = stream_get();
    const_cast<Array<U,D>&>(o).stream = stream_get();
    return *this;
  }

  /**
   * Move assignment.
   * 
   * @param o Source object.
   */
  Array& operator=(Array&& o) {
    if (own && o.own) {
      /* swap */
      std::swap(buf, o.buf);
      std::swap(bytes, o.bytes);
      std::swap(streamAlloc, o.streamAlloc);
      std::swap(stream, o.stream);
      std::swap(shp, o.shp);
    } else {
      /* copy */
      operator=(o);
    }
    return *this;
  }

  /**
   * Value assignment (fill).
   */
  Array& operator=(const T& value) {
    stream_join(stream);
    memset(buf, stride(), value, width(), height());
    stream = stream_get();
    return *this;
  }

  /**
   * Value assignment (fill).
   */
  template<class U, int E = D, std::enable_if_t<E != 0,int> = 0>
  Array& operator=(const Array<U,0>& value) {
    stream_join(stream);
    memset(buf, stride(), value.buffer(), width(), height());
    stream = stream_get();
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
   * Value.
   * 
   * @return For a scalar, the value. For a vector or matrix, `*this`.
   */
  auto value() const {
    if constexpr (D == 0) {
      return *buf;
    } else {
      return *this;
    }
  }

  /**
   * Whole array as slice.
   */
  Array<T,D> slice() {
    return Array(buf, bytes, streamAlloc, stream, shp);
  }

  /**
   * Whole array as slice.
   */
  Array<T,D> slice() const {
    return Array(buf, bytes, streamAlloc, stream, shp);
  }

  /**
   * Slice of array.
   *
   * @tparam Args Argument types.
   * 
   * @param args Ranges or indices defining a slice. An index should be of
   * type `int`, and a range of type `std::pair` giving the first and last
   * indices of the range of elements to select. Indices and ranges are
   * 1-based.
   *
   * @return Array, giving a view of the selected elements of the original
   * array.
   * 
   * The number of dimensions of the returned Array is `D` minus the number of
   * indices among @p args. In particular, if @p args are all indices, the
   * return value will be of type `Array<T,0>` a.k.a. `Scalar<T>` (c.f.
   * dice()).
   */
  template<class... Args, std::enable_if_t<sizeof...(Args) == D &&
      D != 0,int> = 0>
  auto slice(const Args... args) {
    auto offset = shp.offset(args...);
    auto range = shp.range(args...);
    return Array<T,range.dims()>(buf + offset, range.volume()*sizeof(T),
        streamAlloc, stream, range);
  }

  /**
   * @copydoc slice()
   */
  template<class... Args, std::enable_if_t<sizeof...(Args) == D &&
      D != 0,int> = 0>
  const auto slice(const Args... args) const {
    auto offset = shp.offset(args...);
    auto range = shp.range(args...);
    return Array<T,range.dims()>(buf + offset, range.volume()*sizeof(T),
        streamAlloc, stream, range);
  }

  /**
   * Dice of array.
   * 
   * @param args Indices defining the element to select. The indices are
   * 1-based.
   *
   * @return Reference to the selected element.
   * 
   * @see slice(), which returns a `Scalar<T>` rather than `T&` for the same
   * arguments.
   */
  template<class... Args, std::enable_if_t<sizeof...(Args) == D && D != 0 &&
      all_integral_v<Args...>,int> = 0>
  T& dice(const Args... args) {
    stream_wait(stream);
    return buf[shp.offset(args...)];
  }

  /**
   * @copydoc dice()
   */
  template<class... Args, std::enable_if_t<sizeof...(Args) == D && D != 0 &&
      all_integral_v<Args...>,int> = 0>
  const T& dice(const Args... args) const {
    stream_wait(stream);
    return buf[shp.offset(args...)];
  }

  /**
   * Convert to scalar.
   */
  Array<T,0> scal() const {
    if constexpr (D == 0) {
      return *this;
    } else {
      assert(size() == 1);
      return Array<T,0>(buf, bytes, streamAlloc, stream, make_shape());
    }
  }

  /**
   * Convert to vector.
   */
  Array<T,1> vec() const {
    if constexpr (D == 1) {
      return *this;
    } else {
      assert(contiguous());
      return Array<T,1>(buf, bytes, streamAlloc, stream, make_shape(size()));
    }
  }

  /**
   * Convert to matrix.
   * 
   * @param n Number of columns.
   */
  Array<T,2> mat(const int n) const {
    if constexpr (D == 2) {
      return *this;
    } else {
      assert(contiguous());
      assert(size() % n == 0);
      int m = size()/n;
      return Array<T,2>(buf, bytes, streamAlloc, stream, make_shape(m, n));
    }
  }

  /**
   * Get the diagonal of a matrix as a vector.
   */
  template<int E = D, std::enable_if_t<E == 2,int> = 0>
  Array<T,1> diagonal() {
    return Array<T,1>(buf, bytes, streamAlloc, stream, shp.diagonal());
  }

  /**
   * Get the diagonal of a matrix as a vector.
   */
  template<int E = D, std::enable_if_t<E == 2,int> = 0>
  Array<T,1> diagonal() const {
    return Array<T,1>(buf, bytes, streamAlloc, stream, shp.diagonal());
  }

  /**
   * Buffer.
   */
  T* buffer() {
    return buf;
  }

  /**
   * Buffer.
   */
  const T* buffer() const {
    return buf;
  }

  /**
   * Shape.
   */
  Shape<D> shape() const {
    return shp;
  }

  /**
   * Number of elements.
   */
  int64_t size() const {
    return shp.size();
  }

  /**
   * Number of elements allocated.
   */
  int64_t volume() const {
    return shp.volume();
  }

  /**
   * Length. For a scalar this is 1, for a vector its length, for a matrix its
   * number of rows. Same as rows().
   */
  int length() const {
    return shp.rows();
  }

  /**
   * Number of rows. For a scalar this is 1, for a vector its length, for a
   * matrix its number of rows. Same as length().
   */
  int rows() const {
    return shp.rows();
  }

  /**
   * Number of columns. For a scalar or vector this is 1, for a matrix its
   * number of columns.
   */
  int columns() const {
    return shp.columns();
  }

  /**
   * Width, in number of elements. This refers to the 2d memory layout of the
   * array, where the width is the number of elements in each contiguous
   * block. For a scalar or vector it is 1, for a matrix it is the number of
   * rows.
   */
  int width() const {
    return shp.width();
  }

  /**
   * Height, in number of elements. This refers to the 2d memory layout of the
   * array, where the height is the number of contiguous blocks. For a scalar
   * it is 1, for a vector it is the length, for a matrix it is the number of
   * columns.
   */
  int height() const {
    return shp.height();
  }

  /**
   * Stride, in number of elements. This refers to the 2d memory layout of the
   * array, where the stride is the number of elements between the first
   * element of each contiguous block. For a scalar it is 0, for a vector it
   * is the stride between elements, for a matrix it is the stride between
   * columns.
   */
  int stride() const {
    return shp.stride();
  }

  /**
   * Does the shape of this array conform to that of another? Two shapes
   * conform if they have the same number of dimensions and lengths along
   * those dimensions. Strides may differ.
   */
  template<class U, int E>
  bool conforms(const Array<U,E>& o) const {
    return shp.conforms(o.shp);
  }

  /**
   * Is the array storage contiguous?
   */
  bool contiguous() const {
    return shp.contiguous();
  }

  /**
   * Iterator to the first element.
   */
  Iterator<T,D> begin() {
    return Iterator<T,D>(buf, shp, 0);
  }

  /**
   * @copydoc begin()
   */
  Iterator<const T,D> begin() const {
    return Iterator<const T,D>(buf, shp, 0);
  }

  /**
   * Iterator to one past the last element.
   */
  Iterator<T,D> end() {
    return Iterator<T,D>(buf, shp, size());
  }

  /**
   * @copydoc end()
   */
  Iterator<const T,D> end() const {
    return Iterator<const T,D>(buf, shp, size());
  }

  /**
   * Push a value onto the end of a vector, resizing it if necessary.
   * 
   * @param value The value.
   * 
   * The allocation is doubled in size if necessary, using realloc(), in
   * expectation of further uses; push() is typically used to fill vectors of
   * unknown length, such as when reading from a stream.
   */
  template<int E = D, std::enable_if_t<E == 1,int> = 0>
  void push(const T& value) {
    assert(own);
    int oldvol = shp.volume();
    int newvol = oldvol + shp.stride();
    size_t oldbytes = bytes;
    size_t newbytes = newvol*sizeof(T);

    if (newbytes > oldbytes) {
      /* must enlarge the allocation; because use cases for push() often
       * see it called multiple times in succession, overallocate to reduce
       * the need for reallocation on subsequent push() */
      newbytes = std::max<size_t>(2*oldbytes, 64u);
      if (buf) {
        stream_finish(streamAlloc, stream);
        buf = static_cast<T*>(realloc(buf, oldbytes, newbytes));
      } else {
        buf = static_cast<T*>(malloc(newbytes));
        streamAlloc = stream_get();
      }
      bytes = newbytes;
    }

    /* set new element; dicing preferable to slicing here given that the
     * typical use case for push() is reading from a file */
    shp.extend(1);
    buf[shp.offset(length())] = value;
  }

private:
  /**
   * View constructor.
   * 
   * @param buf Buffer.
   * @param bytes Number of bytes.
   * @param streamAlloc Stream of allocation.
   * @param stream Stream of last operation.
   * @param shp Shape.
   */
  Array(T* buf, const size_t bytes, void* streamAlloc, void* stream,
      const Shape<D>& shp) :
      buf(buf),
      bytes(bytes),
      streamAlloc(streamAlloc),
      stream(stream),
      shp(shp),
      own(false) {
    //
  }

  /**
   * Buffer.
   */
  T* buf;

  /**
   * Number of bytes allocated. When nonzero, the array is the owner of the
   * buffer.
   */
  size_t bytes;

  /**
   * Stream of allocation.
   */
  void* streamAlloc;

  /**
   * Stream of last operation.
   */
  void* stream;

  /**
   * Shape.
   */
  Shape<D> shp;

  /**
   * Does this own the buffer?
   */
  bool own;
};

template<class T>
Array(const std::initializer_list<std::initializer_list<T>>&) ->
    Array<std::decay_t<T>,2>;

template<class T>
Array(const std::initializer_list<T>&) ->
    Array<std::decay_t<T>,1>;

template<class T>
Array(const T& value) ->
    Array<std::decay_t<T>,0>;

template<class T, int D>
Array(const T& value, const Shape<D>& shape) ->
    Array<std::decay_t<T>,D>;

template<class T, int D>
Array(const ArrayCOW<T,D>& a) -> Array<T,D>;

}
