/**
 * @file
 */
#pragma once

#include "numbirch/memory.hpp"
#include "numbirch/utility.hpp"
#include "numbirch/array/ArrayControl.hpp"
#include "numbirch/array/ArrayShape.hpp"
#include "numbirch/array/ArrayIterator.hpp"
#include "numbirch/array/Lock.hpp"

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

namespace numbirch {
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
 * @li A slice() returns an Array of `D` dimensions or fewer. This includes an
 * `Array<T,0>` a.k.a. `Scalar<T>` of one element.
 * @li A dice() returns a reference of type `const T&` or `T&` to an
 * individual element.
 * 
 * The "slice and dice" operator()() combines the two, acting as slice() when
 * the result would have one or more dimensions, and a dice() if the result
 * would have zero dimensions. 
 * 
 * Internally, an Array is in one of two corresponding states according to the
 * most recent such operation:
 * 
 * @li *sliced*, where it supports further slice() operations without
 * transition, or
 * @li *diced*, where it supports further dice() operations without
 * transition.
 * 
 * The transition between the two states is automatic, on demand. The only
 * implication is that of performance: a transition from *sliced* to *diced*
 * may require synchronization with the device to ensure that all device reads
 * and writes have concluded before the host can access an individual element.
 * A performance consideration is to be careful with the interleaving of
 * slice() and dice() to minimize the number of potential synchronizations.
 */
template<class T, int D>
class Array {
  template<class U, int E> friend class Array;
public:
  static_assert(is_arithmetic_v<T>, "Array is only for arithmetic types");
  static_assert(!std::is_const_v<T>, "Array cannot have const value type");
  using value_type = T;
  using shape_type = ArrayShape<D>;

  /**
   * Number of dimensions.
   */
  static constexpr int ndims = D;

  /**
   * Default constructor for scalar. The array contains one element.
   */
  template<int E = D, std::enable_if_t<E == 0,int> = 0>
  Array() :
      buf(nullptr),
      ctl(nullptr),
      shp(),
      isView(false),
      isDiced(false) {
    allocate();
  }

  /**
   * Constructor for non-scalar. The array is empty.
   */
  template<int E = D, std::enable_if_t<E != 0,int> = 0>
  Array() :
      buf(nullptr),
      ctl(nullptr),
      shp(),
      isView(false),
      isDiced(false) {
    //
  }

  /**
   * Constructor (scalar only).
   * 
   * @param value 
   */
  template<int E = D, std::enable_if_t<E == 0,int> = 0>
  Array(const T value) :
      buf(nullptr),
      ctl(nullptr),
      shp(),
      isView(false),
      isDiced(false) {
    allocate();
    fill(value);
  }

  /**
   * Constructor.
   *
   * @param shape Shape.
   */
  Array(const shape_type& shape) :
      buf(nullptr),
      ctl(nullptr),
      shp(shape),
      isView(false),
      isDiced(false) {
    allocate();
  }

  /**
   * Constructor.
   *
   * @param shape Shape.
   * @param value Fill value.
   */
  Array(const shape_type& shape, const T value) :
      buf(nullptr),
      ctl(nullptr),
      shp(shape),
      isView(false),
      isDiced(false) {
    allocate();
    fill(value);
  }

  /**
   * Constructor.
   *
   * @param values Values.
   */
  template<int E = D, std::enable_if_t<E == 1,int> = 0>
  Array(const std::initializer_list<T>& values) :
      buf(nullptr),
      ctl(nullptr),
      shp(values.size()),
      isView(false),
      isDiced(false) {
    allocate();
    std::uninitialized_copy(values.begin(), values.end(), begin());
  }

  /**
   * Constructor.
   *
   * @param values Values.
   */
  template<int E = D, std::enable_if_t<E == 2,int> = 0>
  Array(const std::initializer_list<std::initializer_list<T>>& values) :
      buf(nullptr),
      ctl(nullptr),
      shp(values.size(), values.begin()->size()),
      isView(false),
      isDiced(false) {
    allocate();
    T* ptr = diced();
    int64_t t = 0;
    for (auto row : values) {
      for (auto x : row) {
        new (ptr + shp.transpose(t++)) T(x);
      }
    }
  }

  /**
   * Constructor.
   *
   * @param l Lambda called to construct each element. Argument is a 1-based
   * serial index.
   * @param shape Shape.
   */
  template<class L, std::enable_if_t<std::is_invocable_r_v<T,L,int>,int> = 0>
  Array(const L& l, const shape_type& shape) :
      buf(nullptr),
      ctl(nullptr),
      shp(shape),
      isView(false),
      isDiced(false) {
    allocate();
    int64_t n = 0;
    for (auto iter = begin(); iter != end(); ++iter) {
      new (&*iter) T(l(++n));
    }
  }

  /**
   * View constructor.
   */
  Array(const T* buf, const shape_type& shape) :
      buf(const_cast<T*>(buf)),
      ctl(nullptr),
      shp(shape),
      isView(true),
      isDiced(false) {
    assert((buf == nullptr) == (shape.volume() == 0));
  }

  /**
   * Copy constructor.
   */
  Array(const Array& o) :
      buf(nullptr),
      ctl(nullptr),
      shp(o.shp),
      isView(false),
      isDiced(false) {
    if (!o.isView) {
      std::tie(ctl, buf, isDiced) = o.share();
    } else {
      compact();
      allocate();
      uninitialized_copy(o);
    }
  }

  /**
   * Generic copy constructor.
   */
  template<class U, std::enable_if_t<std::is_convertible_v<U,T>,int> = 0>
  Array(const Array<U,D>& o) :
      buf(nullptr),
      ctl(nullptr),
      shp(o.shp),
      isView(false),
      isDiced(false) {
    compact();
    allocate();
    uninitialized_copy(o);
  }

  /**
   * Move constructor.
   */
  Array(Array&& o) :
      buf(nullptr),
      ctl(nullptr),
      shp(o.shp),
      isView(false),
      isDiced(false) {
    if (!o.isView) {
      swap(o);
    } else {
      compact();
      allocate();
      uninitialized_copy(o);
    }
  }

  /**
   * Destructor.
   */
  ~Array() {
    release();
  }

  /**
   * Copy assignment.
   */
  Array& operator=(const Array& o) {
    assign(o);
    return *this;
  }

  /**
   * Move assignment.
   */
  Array& operator=(Array&& o) {
    if (!isView && !o.isView) {
      swap(o);
    } else {
      assign(o);
    }
    return *this;
  }

  /**
   * Value assignment. Fills the entire array with the given value.
   */
  Array& operator=(const T value) {
    fill(value);
    wait();
    return *this;
  }

  /**
   * Value conversion (scalar only).
   */
  template<int E = D, std::enable_if_t<E == 0,int> = 0>
  operator T&() {
    return value();
  }

  /**
   * Value conversion (scalar only).
   */
  template<int E = D, std::enable_if_t<E == 0,int> = 0>
  operator const T() const {
    return value();
  }

  /**
   * Value conversion (scalar only).
   */
  template<class U, int E = D, std::enable_if_t<E == 0 &&
      is_scalar_v<U> && promotes_to_v<T,U>,int> = 0>
  operator U() const {
    return value();
  }

  /**
   * @copydoc value()
   * 
   * @see value()
   */
  auto& operator*() {
    return value();
  }

  /**
   * @copydoc operator*()
   */
  const auto& operator*() const {
    return value();
  }

  /**
   * Iterator to the first element.
   */
  ArrayIterator<T,D> begin() {
    return ArrayIterator<T,D>(diced(), shape(), 0);
  }

  /**
   * @copydoc begin()
   */
  ArrayIterator<const T,D> begin() const {
    return ArrayIterator<const T,D>(diced(), shape(), 0);
  }

  /**
   * Iterator to one past the last element.
   */
  ArrayIterator<T,D> end() {
    return ArrayIterator<T,D>(diced(), shape(), size());
  }

  /**
   * @copydoc end()
   */
  ArrayIterator<const T,D> end() const {
    return ArrayIterator<const T,D>(diced(), shape(), size());
  }

  /**
   * Value.
   * 
   * @return For a scalar, the value. For a vector or matrix, `*this`.
   */
  auto& value() {
    if constexpr (D == 0) {
      return *diced();
    } else {
      return *this;
    }
  }

  /**
   * Value.
   * 
   * @return For a scalar, the value. For a vector or matrix, `*this`.
   */
  const auto& value() const {
    if constexpr (D == 0) {
      return *diced();
    } else {
      return *this;
    }
  }

  /**
   * Is the value (for a scalar) or are the elements (for vectors and
   * matrices) available without waiting for asynchronous device operations to
   * complete?
   * 
   * In the current implementation, once a device operation is launched that
   * depends on the buffer, this returns false until a call to value() (for a
   * scalar) or an element is accessed (for vectors and matrices), after which
   * it returns true.
   */
  bool has_value() const {
    return isDiced;
  }

  /**
   * Slice.
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
  template<class... Args>
  auto slice(Args&&... args) {
    return shp.slice(sliced(), std::forward<Args>(args)...);
  }

  /**
   * @copydoc slice()
   */
  template<class... Args>
  const auto slice(Args&&... args) const {
    return shp.slice(sliced(), std::forward<Args>(args)...);
  }

  /**
   * Dice.
   * 
   * @param args Indices defining the element to select. The indices are
   * 1-based.
   *
   * @return Reference to the selected element.
   * 
   * @see slice(), which returns a `Scalar<T>` rather than `T&` for the same
   * arguments.
   */
  template<class... Args, std::enable_if_t<all_integral_v<Args...>,int> = 0>
  T& dice(const Args&... args) {
    return shp.dice(diced(), args...);
  }

  /**
   * @copydoc dice()
   */
  template<class... Args, std::enable_if_t<all_integral_v<Args...>,int> = 0>
  const T dice(const Args&... args) const {
    return shp.dice(diced(), args...);
  }

  /**
   * @copydoc slice()
   * 
   * @note operator()() is overloaded to behave as slice() when one or more
   * arguments is a range (`std::pair<int,int>`) and dice() otherwise.
   */
  template<class... Args, std::enable_if_t<!all_integral_v<Args...>,int> = 0>
  auto operator()(Args&&... args) {
    return slice(std::forward<Args>(args)...);
  }

  /**
   * @copydoc slice()
   * 
   * @note operator()() is overloaded to behave as slice() when one or more
   * arguments is a range (`std::pair<int,int>`) and dice() otherwise.
   */
  template<class... Args, std::enable_if_t<!all_integral_v<Args...>,int> = 0>
  auto operator()(Args&&... args) const {
    return slice(std::forward<Args>(args)...);
  }

  /**
   * @copydoc dice()
   * 
   * @note operator()() is overloaded to behave as slice() when one or more
   * arguments is a range (`std::pair<int,int>`) and dice() otherwise.
   */
  template<class... Args, std::enable_if_t<all_integral_v<Args...>,int> = 0>
  T& operator()(const Args&... args) {
    return dice(args...);
  }

  /**
   * @copydoc dice()
   * 
   * @note operator()() is overloaded to behave as slice() when one or more
   * arguments is a range (`std::pair<int,int>`) and dice() otherwise.
   */
  template<class... Args, std::enable_if_t<all_integral_v<Args...>,int> = 0>
  const T operator()(const Args&... args) const {
    return dice(args...);
  }

  /**
   * Get the diagonal of a matrix as a vector.
   */
  template<int E = D, std::enable_if_t<E == 2,int> = 0>
  Array<T,1> diagonal() {
    return shp.diagonal(sliced());
  }

  /**
   * Get the diagonal of a matrix as a vector.
   */
  template<int E = D, std::enable_if_t<E == 2,int> = 0>
  Array<T,1> diagonal() const {
    return shp.diagonal(sliced());
  }

  /**
   * Get underlying buffer for use in a slice operation.
   */
  T* sliced() {
    own();
    isDiced = false;
    return buf;
  }

  /**
   * @copydoc sliced()
   */
  const T* sliced() const {
    const_cast<Array*>(this)->isDiced = false;
    return buf;
  }

  /**
   * Get underlying buffer for use in a dice operation.
   */
  T* diced() {
    own();
    if (!isDiced) {
      wait();
      isDiced = true;
    }
    return buf;
  }

  /**
   * @copydoc diced()
   */
  const T* diced() const {
    if (!isDiced) {
      wait();
      const_cast<Array*>(this)->isDiced = true;
    }
    return buf;
  }

  /**
   * Get underlying buffer. Synonym of sliced().
   */
  T* data() {
    return sliced();
  }

  /**
   * Get underlying buffer. Synonym of sliced().
   */
  const T* data() const {
    return sliced();
  }

  /**
   * Shape.
   */
  ArrayShape<D> shape() const {
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
   * Push an element onto the end of a vector. The vector length is increased
   * by one.
   *
   * @param x Value.
   */
  void push(const T x) {
    insert(size(), x);
  }

  /**
   * Insert an element into a vector at a given position. The element is
   * inserted just before the existing element at that position, and the
   * vector length increased by one.
   *
   * @param i Position.
   * @param x Value.
   */
  void insert(const int i, const T x) {
    static_assert(D == 1, "insert() supports only one-dimensional arrays");
    assert(!isView);

    auto n = size();
    ArrayShape<1> s(n + 1);
    if (!buf) {
      Array tmp(s, x);
      swap(tmp);
    } else {
      buf = (T*)realloc((void*)sliced(), s.volume()*sizeof(T));
      ///@todo Use memcpy()
      std::memmove(diced() + i + 1, diced() + i, (n - i)*sizeof(T));
      new (diced() + i) T(x);
      shp = s;
    }
  }

  /**
   * Erase elements from a vector from a given position forward. The vector
   * length is reduced by the number of elements erased.
   *
   * @param i Position.
   * @param len Number of elements to erase.
   */
  void erase(const int i, const int len = 1) {
    static_assert(D == 1, "erase() supports only one-dimensional arrays");
    assert(!isView);
    assert(len > 0);
    assert(size() >= len);

    auto n = size();
    ArrayShape<1> s(n - len);
    if (s.size() == 0) {
      release();
    } else {
      ///@todo Use memcpy()
      std::memmove(diced() + i, diced() + i + len, (n - len - i)*sizeof(T));
      buf = (T*)realloc(sliced(), s.volume()*sizeof(T));
    }
    shp = s;
  }

  /**
   * Fill with scalar value.
   *
   * @param value The value.
   */
  void fill(const T value) {
    memset(data(), shp.stride(), value, shp.width(), shp.height());
  }

  /**
   * Clear, erasing all elements.
   */
  void clear() {
    release();
    shp = ArrayShape<D>();
  }

private:
  /**
   * Copy from another array. For a view the shapes of the two arrays must
   * conform, otherwise a resize is permitted.
   */
  template<class U, int E, std::enable_if_t<D == E &&
      std::is_convertible_v<U,T>,int> = 0>
  void assign(const Array<U,E>& o) {
    if (isView) {
      uninitialized_copy(o);
    } else {
      Array tmp(o);
      swap(tmp);
    }
  }

  /**
   * Copy from another array when this is uninitialized.
   */
  template<class U>
  void uninitialized_copy(const Array<U,D>& o) {
    assert(conforms(o) && "array sizes are different");
    memcpy(data(), shp.stride(), o.data(), o.shp.stride(), shp.width(),
        shp.height());
  }

  /**
   * Swap with another array.
   */
  void swap(Array& o) {
    std::swap(buf, o.buf);
    std::swap(ctl, o.ctl);
    std::swap(shp, o.shp);
    assert(!isView);
    assert(!o.isView);
    std::swap(isDiced, o.isDiced);
  }

  /**
   * Compact the array by reducing the volume to match the size. This is only
   * possible prior to allocation.
   */
  void compact() {
    assert(!buf);
    shp = shp.compact();
  }

  /**
   * Allocate memory for this, leaving uninitialized.
   */
  void allocate() {
    assert(!buf);
    assert(!ctl);
    assert(!isDiced);
    buf = (T*)malloc(volume()*sizeof(T));
  }

  /**
   * Release the buffer, deallocating if this is the last reference to it.
   */
  void release() {
    if (!isView) {
      if (!ctl || ctl->decShared() == 0) {
        free(buf, volume()*sizeof(T));
        delete ctl;
      }
    }
    buf = nullptr;
    ctl = nullptr;
    isView = false;
    isDiced = false;
  }

  /**
   * Share the buffer.
   * 
   * @return A pair giving pointers to the control block and buffer.
   */
  std::tuple<ArrayControl*,T*,bool> share() {
    assert(!isView);
    #pragma omp flush(ctl, buf)
    if (ctl) {
      ctl->incShared();
    } else if (buf) {
      lock.set();
      if (ctl) {  // another thread may have set in the meantime
        ctl->incShared();
      } else {
        ctl = new ArrayControl(2);  // one ref for current, one ref for new
      }
      lock.unset();
    }
    return std::make_tuple(ctl, buf, isDiced);
  }

  /**
   * @copydoc share()
   */
  std::tuple<ArrayControl*,T*,bool> share() const {
    return const_cast<Array*>(this)->share();
  }

  /**
   * Ensure that the buffer is not shared, copying it if necessary. That the
   * buffer is shared is indicated by the presence of a control block.
   */
  void own() {
    #pragma omp flush(ctl, buf)
    if (ctl) {
      assert(!isView);
      lock.set();
      if (!ctl) {  // another thread may have cleared in the meantime
        // last reference optimization already applied by another thread
      } else if (ctl->decShared() == 0) {
        /* apply last reference optimization */
        delete ctl;
        ctl = nullptr;
      } else {
        isDiced = false;
        T* ptr = (T*)malloc(volume()*sizeof(T));
        memcpy(ptr, stride(), this->buf, stride(), width(), height());
        wait();
        buf = ptr;
        ctl = nullptr;
      }
      lock.unset();
    }
  }

  /**
   * Buffer containing elements.
   */
  T* buf;

  /**
   * Control block for sharing the buffer.
   */
  ArrayControl* ctl;

  /**
   * Shape.
   */
  ArrayShape<D> shp;

  /**
   * Is this a view of another array? A view has stricter assignment
   * semantics, as it cannot be resized or moved.
   */
  bool isView;

  /**
   * Is the array prepared for element-wise access? If false, the array is
   * prepared for block-wise access. This is not atomic as a thread only
   * synchronizes with its own device anyway to convert from block-wise to
   * element-wise operations.
   */
  bool isDiced;

  /**
   * Lock for operations requiring mutual exclusion.
   */
  Lock lock;
};

template<class T>
Array(const std::initializer_list<std::initializer_list<T>>&) -> Array<T,2>;

template<class T>
Array(const std::initializer_list<T>&) -> Array<T,1>;

template<class T>
Array(const T value) -> Array<T,0>;

template<class T>
Array(const ArrayShape<1>& shape, const T value) -> Array<T,1>;

template<class T>
Array(const ArrayShape<2>& shape, const T value) -> Array<T,2>;

template<class T>
Array(const ArrayShape<1>& shape, const Array<T,0>& value) -> Array<T,1>;

template<class T>
Array(const ArrayShape<2>& shape, const Array<T,0>& value) -> Array<T,2>;

template<class T, int D>
Array(T* buf, const ArrayShape<D>& shape) -> Array<T,D>;

template<class T, int D>
Array(const T* buf, const ArrayShape<D>& shape) -> Array<T,D>;

}
