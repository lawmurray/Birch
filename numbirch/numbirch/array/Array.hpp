/**
 * @file
 */
#pragma once

#include "numbirch/memory.hpp"
#include "numbirch/utility.hpp"
#include "numbirch/array/ArrayShape.hpp"
#include "numbirch/array/ArrayIterator.hpp"
#include "numbirch/array/ArrayControl.hpp"

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
 * The use of dice() may trigger synchronization with the device to ensure
 * that all device reads and writes have concluded before the host can access
 * an individual element.
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
      ctl(nullptr),
      shp(),
      isView(false) {
    allocate();
  }

  /**
   * Constructor for non-scalar. The array is empty.
   */
  template<int E = D, std::enable_if_t<E != 0,int> = 0>
  Array() :
      ctl(nullptr),
      shp(),
      isView(false) {
    //
  }

  /**
   * Constructor (scalar only).
   * 
   * @param value 
   */
  template<int E = D, std::enable_if_t<E == 0,int> = 0>
  Array(const T value) :
      ctl(nullptr),
      shp(),
      isView(false) {
    allocate();
    fill(value);
  }

  /**
   * Constructor.
   *
   * @param shp Shape.
   */
  Array(const shape_type& shp) :
      ctl(nullptr),
      shp(shp),
      isView(false) {
    allocate();
  }

  /**
   * Constructor.
   *
   * @param shp Shape.
   * @param value Fill value.
   */
  Array(const shape_type& shp, const T value) :
      ctl(nullptr),
      shp(shp),
      isView(false) {
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
      ctl(nullptr),
      shp(make_shape(values.size())),
      isView(false) {
    allocate();
    std::copy(values.begin(), values.end(), begin());
  }

  /**
   * Constructor.
   *
   * @param values Values.
   */
  template<int E = D, std::enable_if_t<E == 2,int> = 0>
  Array(const std::initializer_list<std::initializer_list<T>>& values) :
      ctl(nullptr),
      shp(make_shape(values.size(), values.begin()->size())),
      isView(false) {
    allocate();
    T* ptr = diced();
    int64_t t = 0;
    for (auto row : values) {
      for (auto x : row) {
        ptr[shp.transpose(t++)] = T(x);
      }
    }
  }

  /**
   * Constructor.
   *
   * @param l Lambda called to construct each element. Argument is a 1-based
   * serial index.
   * @param shp Shape.
   */
  template<class L, std::enable_if_t<std::is_invocable_r_v<T,L,int>,int> = 0>
  Array(const L& l, const shape_type& shp) :
      ctl(nullptr),
      shp(shp),
      isView(false) {
    allocate();
    int64_t n = 0;
    auto iter = begin();
    auto to = end();
    for (; iter != to; ++iter) {
      *iter = T(l(++n));
    }
  }

  /**
   * View constructor.
   */
  Array(ArrayControl* ctl, const shape_type& shp) :
      ctl(ctl),
      shp(shp),
      isView(true) {
    assert((ctl == nullptr) == (shp.volume() == 0));
  }

  /**
   * Copy constructor.
   */
  Array(const Array& o) :
      ctl(nullptr),
      shp(o.shp),
      isView(false) {
    if (!o.isView && volume() > 0) {
      auto c = o.control();
      c->incShared();
      ctl.store(c);
    } else {
      shp.compact();
      allocate();
      copy(o);
    }
  }

  /**
   * Generic copy constructor.
   */
  template<class U, std::enable_if_t<std::is_convertible_v<U,T>,int> = 0>
  Array(const Array<U,D>& o) :
      ctl(nullptr),
      shp(o.shp),
      isView(false) {
    shp.compact();
    allocate();
    copy(o);
  }

  /**
   * Move constructor.
   */
  Array(Array&& o) :
      ctl(nullptr),
      shp(o.shp),
      isView(false) {
    if (!o.isView) {
      swap(o);
    } else {
      shp.compact();
      allocate();
      copy(o);
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
  template<class U, std::enable_if_t<is_arithmetic_v<U>,int> = 0>
  Array& operator=(const U value) {
    fill(value);
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
   */
  bool has_value() const {
    return volume() == 0 || control()->test();
  }

  /**
   * @copydoc slice()
   */
  template<class... Args, std::enable_if_t<!all_integral_v<Args...>,int> = 0>
  auto operator()(const Args... args) {
    return slice(args...);
  }

  /**
   * @copydoc slice()
   */
  template<class... Args, std::enable_if_t<!all_integral_v<Args...>,int> = 0>
  auto operator()(const Args... args) const {
    return slice(args...);
  }

  /**
   * @copydoc slice()
   */
  template<class... Args, std::enable_if_t<all_integral_v<Args...>,int> = 0>
  auto operator()(const Args... args) {
    return slice(args...);
  }

  /**
   * @copydoc slice()
   */
  template<class... Args, std::enable_if_t<all_integral_v<Args...>,int> = 0>
  auto operator()(const Args... args) const {
    return slice(args...);
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
  auto slice(const Args... args) {
    own();
    auto view = shp.slice(args...);
    return Array<T,view.dims()>(control(), view);
  }

  /**
   * @copydoc slice()
   */
  template<class... Args>
  auto slice(const Args... args) const {
    auto view = shp.slice(args...);
    return Array<T,view.dims()>(control(), view);
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
  T& dice(const Args... args) {
    own();
    return diced()[shp.dice(args...)];
  }

  /**
   * @copydoc dice()
   */
  template<class... Args, std::enable_if_t<all_integral_v<Args...>,int> = 0>
  const T dice(const Args... args) const {
    return diced()[shp.dice(args...)];
  }

  /**
   * Get the diagonal of a matrix as a vector.
   */
  template<int E = D, std::enable_if_t<E == 2,int> = 0>
  Array<T,1> diagonal() {
    own();
    return Array<T,1>(control(), shp.diagonal());
  }

  /**
   * Get the diagonal of a matrix as a vector.
   */
  template<int E = D, std::enable_if_t<E == 2,int> = 0>
  Array<T,1> diagonal() const {
    return Array<T,1>(control(), shp.diagonal());
  }

  /**
   * Shape.
   */
  ArrayShape<D> shape() const {
    return shp;
  }

  /**
   * Offset into buffer.
   */
  int64_t offset() const {
    return shp.offset();
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
   * Fill with scalar value.
   *
   * @param value The value.
   */
  void fill(const T value) {
    memset(data(sliced()), stride(), value, width(), height());
  }

  /**
   * Push a value onto the end of a vector, resizing it if necessary.
   * 
   * @param value The value.
   * 
   * push() is typically used when initializing vectors of unknown length on
   * host. It works by extending the array by one with a realloc(), then
   * pushing the new value with dice(), not slice(), to keep operations on the
   * host.
   */
  template<int E = D, std::enable_if_t<E == 1,int> = 0>
  void push(const T value) {
    assert(!isView);

    ArrayControl* d = nullptr;
    size_t newsize = (volume() + stride())*sizeof(T);
    if (volume() == 0) {
      /* allocate new */
      d = new ArrayControl(newsize);
    } else {
      /* use ctl as a lock: exchange with nullptr, copy on write if necessary,
       * resize buffer */
      ArrayControl* c = ctl.exchange(nullptr);
      while (!c) {
        c = ctl.exchange(nullptr);
      }
      if (c->numShared() > 1) {
        /* copy-on-write and resize simultaneously */
        d = new ArrayControl(*c, newsize);
        if (c->decShared() == 0) {
          delete c;
        }
      } else {
        d = c;
        d->realloc(newsize);
      }
    }

    /* update shape, return ctl, write last element */
    shp.extend(1);
    d->template diced<T>(offset())[shp.dice(rows() - 1)] = value;
    ctl.store(d);
  }

  /**
   * Clear, erasing all elements.
   */
  void clear() {
    release();
    shp = ArrayShape<D>();
  }

  /**
   * Get underlying buffer for use in a slice operation.
   */
  Recorder<T> sliced() {
    if (volume() > 0) {
      own();
      return control()->template sliced<T>(offset());
    } else {
      return Recorder<T>();
    }
  }

  /**
   * @copydoc sliced()
   */
  Recorder<const T> sliced() const {
    if (volume() > 0) {
      return control()->template sliced<const T>(offset());
    } else {
      return Recorder<const T>();
    }
  }

  /**
   * Get underlying buffer for use in a dice operation.
   */
  T* diced() {
    if (volume() > 0) {
      own();
      return control()->template diced<T>(offset());
    } else {
      return nullptr;
    }
  }

  /**
   * @copydoc diced()
   */
  const T* diced() const {
    if (volume() > 0) {
      return control()->template diced<const T>(offset());
    } else {
      return nullptr;
    }
  }

  /**
   * @internal
   * 
   * Get the control block.
   */
  ArrayControl* control() {
    if (volume() > 0) {
      /* ctl is used as a lock, it may be set to nullptr while another thread
       * is working on a copy-on-write, see own() */
      auto c = ctl.load();
      while (!c) {
        c = ctl.load();
      }
      return c;
    } else {
      return nullptr;
    }
  }

  ArrayControl* control() const {
    return const_cast<Array*>(this)->control();
  }

private:
  /**
   * Copy from another array. For a view the shapes of the two arrays must
   * conform, otherwise a resize is permitted.
   */
  template<class U, int E, std::enable_if_t<D == E &&
      std::is_convertible_v<U,T>,int> = 0>
  void assign(const Array<U,E>& o) {
    if (!std::is_same_v<U,T> || isView) {
      copy(o);
    } else {
      Array tmp(o);
      swap(tmp);
    }
  }

  /**
   * Copy from another array.
   */
  template<class U>
  void copy(const Array<U,D>& o) {
    assert(conforms(o) && "array sizes are different");
    memcpy(data(sliced()), stride(), data(o.sliced()), o.stride(), width(),
        height());
  }

  /**
   * Swap with another array.
   */
  void swap(Array& o) {
    std::swap(ctl, o.ctl);
    std::swap(shp, o.shp);
    assert(!isView);
    assert(!o.isView);
  }

  /**
   * Allocate memory for this, leaving uninitialized.
   */
  void allocate() {
    assert(!ctl.load());
    if (volume() > 0) {
      ctl.store(new ArrayControl(volume()*sizeof(T)));
    }
  }

  /**
   * Release the buffer, deallocating if this is the last reference to it.
   */
  void release() {
    if (!isView && volume() > 0) {
      ArrayControl* c = ctl.exchange(nullptr);
      // ^ c can still be nullptr here, e.g. due to the use of swap in the
      //   move constructor
      if (c && c->decShared() == 0) {
        delete c;
      }
    }
  }

  /**
   * Ensure that the buffer is not shared, copying it if necessary. That the
   * buffer is shared is indicated by the presence of a control block.
   */
  void own() {
    if (!isView && volume() > 0) {
      /* use ctl as a lock: exchange with nullptr, copy on write if necessary,
       * restore value */
      ArrayControl* c = ctl.exchange(nullptr);
      while (!c) {
        c = ctl.exchange(nullptr);
      }
      if (c->numShared() > 1) {
        ctl.store(new ArrayControl(*c));
        if (c->decShared() == 0) {
          delete c;
        }
      } else {
        ctl.store(c);
      }
    }
  }

  /**
   * Buffer control block.
   */
  Atomic<ArrayControl*> ctl;

  /**
   * Shape.
   */
  ArrayShape<D> shp;

  /**
   * Is this a view of another array? A view has stricter assignment
   * semantics, as it cannot be resized or moved.
   */
  bool isView;
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

}
