/**
 * @file
 */
#pragma once

#include "numbirch/memory.hpp"
#include "numbirch/utility.hpp"
#include "numbirch/array/ArrayShape.hpp"
#include "numbirch/array/ArrayIterator.hpp"
#include "numbirch/array/ArrayControl.hpp"
#include "numbirch/array/Sliced.hpp"
#include "numbirch/array/Diced.hpp"

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
   * Default constructor.
   */
  Array() :
      isView(false) {
    allocate();
  }

  /**
   * Constructor.
   *
   * @param shp Shape.
   */
  Array(const shape_type& shp) :
      shp(shp),
      isView(false) {
    allocate();
  }

  /**
   * Constructor.
   * 
   * @param value Fill value.
   * 
   * Typically this is used for construction of scalars, which have one
   * element by default. Arrays of higher dimension have no elements by
   * default.
   */
  Array(const T value) :
      isView(false) {
    allocate();
    fill(value);
  }

  /**
   * Constructor.
   *
   * @param shp Shape.
   * @param value Fill value.
   */
  Array(const shape_type& shp, const T value) :
      shp(shp),
      isView(false) {
    allocate();
    fill(value);
  }

  /**
   * Constructor.
   *
   * @param shp Shape.
   * @param value Fill value.
   */
  Array(const shape_type& shp, const Array<T,0>& value) :
      shp(shp),
      isView(false) {
    allocate();
    fill(value);
  }

  /**
   * Constructor.
   *
   * @param shp Shape.
   * @param l Lambda called to construct each element. Argument is a 1-based
   * serial index.
   */
  template<class L, std::enable_if_t<std::is_invocable_r_v<T,L,int>,int> = 0>
  Array(const shape_type& shp, const L& l) :
      shp(shp),
      isView(false) {
    allocate();
    if (volume() > 0) {
      int64_t n = 0;
      auto iter = begin();
      auto to = end();
      for (; iter != to; ++iter) {
        *iter = l(++n);
      }
    }
  }

  /**
   * Constructor.
   *
   * @param values Values.
   */
  template<int E = D, std::enable_if_t<E == 1,int> = 0>
  Array(const std::initializer_list<T>& values) :
      shp(make_shape(values.size())),
      isView(false) {
    allocate();
    if (volume() > 0) {
      std::copy(values.begin(), values.end(), begin());
    }
  }

  /**
   * Constructor.
   *
   * @param values Values.
   */
  template<int E = D, std::enable_if_t<E == 2,int> = 0>
  Array(const std::initializer_list<std::initializer_list<T>>& values) :
      shp(make_shape(values.size(), values.begin()->size())),
      isView(false) {
    allocate();
    if (volume() > 0) {
      T* ptr = diced();
      int64_t t = 0;
      for (auto row : values) {
        for (auto x : row) {
          ptr[shp.transpose(t++)] = T(x);
        }
      }
    }
  }

  /**
   * @internal
   * 
   * Constructor from individual components. Used when forming views and
   * reshaping.
   */
  Array(ArrayControl* ctl, const shape_type& shp) :
      ctl(ctl),
      shp(shp),
      isView(true) {
    //
  }

  /**
   * Copy constructor.
   * 
   * @param o Source object.
   * @param immediate Copy immediately? By default this is false, enabling
   * copy-on-write. If the copy is to be written to immediately, there are
   * some minor savings in reference count overhead by setting this to true to
   * copy immediately instead.
   */
  Array(const Array& o, const bool immediate = false) :
      shp(o.shp),
      isView(false) {
    if (immediate || o.isView) {
      allocate();
      copy(o);
    } else if (volume() > 0) {
      auto c = o.control();
      assert(c);
      c->incShared();
      ctl.store(c);
    }
  }

  /**
   * Generic copy constructor.
   */
  template<class U, std::enable_if_t<is_arithmetic_v<U>,int> = 0>
  Array(const Array<U,D>& o) :
      shp(o.shp),
      isView(false) {
    allocate();
    copy(o);
  }

  /**
   * Move constructor.
   */
  Array(Array&& o) :
      isView(false) {
    if (o.isView) {
      shp = o.shp;
      allocate();
      copy(o);
    } else {
      swap(std::move(o), false);
    }
  }

  /**
   * Destructor.
   */
  ~Array() {
    if (!isView && volume() > 0) {
      ArrayControl* c = ctl.load();
      if (c && c->decShared() == 0) {
        delete c;
      }
    }
  }

  /**
   * Copy assignment.
   */
  Array& operator=(const Array& o) {
    if (isView) {
      copy(o);
    } else {
      swap(Array(o));
    }
    return *this;
  }

  /**
   * Move assignment.
   */
  Array& operator=(Array&& o) {
    if (isView) {
      copy(o);
    } else if (o.isView) {
      swap(Array(o));
    } else {
      swap(std::move(o));
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
  template<class U, int E = D, std::enable_if_t<E == 0 &&
      is_arithmetic_v<U>,int> = 0>
  operator U() const {
    return value();
  }

  /**
   * @copydoc value()
   * 
   * @see value()
   */
  decltype(auto) operator*() {
    return value();
  }

  /**
   * @copydoc operator*()
   */
  decltype(auto) operator*() const {
    return value();
  }

  /**
   * Value.
   * 
   * @return For a scalar, the value. For a vector or matrix, `*this`.
   */
  decltype(auto) value() {
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
  decltype(auto) value() const {
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
   * @copydoc slice()
   */
  template<class... Args, std::enable_if_t<sizeof...(Args) == D &&
      !all_integral_v<Args...>,int> = 0>
  auto operator()(const Args... args) {
    return slice(args...);
  }

  /**
   * @copydoc slice()
   */
  template<class... Args, std::enable_if_t<sizeof...(Args) == D &&
      !all_integral_v<Args...>,int> = 0>
  const auto operator()(const Args... args) const {
    return slice(args...);
  }

  /**
   * @copydoc dice()
   */
  template<class... Args, std::enable_if_t<sizeof...(Args) == D &&
      all_integral_v<Args...>,int> = 0>
  T& operator()(const Args... args) {
    return dice(args...);
  }

  /**
   * @copydoc slice()
   */
  template<class... Args, std::enable_if_t<sizeof...(Args) == D &&
      all_integral_v<Args...>,int> = 0>
  const T& operator()(const Args... args) const {
    return dice(args...);
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
  template<class... Args, std::enable_if_t<sizeof...(Args) == D,int> = 0>
  auto slice(const Args... args) {
    auto view = shp.slice(args...);
    return Array<T,view.dims()>(control(), view);
  }

  /**
   * @copydoc slice()
   */
  template<class... Args, std::enable_if_t<sizeof...(Args) == D,int> = 0>
  const auto slice(const Args... args) const {
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
  template<class... Args, std::enable_if_t<sizeof...(Args) == D &&
      all_integral_v<Args...>,int> = 0>
  T& dice(const Args... args) {
    return diced()[shp.dice(args...)];
  }

  /**
   * @copydoc dice()
   */
  template<class... Args, std::enable_if_t<sizeof...(Args) == D &&
      all_integral_v<Args...>,int> = 0>
  const T& dice(const Args... args) const {
    return diced()[shp.dice(args...)];
  }

  /**
   * Get the diagonal of a matrix as a vector.
   */
  template<int E = D, std::enable_if_t<E == 2,int> = 0>
  Array<T,1> diagonal() {
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
   * Push a value onto the end of a vector, resizing it if necessary.
   * 
   * @param value The value.
   * 
   * push() is typically used when initializing vectors of unknown length on
   * host. It works by extending the array by one with a realloc(), then
   * pushing the new value.
   */
  template<int E = D, std::enable_if_t<E == 1,int> = 0>
  void push(const T value) {
    assert(!isView);

    ArrayControl *c = nullptr, *d = nullptr;
    if (volume() == 0) {
      /* allocate new; as use cases for push() often see it called multiple
       * times in succession, overallocate a little to start to accommodate
       * some further pushes without reallocation */
      d = new ArrayControl(16*sizeof(T));
    } else {
      /* load control without obtaining lock */
      do {
        c = ctl.load();
      } while (!c);
      d = c;

      size_t newbytes = c->bytes;
      if ((volume() + stride())*sizeof(T) > newbytes) {
        /* must enlarge the allocation; as use cases for push() often see it
         * called multiple times in succession, overallocate to reduce the
         * need for reallocation on subsequent push() */
        newbytes = std::max(2*newbytes, size_t(volume() + stride())*sizeof(T));
      }

      if (c->numShared() > 1 || newbytes > c->bytes) {
        /* reacquire control while obtaining lock*/
        do {
          c = ctl.exchange(nullptr);
        } while (!c);
        d = c;

        if (c->numShared() > 1) {
          /* copy-on-write and resize simultaneously */
          d = new ArrayControl(*c, newbytes);
          if (c->decShared() == 0) {
            delete c;
          }
        } else if (newbytes > c->bytes) {
          /* reallocate only if necessary */
          c->realloc(newbytes);
        }
        c = nullptr;
      }
    }

    /* set new element; dicing preferable to slicing here given that the
     * typical use case for push() is reading from a file */
    static_cast<T*>(d->buf)[volume()] = value;
    //memset(Sliced<T>(d, volume(), true).data(), stride(), value, 1, 1);

    shp.extend(1);
    if (d != c) {
      ctl.store(d);  // also unlocks
    }
  }

  /**
   * Get underlying buffer for use in a slice operation.
   */
  Sliced<T> sliced() {
    if (volume() > 0) {
      return Sliced<T>(control(), offset(), true);
    } else {
      return Sliced<T>(nullptr, 0, true);
    }
  }

  /**
   * @copydoc sliced()
   */
  const Sliced<T> sliced() const {
    if (volume() > 0) {
      return Sliced<T>(control(), offset(), false);
    } else {
      return Sliced<T>(nullptr, 0, false);
    }
  }

  /**
   * Get underlying buffer for use in a dice operation.
   */
  Diced<T> diced() {
    if (volume() > 0) {
      return Diced<T>(control(), offset());
    } else {
      return Diced<T>(nullptr, 0);
    }
  }

  /**
   * @copydoc diced()
   */
  const Diced<T> diced() const {
    if (volume() > 0) {
      return Diced<T>(control(), offset());
    } else {
      return Diced<T>(nullptr, 0);
    }
  }

  /**
   * @internal
   * 
   * Get the control block.
   */
  ArrayControl* control() {
    ArrayControl* c = nullptr;
    if (volume() > 0) {  // avoid unnecessary atomics
      /* ctl is used as a lock, it may be set to nullptr while another thread
       * is working on a copy-on-write */
      do {
        c = ctl.load();
      } while (!c);

      /* copy on write if necessary */
      if (!isView && c->numShared() > 1) {
        /* acquire exclusive lock */
        do {
          c = ctl.exchange(nullptr);
        } while (!c);

        if (c->numShared() > 1) {
          /* copy on write */
          ArrayControl* d = new ArrayControl(*c);
          ctl.store(d);
          if (c->decShared() == 0) {
            delete c;
          }
          c = d;
        } else {
          /* another thread did copy on write, return the original */
          ctl.store(c);
        }
      }
    }
    return c;
  }

  /**
   * @internal
   * 
   * Get the control block.
   */
  ArrayControl* control() const {
    ArrayControl* c = nullptr;
    if (volume() > 0) {  // avoid unnecessary atomics
      /* ctl is used as a lock, it may be set to nullptr while another thread
       * is working on a copy-on-write */
      do {
        c = ctl.load();
      } while (!c);
    }
    return c;
  }

  /**
   * @internal
   * 
   * Can the array be reshaped? True if the elements are contiguous within its
   * allocation.
   */
  bool canReshape() const {
    return size() == 1 || size() == volume();
  }

  /**
   * @internal
   * 
   * Convert to scalar.
   */
  Array<T,0> scal() const {
    if constexpr (D == 0) {
      return *this;
    } else {
      assert(size() == 1);
      return Array<T,0>(control(), ArrayShape<0>(offset()));
    }
  }

  /**
   * @internal
   * 
   * Convert to vector.
   */
  Array<T,1> vec() const {
    if constexpr (D == 1) {
      return *this;
    } else {
      assert(canReshape());
      return Array<T,1>(control(), ArrayShape<1>(offset(), size(), 1));
    }
  }

  /**
   * @internal
   * 
   * Convert to matrix.
   * 
   * @param n Number of columns.
   */
  Array<T,2> mat(const int n) const {
    if constexpr (D == 2) {
      return *this;
    } else {
      assert(canReshape());
      assert(size() % n == 0);
      int m = size()/n;
      return Array<T,2>(control(), ArrayShape<2>(offset(), m, n, m));
    }
  }

private:
  /**
   * Fill with scalar value.
   *
   * @param value The value.
   */
  void fill(const T value) {
    if (volume() > 0) {
      memset(sliced().data(), stride(), value, width(), height());
    }
  }

  /**
   * Fill with scalar value.
   *
   * @param value The value.
   */
  template<class U>
  void fill(const Array<U,0>& value) {
    if (volume() > 0) {
      memcpy(sliced().data(), stride(), value.sliced().data(), value.stride(),
          width(), height());
    }
  }

  /**
   * Copy from another array.
   * 
   * @param o The other array.
   */
  template<class U>
  void copy(const Array<U,D>& o) {
    assert(conforms(o) && "array sizes are different");
    if (volume() > 0) {
      memcpy(sliced().data(), stride(), o.sliced().data(), o.stride(),
          width(), height());
    }
  }

  /**
   * Swap with another array.
   * 
   * @param o The other array.
   * @param allocated Does this have an allocation? For scalar arrays, the
   * volume is one, but the allocation may not have been performed yet if we
   * are in a constructor, and so the usual `volume() > 0` check is
   * insufficient.
   */
  void swap(Array&& o, const bool allocated = true) {
    assert(!isView);
    assert(!o.isView);
    auto c = (allocated && volume() > 0) ? ctl.exchange(nullptr) : nullptr;
    auto d = o.volume() > 0 ? o.ctl.exchange(nullptr) : nullptr;
    std::swap(shp, o.shp);
    if (volume() > 0) {
      assert(d);
      ctl.store(d);
    }
    if (o.volume() > 0) {
      o.ctl.store(c);
    }
  }

  /**
   * Allocate memory for this, leaving uninitialized.
   */
  void allocate() {
    shp.compact();
    if (volume() > 0) {
      ctl.store(new ArrayControl(shp.volume()*sizeof(T)));
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
Array(const ArrayShape<0>& shape, const T value) -> Array<T,0>;

template<class T>
Array(const ArrayShape<1>& shape, const T value) -> Array<T,1>;

template<class T>
Array(const ArrayShape<2>& shape, const T value) -> Array<T,2>;

template<class T>
Array(const ArrayShape<0>& shape, const Array<T,0>& value) -> Array<T,0>;

template<class T>
Array(const ArrayShape<1>& shape, const Array<T,0>& value) -> Array<T,1>;

template<class T>
Array(const ArrayShape<2>& shape, const Array<T,0>& value) -> Array<T,2>;

}
