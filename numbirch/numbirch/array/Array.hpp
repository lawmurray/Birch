/**
 * @file
 */
#pragma once

#include "numbirch/memory.hpp"
#include "numbirch/type.hpp"
#include "numbirch/array/external.hpp"
#include "numbirch/array/ArrayControl.hpp"
#include "numbirch/array/ArrayShape.hpp"
#include "numbirch/array/ArrayIterator.hpp"
#include "numbirch/array/Atomic.hpp"
#include "numbirch/array/Lock.hpp"

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
 * @li *diced*, where it supports further dice() operations without transition.
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
    if (!is_arithmetic_v<T>) {
      dicer();
      initialize();
    }
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
  Array(const T& value) :
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
    if (!is_arithmetic_v<T>) {
      dicer();
      initialize();
    }
  }

  /**
   * Constructor.
   *
   * @param shape Shape.
   * @param value Fill value.
   */
  Array(const shape_type& shape, const T& value) :
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
   * @tparam ...Args Constructor parameter types.
   *
   * @param shape Shape.
   * @param args Constructor arguments.
   */
  template<class... Args>
  Array(const shape_type& shape, Args&&... args) :
      buf(nullptr),
      ctl(nullptr),
      shp(shape),
      isView(false),
      isDiced(false) {
    allocate();
    dicer();
    initialize(std::forward<Args>(args)...);
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
    dicer();
    std::uninitialized_copy(values.begin(), values.end(), beginInternal());
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
    dicer();
    int64_t t = 0;
    for (auto row : values) {
      for (auto x : row) {
        new (buf + shp.transpose(t++)) T(x);
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
    dicer();
    int64_t n = 0;
    for (auto iter = beginInternal(); iter != endInternal(); ++iter) {
      new (&*iter) T(l(++n));
    }
  }

  /**
   * View constructor.
   */
  Array(T* buf, const shape_type& shape) :
      buf(buf),
      ctl(nullptr),
      shp(shape),
      isView(true),
      isDiced(false) {
    //
  }

  /**
   * Copy constructor.
   */
  Array(const Array& o) :
      buf(nullptr),
      ctl(nullptr),
      shp(),
      isView(false),
      isDiced(false) {
    shp = o.shp;
    if (!o.isView && is_arithmetic_v<T>) {
      ArrayControl* ctl;
      std::tie(ctl, buf, isDiced) = o.share();
      this->ctl.store(ctl);
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
      shp(),
      isView(false),
      isDiced(false) {
    shp = o.shp;
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
      shp(),
      isView(false),
      isDiced(false) {
    if (!o.isView) {
      swap(o);
    } else {
      shp = o.shp;
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
   * Value assignment (scalar only).
   */
  template<int E = D, std::enable_if_t<E == 0,int> = 0>
  Array& operator=(const T& value) {
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
  operator const T&() const {
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
   * Member access (scalar only).
   */
  template<int E = D, std::enable_if_t<(E == 0) &&
      !is_arithmetic_v<T>,int> = 0>
  T& operator->() {
    return value();
  }

  /**
   * Member access (scalar only).
   */
  template<int E = D, std::enable_if_t<(E == 0) &&
      !is_arithmetic_v<T>,int> = 0>
  const T& operator->() const {
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
    own();
    dicer();
    return ArrayIterator<T,D>(buf, shp);
  }

  /**
   * @copydoc begin()
   */
  ArrayIterator<T,D> begin() const {
    dicer();
    return ArrayIterator<T,D>(buf, shp);
  }

  /**
   * Iterator to one past the last element.
   */
  ArrayIterator<T,D> end() {
    return begin().operator+(size());
  }

  /**
   * @copydoc end()
   */
  ArrayIterator<T,D> end() const {
    return begin().operator+(size());
  }

  /**
   * Value.
   * 
   * @return For a scalar, the value. For a vector or matrix, `*this`.
   */
  auto& value() {
    if constexpr (D == 0) {
      own();
      dicer();
      return *buf;
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
      dicer();
      return *buf;
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
    return !is_arithmetic_v<T> || isDiced;
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
    own();
    slicer();
    return shp.slice(buf, std::forward<Args>(args)...);
  }

  /**
   * @copydoc slice()
   */
  template<class... Args>
  const auto slice(Args&&... args) const {
    slicer();
    return shp.slice(buf, std::forward<Args>(args)...);
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
    own();
    dicer();
    return shp.dice(buf, args...);
  }

  /**
   * @copydoc dice()
   */
  template<class... Args, std::enable_if_t<all_integral_v<Args...>,int> = 0>
  const T& dice(const Args&... args) const {
    slicer();
    return shp.dice(buf, args...);
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
  const T& operator()(const Args&... args) const {
    return dice(args...);
  }

  /**
   * Underlying buffer.
   */
  T* data() {
    own();
    slicer();
    return buf;
  }

  /**
   * @copydoc data()
   */
  const T* data() const {
    slicer();
    return buf;
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
   * element of each contiguous block. For a scalar it is 1, for a vector it
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
   * Get the diagonal of a matrix as a vector.
   */
  template<int E = D, std::enable_if_t<E == 2,int> = 0>
  Array<T,1> diagonal() const {
    return shp.diagonal(buf);
  }

  /**
   * Push an element onto the end of a vector. The vector length is increased
   * by one.
   *
   * @param x Value.
   */
  void push(const T& x) {
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
  void insert(const int i, const T& x) {
    static_assert(D == 1, "insert() supports only one-dimensional arrays");
    assert(!isView);

    auto n = size();
    ArrayShape<1> s(n + 1);
    if (!buf) {
      Array tmp(s, x);
      swap(tmp);
    } else {
      own();
      if (is_arithmetic_v<T>) {
        slicer();
        buf = (T*)realloc((void*)buf, s.volume()*sizeof(T));
        dicer();
      } else {
        buf = (T*)std::realloc((void*)buf, s.volume()*sizeof(T));
      }
      std::memmove((void*)(buf + i + 1), (void*)(buf + i),
          (n - i)*sizeof(T));
      new (buf + i) T(x);
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
      own();
      dicer();
      std::destroy(buf + i, buf + i + len);
      std::memmove((void*)(buf + i), (void*)(buf + i + len),
          (n - len - i)*sizeof(T));
      if (is_arithmetic_v<T>) {
        slicer();
        buf = (T*)realloc((void*)buf, s.volume()*sizeof(T));
      } else {
        buf = (T*)std::realloc((void*)buf, s.volume()*sizeof(T));
      }
    }
    shp = s;
  }

  /**
   * Clear the array, erasing all elements.
   */
  void clear() {
    release();
    shp = ArrayShape<D>();
  }

private:
  /**
   * Iterator for use internally.
   */
  ArrayIterator<T,D> beginInternal() {
    return ArrayIterator<T,D>(buf, shp);
  }

  /**
   * @copydoc beginInternal()
   */
  ArrayIterator<T,D> beginInternal() const {
    return ArrayIterator<T,D>(buf, shp);
  }

  /**
   * Iterator for use internally.
   */
  ArrayIterator<T,D> endInternal() {
    return beginInternal().operator+(size());
  }

  /**
   * @copydoc endInternal()
   */
  ArrayIterator<T,D> endInternal() const {
    return beginInternal().operator+(size());
  }

  /**
   * Copy assignment. For a view the shapes of the two arrays must conform,
   * otherwise a resize is permitted.
   */
  void assign(const Array& o) {
    if (isView) {
      assert(conforms(o) && "array sizes are different");
      if (is_arithmetic_v<T>) {
        slicer();
        memcpy(data(), shp.stride()*sizeof(T), o.data(),
            o.shp.stride()*sizeof(T), shp.width()*sizeof(T),
            shp.height());
      } else {
        auto n = std::min(size(), o.size());
        auto begin1 = o.beginInternal();
        auto end1 = begin1.operator+(n);
        auto begin2 = beginInternal();
        auto end2 = begin2.operator+(n);
        if (begin1 <= begin2 && begin2 < end1) {
          std::copy_backward(begin1, end1, end2);
        } else {
          std::copy(begin1, end1, begin2);
        }
      }
    } else {
      Array tmp(o);
      swap(tmp);
    }
  }

  /**
   * Swap with another array.
   */
  void swap(Array& o) {
    assert(!isView);
    assert(!o.isView);
    std::swap(shp, o.shp);
    std::swap(buf, o.buf);
    std::swap(ctl, o.ctl);
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
    assert(!ctl.load());
    assert(!isDiced);

    if (is_arithmetic_v<T>) {
      slicer();
      buf = (T*)malloc(volume()*sizeof(T));
    } else {
      buf = (T*)std::malloc(volume()*sizeof(T));
    }
  }

  /**
   * Release the buffer, deallocating if this is the last reference to it.
   */
  void release() {
    if (!isView) {
      auto ctl = this->ctl.exchange(nullptr);
      if (!ctl || ctl->decShared() == 0) {
        if (is_arithmetic_v<T>) {
          free((void*)buf);
        } else {
          std::destroy(beginInternal(), endInternal());
          std::free((void*)buf);
        }
        delete ctl;
      }
    }
    buf = nullptr;
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
    auto ctl = this->ctl.load();
    if (ctl) {
      ctl->incShared();
    } else if (buf) {
      lock.set();
      ctl = this->ctl.load();  // another thread may have updated in meantime
      if (ctl) {
        ctl->incShared();
      } else {
        ctl = new ArrayControl(2);  // one ref for current, one ref for new
        this->ctl.store(ctl);
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
    auto ctl = this->ctl.load();
    if (ctl) {
      assert(!isView);
      lock.set();
      ctl = this->ctl.load();  // another thread may have updated in meantime
      if (!ctl) {
        // last reference optimization already applied by another thread
      } else if (ctl->numShared() == 1) {
        /* apply last reference optimization */
        delete ctl;
        this->ctl.store(nullptr);
      } else {
        T* buf = nullptr;
        if (is_arithmetic_v<T>) {
          slicer();
          buf = (T*)malloc(volume()*sizeof(T));
          memcpy(buf, shp.stride()*sizeof(T), this->buf,
              shp.stride()*sizeof(T), shp.width()*sizeof(T), shp.height());
        } else {
          buf = (T*)std::malloc(volume()*sizeof(T));
          std::uninitialized_copy(beginInternal(), endInternal(), buf);
        }

        /* memory order is important here: the new control block should not
         * become visible to other threads until after the buffer is set, if
         * the use of the control block as a lock is to be successful */
        this->buf = buf;
        this->isDiced = false;
        this->ctl.store(nullptr);
      }
      lock.unset();
    }
  }

  /**
   * Prepare for block-wise access. This allows asynchronous read or write by
   * a device.
   */
  void slicer() {
    isDiced = false;
  }

  /**
   * @copydoc slicer()
   */
  void slicer() const {
    const_cast<Array*>(this)->slicer();
  }

  /**
   * Prepare for element-wise access. If the array is currently prepared for
   * block-wise access, this requires synchronization with the device to
   * ensure that all asynchronous reads and writes have completed.
   */
  void dicer() {
    if (!isDiced) {
      if (is_arithmetic_v<T>) {
        wait();
      }
      isDiced = true;
    }
  }

  /**
   * @copydoc dicer()
   */
  void dicer() const {
    const_cast<Array*>(this)->dicer();
  }

  /**
   * Initialize allocated memory.
   *
   * @param args Constructor arguments.
   */
  template<class ... Args, std::enable_if_t<
      std::is_constructible_v<T,Args...>,int> = 0>
  void initialize(Args&&... args) {
    auto iter = beginInternal();
    auto last = endInternal();
    for (; iter != last; ++iter) {
      new (&*iter) T(std::forward<Args>(args)...);
    }
  }

  /**
   * Fill allocated memory with value.
   *
   * @param value The value.
   */
  void fill(const T& value) {
    memset(data(), shp.stride()*sizeof(T), value, shp.width()*sizeof(T),
        shp.height());
  }

  /**
   * Copy from another array.
   */
  template<class U, int E, std::enable_if_t<D == E &&
      std::is_convertible_v<U,T>,int> = 0>
  void uninitialized_copy(const Array<U,E>& o) {
    if (is_arithmetic_v<T> && std::is_same_v<T,U>) {
      slicer();
      memcpy(data(), shp.stride()*sizeof(T), o.data(),
          o.shp.stride()*sizeof(T), shp.width()*sizeof(T),
          shp.height());
    } else {
      auto n = std::min(size(), o.size());
      auto begin1 = o.beginInternal();
      auto end1 = begin1.operator+(n);
      auto begin2 = beginInternal();
      std::uninitialized_copy(begin1, end1, begin2);
    }
  }

  /**
   * Buffer containing elements.
   */
  T* buf;

  /**
   * Control block for sharing the buffer.
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
Array(const T& value) -> Array<T,0>;

template<class T>
Array(const ArrayShape<1>& shape, const T& value) -> Array<T,1>;

template<class T>
Array(const ArrayShape<2>& shape, const T& value) -> Array<T,2>;

template<class T>
Array(const ArrayShape<1>& shape, const Array<T,0>& value) -> Array<T,1>;

template<class T>
Array(const ArrayShape<2>& shape, const Array<T,0>& value) -> Array<T,2>;

}
