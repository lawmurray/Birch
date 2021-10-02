/**
 * @file
 */
#pragma once

#include "numbirch/memory.hpp"
#include "numbirch/array/external.hpp"
#include "numbirch/array/ArrayControl.hpp"
#include "numbirch/array/ArrayShape.hpp"
#include "numbirch/array/ArrayIterator.hpp"
#include "numbirch/array/Atomic.hpp"
#include "numbirch/array/Lock.hpp"

namespace numbirch {
/**
 * @internal
 * 
 * Are all argument types integral? This is used to determine whether a slice
 * will return a view of an array, or a single element.
 * 
 * @ingroup array
 */
template<class... Args>
struct is_index {
  //
};

template<class Arg>
struct is_index<Arg> {
  static const bool value = std::is_integral<
      typename std::decay<Arg>::type>::value;
};

template<class Arg, class... Args>
struct is_index<Arg,Args...> {
  static const bool value = is_index<Arg>::value && is_index<Args...>::value;
};

/**
 * Multidimensional array with copy-on-write.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam D Number of dimensions.
 */
template<class T, int D>
class Array {
  template<class U, int E> friend class Array;
public:
  using value_type = T;
  using shape_type = ArrayShape<D>;

  /**
   * Constructor.
   */
  Array() :
      buf(nullptr),
      ctl(nullptr),
      shp(),
      isView(false),
      isElementWise(false) {
    assert(shp.volume() == 0);
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
      isElementWise(false) {
    allocate();
    if (!std::is_trivial<T>::value) {
      atomize();
      initialize();
    }
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
      isElementWise(false) {
    allocate();
    atomize();
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
      isElementWise(false) {
    allocate();
    atomize();
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
      isElementWise(false) {
    allocate();
    atomize();
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
   * @param l Lambda called to construct each element.
   * @param shape Shape.
   */
  template<class L>
  Array(const L& l, const shape_type& shape) :
      buf(nullptr),
      ctl(nullptr),
      shp(shape),
      isView(false),
      isElementWise(false) {
    allocate();
    atomize();
    int64_t n = 0;
    for (auto iter = beginInternal(); iter != endInternal(); ++iter) {
      new (&*iter) T(l(n++));
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
      isElementWise(false) {
    //
  }

  /**
   * Copy constructor.
   */
  Array(const Array& o) : Array() {
    shp = o.shp;
    if (!o.isView && std::is_trivial<T>::value) {
      ArrayControl* ctl;
      std::tie(ctl, buf, isElementWise) = o.share();
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
  template<class U, int E, std::enable_if_t<D == E &&
      std::is_convertible<U,T>::value,int> = 0>
  Array(const Array<U,E>& o) : Array() {
    shp = o.shp;
    compact();
    allocate();
    uninitialized_copy(o);
  }

  /**
   * Move constructor.
   */
  Array(Array&& o) : Array() {
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
   * Iterator to the first element.
   */
  ArrayIterator<T,D> begin() {
    own();
    atomize();
    return ArrayIterator<T,D>(buf, shp);
  }

  /**
   * @copydoc begin()
   */
  ArrayIterator<T,D> begin() const {
    atomize();
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
   * Slice.
   *
   * @tparam Args Slice argument types.
   * 
   * @param args Ranges or indices defining slice. An index should be of type
   * `int`, and a range of type `std::pair<int,int>` giving the first and last
   * indices of the range of elements to select.
   *
   * @return The resulting view or element.
   * 
   * @attention Currently ranges and indices for slices are 1-based rather
   * than 0-based, as Array is used directly from Birch, in which arrays use
   * 1-based indexing, rather than C++, in which arrays use 0-based indexing.
   */
  template<class... Args, std::enable_if_t<!is_index<Args...>::value,int> = 0>
  auto operator()(Args&&... args) {
    own();
    crystallize();
    return shp.slice(buf, std::forward<Args>(args)...);
  }

  /**
   * @copydoc operator()()
   */
  template<class... Args, std::enable_if_t<!is_index<Args...>::value,int> = 0>
  auto operator()(Args&&... args) const {
    crystallize();
    return shp.slice(buf, std::forward<Args>(args)...);
  }

  /**
   * @copydoc operator()()
   */
  template<class... Args, std::enable_if_t<is_index<Args...>::value,int> = 0>
  decltype(auto) operator()(Args&&... args) {
    own();
    atomize();
    return shp.slice(buf, std::forward<Args>(args)...);
  }

  /**
   * @copydoc operator()()
   */
  template<class... Args, std::enable_if_t<is_index<Args...>::value,int> = 0>
  auto operator()(Args&&... args) const {
    atomize();
    return shp.slice(buf, std::forward<Args>(args)...);
  }

  /**
   * Underlying buffer.
   */
  T* data() {
    own();
    crystallize();
    return buf;
  }

  /**
   * @copydoc data()
   */
  const T* data() const {
    crystallize();
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
      if (std::is_trivial<T>::value) {
        crystallize();
        buf = (T*)realloc((void*)buf, s.volume()*sizeof(T));
        atomize();
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
      atomize();
      std::destroy(buf + i, buf + i + len);
      std::memmove((void*)(buf + i), (void*)(buf + i + len),
          (n - len - i)*sizeof(T));
      if (std::is_trivial<T>::value) {
        crystallize();
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
      if (std::is_trivial<T>::value) {
        crystallize();
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
    std::swap(isElementWise, o.isElementWise);
  }

  /**
   * Does the shape of this array conform to that of another? Two shapes
   * conform if they have the same number of dimensions and lengths along
   * those dimensions. Strides may differ.
   */
  template<class U>
  bool conforms(const Array<U,D>& o) const {
    return shp.conforms(o.shp);
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
    assert(!isElementWise);

    if (std::is_trivial<T>::value) {
      crystallize();
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
        if (std::is_trivial<T>::value) {
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
    isElementWise = false;
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
    return std::make_tuple(ctl, buf, isElementWise);
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
        if (std::is_trivial<T>::value) {
          crystallize();
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
        this->isElementWise = false;
        this->ctl.store(nullptr);
      }
      lock.unset();
    }
  }

  /**
   * Prepare for block-wise access. This allows asynchronous read or write by
   * a device.
   */
  void crystallize() {
    isElementWise = false;
  }

  /**
   * @copydoc crystallize()
   */
  void crystallize() const {
    const_cast<Array*>(this)->crystallize();
  }

  /**
   * Prepare for element-wise access. If the array is currently prepared for
   * block-wise access, this requires synchronization with the device to
   * ensure that all asynchronous reads and writes have completed.
   */
  void atomize() {
    if (!isElementWise) {
      if (std::is_trivial<T>::value) {
        wait();
      }
      isElementWise = true;
    }
  }

  /**
   * @copydoc atomize()
   */
  void atomize() const {
    const_cast<Array*>(this)->atomize();
  }

  /**
   * Initialize allocated memory.
   *
   * @param args Constructor arguments.
   */
  template<class ... Args, std::enable_if_t<
      std::is_constructible<T,Args...>::value,int> = 0>
  void initialize(Args&&... args) {
    auto iter = beginInternal();
    auto last = endInternal();
    for (; iter != last; ++iter) {
      new (&*iter) T(std::forward<Args>(args)...);
    }
  }

  /**
   * Copy from another array.
   */
  template<class U, int E, std::enable_if_t<D == E &&
      std::is_convertible<U,T>::value,int> = 0>
  void uninitialized_copy(const Array<U,E>& o) {
    if (std::is_trivial<T>::value && std::is_same<T,U>::value) {
      crystallize();
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
  bool isElementWise;

  /**
   * Lock for operations requiring mutual exclusion.
   */
  Lock lock;
};

template<class T>
Array(const std::initializer_list<std::initializer_list<T>>&) -> Array<T,2>;

template<class T>
Array(const std::initializer_list<T>&) -> Array<T,1>;

template<class L>
Array(const L& l, const ArrayShape<1>& shape) -> Array<decltype(l(0)),1>;

template<class L>
Array(const L& l, const ArrayShape<2>& shape) -> Array<decltype(l(0)),2>;

template<class T>
Array(const ArrayShape<1>& shape, const T& value) -> Array<T,1>;

template<class T>
Array(const ArrayShape<2>& shape, const T& value) -> Array<T,2>;

}
