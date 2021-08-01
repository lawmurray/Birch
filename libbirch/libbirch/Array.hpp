/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/ArrayControl.hpp"
#include "libbirch/ArrayShape.hpp"
#include "libbirch/ArrayIterator.hpp"
#include "libbirch/Lock.hpp"

namespace libbirch {
/**
 * Are all argument types integral? This is used to determine whether a slice
 * will return a view of an array, or a single element.
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
 * Array.
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
      buffer(nullptr),
      control(nullptr),
      shape(),
      isView(false),
      isElementWise(false) {
    assert(shape.volume() == 0);
  }

  /**
   * Constructor.
   *
   * @param shape Shape.
   */
  Array(const shape_type& shape) :
      buffer(nullptr),
      control(nullptr),
      shape(shape),
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
      buffer(nullptr),
      control(nullptr),
      shape(shape),
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
      buffer(nullptr),
      control(nullptr),
      shape(values.size()),
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
      buffer(nullptr),
      control(nullptr),
      shape(values.size(), values.begin()->size()),
      isView(false),
      isElementWise(false) {
    allocate();
    atomize();
    int64_t t = 0;
    for (auto row : values) {
      for (auto x : row) {
        new (buffer + shape.transpose(t++)) T(x);
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
      buffer(nullptr),
      control(nullptr),
      shape(shape),
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
  Array(T* buffer, const shape_type& shape) :
      buffer(buffer),
      control(nullptr),
      shape(shape),
      isView(true),
      isElementWise(false) {
    //
  }

  /**
   * Copy constructor.
   */
  Array(const Array& o) : Array() {
    shape = o.shape;
    if (!o.isView && std::is_trivial<T>::value) {
      std::tie(control, buffer, isElementWise) = o.share();
    } else {
      compact();
      allocate();
      crystallize();
      uninitialized_copy(o);
    }
  }

  /**
   * Generic copy constructor.
   */
  template<class U, int E, std::enable_if_t<D == E &&
      std::is_convertible<U,T>::value,int> = 0>
  Array(const Array<U,E>& o) : Array() {
    shape = o.shape;
    compact();
    allocate();
    crystallize();
    uninitialized_copy(o);
  }

  /**
   * Move constructor.
   */
  Array(Array&& o) : Array() {
    if (!o.isView) {
      swap(o);
    } else {
      shape = o.shape;
      compact();
      allocate();
      crystallize();
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
    return ArrayIterator<T,D>(buffer, shape);
  }

  /**
   * @copydoc begin()
   */
  ArrayIterator<T,D> begin() const {
    atomize();
    return ArrayIterator<T,D>(buffer, shape);
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
    return shape.slice(buffer, std::forward<Args>(args)...);
  }

  /**
   * @copydoc operator()()
   */
  template<class... Args, std::enable_if_t<!is_index<Args...>::value,int> = 0>
  auto operator()(Args&&... args) const {
    crystallize();
    return shape.slice(buffer, std::forward<Args>(args)...);
  }

  /**
   * @copydoc operator()()
   */
  template<class... Args, std::enable_if_t<is_index<Args...>::value,int> = 0>
  decltype(auto) operator()(Args&&... args) {
    own();
    atomize();
    return shape.slice(buffer, std::forward<Args>(args)...);
  }

  /**
   * @copydoc operator()()
   */
  template<class... Args, std::enable_if_t<is_index<Args...>::value,int> = 0>
  auto operator()(Args&&... args) const {
    atomize();
    return shape.slice(buffer, std::forward<Args>(args)...);
  }

  /**
   * Underlying buffer.
   */
  T* data() {
    own();
    crystallize();
    return buffer;
  }

  /**
   * @copydoc data()
   */
  const T* data() const {
    crystallize();
    return buffer;
  }

  /**
   * Number of elements.
   */
  int64_t size() const {
    return shape.size();
  }

  /**
   * Number of elements allocated.
   */
  int64_t volume() const {
    return shape.volume();
  }

  /**
   * Number of rows. For a vector, this is the length.
   */
  int rows() const {
    return shape.rows();
  }

  /**
   * Number of columns. For a vector, this is 1.
   */
  int columns() const {
    return shape.columns();
  }

  /**
   * Stride. For a vector this is the stride between elements, for a matrix
   * this is the stride between columns.
   */
  int stride() const {
    return shape.stride();
  }

  /**
   * For a one-dimensional array, push an element onto the end. This increases
   * the array size by one.
   *
   * @param x Value.
   */
  void push(const T& x) {
    insert(size(), x);
  }

  /**
   * For a one-dimensional array, insert an element at a given position. This
   * increases the array size by one.
   *
   * @param i Position.
   * @param x Value.
   */
  void insert(const int i, const T& x) {
    static_assert(D == 1, "insert() supports only one-dimensional arrays");
    assert(!isView);

    auto n = size();
    ArrayShape<1> s(n + 1);
    if (!buffer) {
      Array tmp(s, x);
      swap(tmp);
    } else {
      own();
      atomize();
      #if HAVE_NUMBIRCH_HPP
      if (std::is_trivial<T>::value) {
        buffer = (T*)numbirch::realloc((void*)buffer, s.volume()*sizeof(T));
      } else
      #endif
      {
        buffer = (T*)std::realloc((void*)buffer, s.volume()*sizeof(T));
      }
      std::memmove((void*)(buffer + i + 1), (void*)(buffer + i),
          (n - i)*sizeof(T));
      new (buffer + i) T(x);
      shape = s;
    }
  }

  /**
   * For a one-dimensional array, erase elements from a given position. This
   * decreases the array size by the number of elements.
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
      std::destroy(buffer + i, buffer + i + len);
      std::memmove((void*)(buffer + i), (void*)(buffer + i + len),
          (n - len - i)*sizeof(T));
      #if HAVE_NUMBIRCH_HPP
      if (std::is_trivial<T>::value) {
        buffer = (T*)numbirch::realloc((void*)buffer, s.volume()*sizeof(T));
      } else
      #endif
      {
        buffer = (T*)std::realloc((void*)buffer, s.volume()*sizeof(T));
      }
    }
    shape = s;
  }

private:
  /**
   * Iterator for use internally.
   */
  ArrayIterator<T,D> beginInternal() {
    return ArrayIterator<T,D>(buffer, shape);
  }

  /**
   * @copydoc beginInternal()
   */
  ArrayIterator<T,D> beginInternal() const {
    return ArrayIterator<T,D>(buffer, shape);
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
   * Copy assignment. For a view the shapes of the two arrays must
   * conform, otherwise a resize is permitted.
   */
  void assign(const Array& o) {
    if (isView) {
      assert(conforms(o) && "array sizes are different");
      #if HAVE_NUMBIRCH_HPP
      if (std::is_trivial<T>::value) {
        numbirch::memcpy(data(), shape.stride()*sizeof(T), o.data(),
            o.shape.stride()*sizeof(T), shape.width()*sizeof(T),
            shape.height());
      } else
      #endif
      {
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
    std::swap(shape, o.shape);
    std::swap(buffer, o.buffer);
    std::swap(control, o.control);
    std::swap(isElementWise, o.isElementWise);
  }

  /**
   * Does the shape of this array conform to that of another? Two shapes
   * conform if they have the same number of dimensions and lengths along
   * those dimensions. Strides may differ.
   */
  template<class U>
  bool conforms(const Array<U,D>& o) const {
    return shape.conforms(o.shape);
  }

  /**
   * Compact the array by reducing the volume to match the size. This is only
   * possible prior to allocation.
   */
  void compact() {
    assert(!buffer);
    shape.compact();
  }

  /**
   * Allocate memory for this, leaving uninitialized.
   */
  void allocate() {
    assert(!buffer);
    assert(!control);
    assert(!isElementWise);

    #if HAVE_NUMBIRCH_HPP
    if (std::is_trivial<T>::value) {
      buffer = (T*)numbirch::malloc(volume()*sizeof(T));
    } else
    #endif
    {
      buffer = (T*)std::malloc(volume()*sizeof(T));
    }
  }

  /**
   * Share the buffer.
   * 
   * @return A pair giving pointers to the control block and buffer.
   */
  std::tuple<ArrayControl*,T*,bool> share() {
    assert(!isView);

    if (buffer) {
      if (control) {
        control->incShared_();
      } else {
        lock.set();
        if (control) {  // another thread may have created in the meantime
          control->incShared_();
        } else {
          control = new ArrayControl(2);
        }
        lock.unset();
      }
    }
    return std::make_tuple(control, buffer, isElementWise);
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
    if (control) {
      assert(!isView);
      lock.set();
      if (control) {  // another thread may have cleared in the meantime
        if (control->numShared_() > 1) {
          T* buf = nullptr;
          #if HAVE_NUMBIRCH_HPP
          if (std::is_trivial<T>::value) {
            buf = (T*)numbirch::malloc(size()*sizeof(T));
            numbirch::memcpy(buf, shape.width()*sizeof(T), buffer,
                shape.stride()*sizeof(T), shape.width()*sizeof(T),
                shape.height());
            crystallize();
          } else
          #endif
          {
            buf = (T*)std::malloc(size()*sizeof(T));
            std::uninitialized_copy(beginInternal(), endInternal(), buf);
          }
          release();
          compact();
          this->buffer = buf;
        } else {
          /* last reference */
          delete control;
          control = nullptr;
        }
      }
      lock.unset();
    }
    assert(!control);
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
      #if HAVE_NUMBIRCH_HPP
      if (std::is_trivial<T>::value) {
        numbirch::wait();
      }
      #endif
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
   * Release the buffer, deallocating if this is the last reference to it.
   */
  void release() {
    if (!isView && (!control || control->decShared_() == 0)) {
      #if HAVE_NUMBIRCH_HPP
      if (std::is_trivial<T>::value) {
        numbirch::free((void*)buffer);
      } else
      #endif
      {
        std::destroy(beginInternal(), endInternal());
        std::free((void*)buffer);
      }
      delete control;
    }
    buffer = nullptr;
    control = nullptr;
    isView = false;
    isElementWise = false;
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
    #if HAVE_NUMBIRCH_HPP
    if (std::is_trivial<T>::value && std::is_same<T,U>::value) {
      numbirch::memcpy(data(), shape.stride()*sizeof(T), o.data(),
          o.shape.stride()*sizeof(T), shape.width()*sizeof(T),
          shape.height());
    } else
    #endif
    {
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
  T* buffer;

  /**
   * Control block for sharing buffer.
   */
  ArrayControl* control;

  /**
   * Shape.
   */
  ArrayShape<D> shape;

  /**
   * Is this a view of another array? A view has stricter assignment
   * semantics, as it cannot be resized or moved.
   */
  bool isView;

  /**
   * Is the array prepared for element-wise access? If false, the array is
   * prepared for block-wise access.
   */
  bool isElementWise;

  /**
   * Lock for operations requiring mutual exclusion.
   */
  Lock lock;
};
}
