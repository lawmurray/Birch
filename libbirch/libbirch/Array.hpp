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
      isElementWise(false),
      isBorrowed(false) {
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
      isElementWise(false),
      isBorrowed(false) {
    allocate();
    if (!std::is_trivial<T>::value) {
      synchronize();
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
      isElementWise(false),
      isBorrowed(false) {
    allocate();
    synchronize();
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
      isElementWise(false),
      isBorrowed(false) {
    allocate();
    synchronize();
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
      isElementWise(false),
      isBorrowed(false) {
    allocate();
    synchronize();
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
      isElementWise(false),
      isBorrowed(false) {
    allocate();
    synchronize();
    int64_t n = 0;
    for (auto iter = beginInternal(); iter != endInternal(); ++iter) {
      new (&*iter) T(l(n++));
    }
  }

  /**
   * View constructor.
   */
  Array(const shape_type& shape, T* buffer, bool isElementWise,
      bool isBorrowed) :
      buffer(buffer),
      control(nullptr),
      shape(shape),
      isView(true),
      isElementWise(isElementWise),
      isBorrowed(isBorrowed) {
    //
  }

  /**
   * Copy constructor.
   */
  Array(const Array& o) : Array() {
    shape = o.shape;
    if (!o.isView && std::is_trivial<T>::value) {
      std::tie(control, buffer, isBorrowed) = o.share();
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
    shape = o.shape;
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
      shape = o.shape;
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
    elementize();
    synchronize();
    return ArrayIterator<T,D>(buffer, shape);
  }

  /**
   * @copydoc begin()
   */
  ArrayIterator<T,D> begin() const {
    synchronize();
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
  template<class... Args>
  decltype(auto) operator()(Args&&... args) {
    elementize();
    synchronize();
    return shape.slice(data(), isElementWise, isBorrowed,
        std::forward<Args>(args)...);
  }

  /**
   * @copydoc operator()()
   */
  template<class... Args>
  decltype(auto) operator()(Args&&... args) const {
    synchronize();
    return shape.slice(data(), isElementWise, isBorrowed,
        std::forward<Args>(args)...);
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
   * Underlying buffer.
   */
  T* data() {
    elementize();
    borrow();
    return buffer;
  }

  /**
   * @copydoc data()
   */
  const T* data() const {
    borrow();
    return buffer;
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
      Array<T,D> tmp(s, x);
      swap(tmp);
    } else {
      elementize();
      synchronize();
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
      elementize();
      synchronize();
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
   * Iterator for use internally, avoiding elementize() or synchronize().
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
   * Iterator for use internally, avoiding elementize() or synchronize().
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
   * Flag underlying buffer as borrowed.
   */
  void borrow() {
    #if HAVE_NUMBIRCH_HPP
    isBorrowed = true;
    #endif
  }

  /**
   * @copydoc borrow()
   */
  void borrow() const {
    const_cast<Array*>(this)->borrow();
  }

  /**
   * Flag underlying buffer as no longer borrowed.
   */
  void unborrow() {
    #if HAVE_NUMBIRCH_HPP
    isBorrowed = false;
    #endif
  }

  /**
   * @copydoc unborrow()
   */
  void unborrow() const {
    const_cast<Array*>(this)->unborrow();
  }

  /**
   * Copy assignment. For a view the shapes of the two arrays must
   * conform, otherwise a resize is permitted.
   */
  void assign(const Array<T,D>& o) {
    if (isView) {
      assert(conforms(o) && "array sizes are different");
      copy(o);
    } else {
      Array<T,D> tmp(o);
      swap(tmp);
    }
  }

  /**
   * Swap with another array.
   */
  void swap(Array<T,D>& o) {
    assert(!isView);
    assert(!o.isView);
    std::swap(shape, o.shape);
    std::swap(buffer, o.buffer);
    std::swap(control, o.control);
    std::swap(isElementWise, o.isElementWise);
    std::swap(isBorrowed, o.isBorrowed);
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
    #if HAVE_NUMBIRCH_HPP
    if (std::is_trivial<T>::value) {
      buffer = (T*)numbirch::malloc(volume()*sizeof(T));
      borrow();
    } else
    #endif
    {
      buffer = (T*)std::malloc(volume()*sizeof(T));
    }
  }

  /**
   * Share the buffer of this array.
   * 
   * @return A triple giving pointers to the control block and buffer, and
   * the `isBorrowed` flag.
   */
  std::tuple<ArrayControl*,T*,bool> share() {
    assert(!isView);
    auto control = this->control;
    auto buffer = this->buffer;

    if (buffer) {
      if (isElementWise) {
        /* can't share, create a new buffer instead */
        #if HAVE_NUMBIRCH_HPP
        if (std::is_trivial<T>::value) {
          buffer = (T*)numbirch::malloc(size()*sizeof(T));
          numbirch::memcpy(buffer, shape.width()*sizeof(T), data(),
              shape.stride()*sizeof(T), shape.width()*sizeof(T),
              shape.height());
        } else
        #endif
        {
          buffer = (T*)std::malloc(size()*sizeof(T));
          std::uninitialized_copy(beginInternal(), endInternal(), buffer);
        }
      } else if (control) {
        control->incShared_();
      } else {
        lock.set();
        if (control) {  // another thread may have created in the meantime
          control->incShared_();
        } else {
          control = new ArrayControl(2);
          this->control = control;
        }
        lock.unset();
      }
    }
    return std::make_tuple(control, buffer, isBorrowed);
  }

  /**
   * @copydoc share()
   */
  std::tuple<ArrayControl*,T*,bool> share() const {
    return const_cast<Array*>(this)->share();
  }

  /**
   * Prepare for element-wise writes. If the buffer is currently shared, a new
   * buffer is created as a copy of the existing buffer.
   */
  void elementize() {
    assert(!isView);
    if (!isElementWise) {
      lock.set();
      if (!isElementWise) {
        if (control) {
          /* buffer may be shared, copy into new buffer to allow element-wise
           * write */
          T* buf = nullptr;
          #if HAVE_NUMBIRCH_HPP
          if (std::is_trivial<T>::value) {
            buf = (T*)numbirch::malloc(size()*sizeof(T));
            numbirch::memcpy(buf, shape.width()*sizeof(T), buffer,
                shape.stride()*sizeof(T), shape.width()*sizeof(T),
                shape.height());
            borrow();
          } else
          #endif
          {
            buf = (T*)std::malloc(size()*sizeof(T));
            std::uninitialized_copy(beginInternal(), endInternal(), buf);
          }
          release();
          compact();
          buffer = buf;
        }
        isElementWise = true;
      }
      lock.unset();
    }
    assert(!control);
  }

  /**
   * Prepare for element-wise reads. If the buffer is currently borrowed, this
   * requires synchronization with the device to ensure that all writes are
   * completed before the reads. 
   */
  void synchronize() const {
    #if HAVE_NUMBIRCH_HPP
    if (std::is_trivial<T>::value && isBorrowed) {
      numbirch::wait();
      unborrow();
    }
    #endif
  }

  /**
   * Release the buffer of this array, deallocating it if this is the last
   * reference to it.
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
    isBorrowed = false;
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
   * Assign from another array.
   */
  template<class U, int E, std::enable_if_t<D == E &&
      std::is_convertible<U,T>::value,int> = 0>
  void copy(const Array<U,E>& o) {
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
      auto end2 = begin2.operator+(n);
      if (begin1 <= begin2 && begin2 < end1) {
        std::copy_backward(begin1, end1, end2);
      } else {
        std::copy(begin1, end1, begin2);
      }
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
   * Are element-wise writes available for this array?
   */
  bool isElementWise;

  /**
   * Has the buffer been borrowed by a device? If so, the next access to the
   * elements of the buffer will require synchronization with that device.
   */
  bool isBorrowed;

  /**
   * Lock for operations requiring mutual exclusion.
   */
  Lock lock;
};
}
