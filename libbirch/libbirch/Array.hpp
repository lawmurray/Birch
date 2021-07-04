/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/thread.hpp"
#include "libbirch/type.hpp"
#include "libbirch/Shape.hpp"
#include "libbirch/Iterator.hpp"
#include "libbirch/ArrayControl.hpp"
#include "libbirch/Lock.hpp"

namespace libbirch {
/**
 * Array.
 *
 * @tparam T Value type.
 * @tparam F Shape type.
 */
template<class T, class F>
class Array {
  template<class U, class G> friend class Array;
public:
  using this_type = Array<T,F>;
  using value_type = T;
  using shape_type = F;

  /**
   * Constructor.
   */
  Array() :
      shape(),
      buffer(nullptr),
      control(nullptr),
      isView(false),
      isElementWise(false) {
    assert(shape.volume() == 0);
  }

  /**
   * Constructor.
   *
   * @param shape Shape.
   */
  Array(const F& shape) :
      shape(shape),
      buffer(nullptr),
      control(nullptr),
      isView(false),
      isElementWise(false) {
    allocate();
    if (!std::is_trivially_copyable<T>::value) {
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
  Array(const F& shape, Args&&... args) :
      shape(shape),
      buffer(nullptr),
      control(nullptr),
      isView(false),
      isElementWise(false) {
    allocate();
    initialize(std::forward<Args>(args)...);
  }

  /**
   * Constructor.
   *
   * @param values Values.
   */
  template<class G = F, std::enable_if_t<G::count() == 1,int> = 0>
  Array(const std::initializer_list<T>& values) :
      shape(values.size()),
      buffer(nullptr),
      control(nullptr),
      isView(false),
      isElementWise(false) {
    allocate();
    std::uninitialized_copy(values.begin(), values.end(), beginInternal());
  }

  /**
   * Constructor.
   *
   * @param values Values.
   */
  template<class G = F, std::enable_if_t<G::count() == 2,int> = 0>
  Array(const std::initializer_list<std::initializer_list<T>>& values) :
      shape(values.size(), values.begin()->size()),
      buffer(nullptr),
      control(nullptr),
      isView(false),
      isElementWise(false) {
    allocate();
    auto ptr = buffer;
    for (auto row : values) {
      for (auto value : row) {
        new (ptr++) T(value);
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
  Array(const L& l, const F& shape) :
      shape(shape),
      buffer(nullptr),
      control(nullptr),
      isView(false),
      isElementWise(false) {
    allocate();
    int64_t n = 0;
    for (auto iter = beginInternal(); iter != endInternal(); ++iter) {
      new (&*iter) T(l(n++));
    }
  }

  /**
   * Copy constructor.
   */
  Array(const Array& o) :
      shape(),
      buffer(nullptr),
      control(nullptr),
      isView(false),
      isElementWise(false) {
    if (o.isView || !std::is_trivially_copyable<T>::value) {
      shape = o.shape.compact();
      allocate();
      uninitialized_copy(o);
    } else {
      shape = o.shape;
      std::tie(control, buffer) = o.share();
    }
  }

  /**
   * Generic copy constructor.
   */
  template<class U, class G, std::enable_if_t<F::count() == G::count() &&
      std::is_convertible<U,T>::value,int> = 0>
  Array(const Array<U,G>& o) :
      shape(o.shape.compact()),
      buffer(nullptr),
      control(nullptr),
      isView(false),
      isElementWise(false) {
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
      shape = o.shape.compact();
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
   * @name Elements
   */
  ///@{
  /**
   * Iterator pointing to the first element.
   *
   * Iterators are used to access the elements of an array sequentially.
   * Elements are visited in the order in which they are stored in memory;
   * the rightmost dimension is the fastest moving (for a matrix, this is
   * "row major" order).
   */
  Iterator<T,F> begin() {
    elementize();
    return Iterator<T,F>(buffer, shape);
  }

  /**
   * Iterator pointing to one past the last element.
   */
  Iterator<T,F> end() {
    return begin() + size();  // elementize() called by begin()
  }

  /**
   * Iterator pointing to the first element.
   */
  Iterator<T,F> begin() const {
    return Iterator<T,F>(buffer, shape);
  }

  /**
   * Iterator pointing to one past the last element.
   */
  Iterator<T,F> end() const {
    return begin() + size();
  }

  /**
   * Slice.
   *
   * @tparam V Slice type.
   *
   * @param slice Slice.
   *
   * @return The resulting view or element.
   */
  template<class V, std::enable_if_t<V::rangeCount() != 0,int> = 0>
  auto slice(const V& slice) {
    elementize();
    return Array<T,decltype(shape(slice))>(shape(slice), buffer +
        shape.serial(slice), true);
  }

  /**
   * @copydoc slice
   */
  template<class V, std::enable_if_t<V::rangeCount() == 0,int> = 0>
  auto& slice(const V& slice) {
    elementize();
    return *(buffer + shape.serial(slice));
  }

  /**
   * @copydoc slice
   */
  template<class V, std::enable_if_t<V::rangeCount() != 0,int> = 0>
  const auto slice(const V& slice) const {
    return Array<T,decltype(shape(slice))>(shape(slice), buffer +
        shape.serial(slice), false);
  }

  /**
   * @copydoc slice
   */
  template<class V, std::enable_if_t<V::rangeCount() == 0,int> = 0>
  auto slice(const V& slice) const {
    return *(buffer + shape.serial(slice));
  }

  /**
   * Slice.
   *
   * @tparam Args Slice argument types.
   *
   * @param args Slice arguments.
   *
   * @return The resulting view or element.
   */
  template<class... Args>
  decltype(auto) operator()(Args&&... args) {
    return slice(make_slice(std::forward<Args>(args)...));
  }

  /**
   * @copydoc operator()()
   */
  template<class... Args>
  decltype(auto) operator()(Args&&... args) const {
    return slice(make_slice(std::forward<Args>(args)...));
  }
  ///@}

  /**
   * @name Size
   */
  ///@{
  /**
   * Number of elements.
   */
  auto size() const {
    return shape.size();
  }

  /**
   * Number of elements allocated.
   */
  auto volume() const {
    return shape.volume();
  }

  /**
   * Number of rows. For a vector, this is the length.
   */
  auto rows() const {
    assert(1 <= F::count() && F::count() <= 2);
    return shape.length(0);
  }

  /**
   * Number of columns. For a vector, this is 1.
   */
  auto columns() const {
    assert(1 <= F::count() && F::count() <= 2);
    return F::count() == 1 ? 1 : shape.length(1);
  }

  /**
   * Stride between rows. For a vector this is the stride between elements,
   * for a matrix this is the stride between rows.
   */
  auto rowStride() const {
    assert(1 <= F::count() && F::count() <= 2);
    return shape.stride(0);
  }

  /**
   * Stride between columns. For a vector this is zero, for a matrix this is
   * the stride between elements in each row.
   */
  auto colStride() const {
    assert(1 <= F::count() && F::count() <= 2);
    return F::count() == 1 ? 0 : shape.stride(1);
  }
  ///@}

  /**
   * @name Resize
   */
  ///@{
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
  void insert(const int64_t i, const T& x) {
    static_assert(F::count() == 1, "can only enlarge one-dimensional arrays");
    assert(!isView);

    elementize();
    auto n = size();
    auto s = F(n + 1);
    if (!buffer) {
      Array<T,F> tmp(s, x);
      swap(tmp);
    } else {
      buffer = (T*)std::realloc((void*)buffer, s.volume()*sizeof(T));
      std::memmove((void*)(buffer + i + 1), (void*)(buffer + i), (n - i)*sizeof(T));
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
  void erase(const int64_t i, const int64_t len = 1) {
    static_assert(F::count() == 1, "can only shrink one-dimensional arrays");
    assert(!isView);
    assert(len > 0);
    assert(size() >= len);

    elementize();
    auto n = size();
    auto s = F(n - len);
    if (s.size() == 0) {
      release();
    } else {
      for (int j = i; j < i + len; ++j) {
        buffer[j].~T();
      }
      std::memmove((void*)(buffer + i), (void*)(buffer + i + len), (n - len - i)*sizeof(T));
      buffer = (T*)std::realloc((void*)buffer, s.volume()*sizeof(T));
    }
    shape = s;
  }
  ///@}

  /**
   * @name Low-level
   */
  ///@{
  /**
   * Direct access to the underlying buffer.
   */
  T* data() {
    elementize();
    return buffer;
  }

  /**
   * Direct access to the underlying buffer.
   */
  const T* data() const {
    return buffer;
  }
  ///@}

private:
  /**
   * Constructor for views.
   */
  Array(const F& shape, T* buffer, bool isElementWise) :
      shape(shape),
      buffer(buffer),
      control(nullptr),
      isView(true),
      isElementWise(isElementWise) {
    //
  }

  /**
   * Iterator for use internally, avoiding elementize().
   */
  Iterator<T,F> beginInternal() {
    return Iterator<T,F>(buffer, shape);
  }

  /**
   * Iterator for use internally, avoiding elementize().
   */
  Iterator<T,F> endInternal() {
    return beginInternal() + size();
  }

  /**
   * Iterator for use internally, avoiding elementize().
   */
  Iterator<T,F> beginInternal() const {
    return Iterator<T,F>(buffer, shape);
  }

  /**
   * Iterator for use internally, avoiding elementize().
   */
  Iterator<T,F> endInternal() const {
    return beginInternal() + size();
  }

  /**
   * Copy assignment. For a view the shapes of the two arrays must
   * conform, otherwise a resize is permitted.
   */
  void assign(const Array<T,F>& o) {
    if (isView) {
      assert(o.shape.conforms(shape) && "array sizes are different");
      copy(o);
    } else {
      Array<T,F> tmp(o);
      swap(tmp);
    }
  }

  /**
   * Swap with another array.
   */
  void swap(Array<T,F>& o) {
    assert(!isView);
    assert(!o.isView);
    std::swap(shape, o.shape);
    std::swap(buffer, o.buffer);
    std::swap(control, o.control);
    std::swap(isElementWise, o.isElementWise);
  }

  /**
   * Allocate memory for this, leaving uninitialized.
   */
  void allocate() {
    assert(!buffer);
    buffer = (T*)std::malloc(volume()*sizeof(T));
  }

  /**
   * Share the buffer of this array.
   * 
   * @return A pair giving pointers to the control block and buffer.
   */
  std::pair<ArrayControl*,T*> share() {
    assert(!isView);
    if (buffer) {
      if (control) {
        control->incShared_();
      } else if (isElementWise) {
        /* can't share, create a new buffer instead */
        ArrayControl* control = nullptr;
        T* buffer = (T*)std::malloc(size()*sizeof(T));
        std::uninitialized_copy(beginInternal(), endInternal(), buffer);
        return std::make_pair(control, buffer);
      } else {
        lock.set();
        if (!control) {
          control = new ArrayControl(2);
        } else {
          control->incShared_();
        }
        lock.unset();
      }
    }
    return std::make_pair(control, buffer);
  }

  /**
   * Share the buffer of this array.
   * 
   * @return A pair giving pointers to the control block and buffer.
   */
  std::pair<ArrayControl*,T*> share() const {
    return const_cast<Array<T,F>*>(this)->share();
  }

  /**
   * Prepare this array for element-wise writes. If the buffer is currently
   * shared, a new buffer is created as a copy of the existing buffer.
   */
  void elementize() {
    assert(!isView);
    if (!isElementWise) {
      lock.set();
      if (!isElementWise) {
        if (control) {
          /* buffer may be shared, copy into new buffer to allow element-wise
           * write */
          T* buffer = (T*)std::malloc(size()*sizeof(T));
          std::uninitialized_copy(beginInternal(), endInternal(), buffer);
          release();
          this->buffer = buffer;
        }
        isElementWise = true;
      }
      lock.unset();
    }
    assert(!control);
  }

  /**
   * Release the buffer of this array, deallocating it if this is the last
   * reference to it.
   */
  void release() {
    if (!isView && (!control || control->decShared_() == 0)) {
      std::destroy(beginInternal(), endInternal());
      std::free(buffer);
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
  template<class ... Args, std::enable_if_t<std::is_constructible<T,Args...>::value,int> = 0>
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
  template<class U>
  void copy(const U& o) {
    auto n = std::min(size(), o.size());
    auto begin1 = o.beginInternal();
    auto end1 = begin1 + n;
    auto begin2 = beginInternal();
    auto end2 = begin2 + n;
    if (inside(begin1, end1, begin2)) {
      std::copy_backward(begin1, end1, end2);
    } else {
      std::copy(begin1, end1, begin2);
    }
  }

  /**
   * Copy from another array.
   */
  template<class U>
  void uninitialized_copy(const U& o) {
    auto n = std::min(size(), o.size());
    std::uninitialized_copy(o.beginInternal(), o.beginInternal() + n,
        beginInternal());
  }

  /**
   * Shape.
   */
  F shape;

  /**
   * Buffer containing elements.
   */
  T* buffer;

  /**
   * Control block for sharing buffer.
   */
  ArrayControl* control;

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
   * Lock for operations requiring mutual exclusion.
   */
  Lock lock;
};

/**
 * Default array for `D` dimensions.
 */
template<class T, int D>
using DefaultArray = Array<T,typename DefaultShape<D>::type>;

}
