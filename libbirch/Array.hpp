/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/type.hpp"
#include "libbirch/Label.hpp"
#include "libbirch/Shape.hpp"
#include "libbirch/Buffer.hpp"
#include "libbirch/Iterator.hpp"
#include "libbirch/Eigen.hpp"
#include "libbirch/ExclusiveLock.hpp"

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
  using eigen_type = typename eigen_type<this_type>::type;
  using eigen_stride_type = typename eigen_stride_type<this_type>::type;

  /**
   * Constructor.
   *
   * @param shape Shape.
   */
  Array() :
      shape(),
      buffer(nullptr),
      offset(0),
      isView(false) {
    //
  }

  /**
   * Constructor.
   *
   * @param shape Shape.
   */
  template<IS_VALUE(T)>
  Array(const F& shape) :
      shape(shape),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
  }

  /**
   * Constructor.
   *
   * @param context Current context.
   * @param shape Shape.
   */
  template<IS_NOT_VALUE(T)>
  Array(Label* context, const F& shape) :
      shape(shape),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    initialize(context);
  }

  /**
   * Constructor.
   *
   * @param shape Shape.
   * @param values Values.
   */
  template<IS_VALUE(T)>
  Array(const F& shape, const std::initializer_list<T>& values) :
      shape(shape),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    std::uninitialized_copy(values.begin(), values.end(), as_const().begin());
  }

  /**
   * Constructor.
   *
   * @tparam ...Args Constructor parameter types.
   *
   * @param context Current context.
   * @param shape Shape.
   * @param values Values.
   */
  template<IS_NOT_VALUE(T), class U>
  Array(Label* context, const F& shape, const std::initializer_list<U>& values) :
      shape(shape),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    std::uninitialized_copy(values.begin(), values.end(), as_const().begin());
  }

  /**
   * Constructor.
   *
   * @tparam ...Args Constructor parameter types.
   *
   * @param context Current context.
   * @param shape Shape.
   * @param args Constructor arguments.
   */
  template<IS_NOT_VALUE(T), class... Args>
  Array(Label* context, const F& shape, Args ... args) :
      shape(shape),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    initialize(context, args...);
  }

  /**
   * Copy constructor.
   */
  Array(const Array<T,F>& o) :
      shape(o.shape),
      buffer(o.buffer),
      offset(o.offset),
      isView(o.isView) {
    if (buffer) {
      buffer->incUsage();
    }
  }

  /**
   * Copy constructor.
   */
  template<class U, class G>
  Array(const Array<U,G>& o) :
      shape(o.shape.compact()),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    uninitialized_copy(o);
  }

  /**
   * Copy constructor.
   */
  template<IS_NOT_VALUE(T), class U, class G>
  Array(Label* context, const Array<U,G>& o) :
      shape(o.shape.compact()),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    uninitialized_copy(context, o);
  }

  /**
   * Move constructor.
   */
  Array(Array<T,F>&& o) :
      shape(o.shape),
      buffer(o.buffer),
      offset(o.offset),
      isView(o.isView) {
    o.buffer = nullptr;
    o.offset = 0;
  }

  /**
   * Deep copy constructor.
   */
  template<IS_NOT_VALUE(T)>
  Array(Label* context, Label* label, const Array<T,F>& o) :
      shape(o.shape.compact()),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    uninitialized_copy(context, label, o);
  }

  /**
   * Destructor.
   */
  ~Array() {
    if (!isView) {
      release();
    }
  }

  /**
   * Copy assignment operator.
   */
  Array<T,F>& operator=(const Array<T,F>& o) {
    return assign(o);
  }

  /**
   * Move assignment operator.
   */
  Array<T,F>& operator=(Array<T,F>&& o) {
    return assign(std::move(o));
  }

  /**
   * Copy assignment. For a view the shapes of the two arrays must
   * conform, otherwise a resize is permitted.
   */
  template<IS_VALUE(T)>
  Array<T,F>& assign(const Array<T,F>& o) {
    if (isView) {
      libbirch_assert_msg_(o.shape.conforms(shape), "array sizes are different");
      copy(o);
    } else {
      lock();
      if (o.isView) {
        Array<T,F> tmp(o.shape, o);
        swap(tmp);
      } else {
        Array<T,F> tmp(o);
        swap(tmp);
      }
      unlock();
    }
    return *this;
  }

  /**
   * Copy assignment. For a view the shapes of the two arrays must
   * conform, otherwise a resize is permitted.
   */
  template<IS_NOT_VALUE(T)>
  Array<T,F>& assign(Label* context, const Array<T,F>& o) {
    if (isView) {
      libbirch_assert_msg_(o.shape.conforms(shape), "array sizes are different");
      copy(context, o);
    } else {
      lock();
      if (o.isView) {
        Array<T,F> tmp(o.shape, o);
        swap(tmp);
      } else {
        Array<T,F> tmp(o);
        swap(tmp);
      }
      unlock();
    }
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_NOT_VALUE(T)>
  Array<T,F>& assign(Label* context, Array<T,F>&& o) {
    if (isView) {
      libbirch_assert_msg_(o.shape.conforms(shape), "array sizes are different");
      copy(context, o);
    } else {
      lock();
      if (o.isView) {
        Array<T,F> tmp(o.shape, o);
        swap(tmp);
      } else {
        swap(o);
      }
      unlock();
    }
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_VALUE(T)>
  Array<T,F>& assign(Array<T,F> && o) {
    if (isView) {
      libbirch_assert_msg_(o.shape.conforms(shape), "array sizes are different");
      copy(o);
    } else {
      lock();
      if (o.isView) {
        Array<T,F> tmp(o.shape, o);
        swap(tmp);
      } else {
        swap(o);
      }
      unlock();
    }
    return *this;
  }

  /**
   * Return const reference to the array. This can be used to ensure that the
   * array is being accessed in a const context, to avoid unnecessary copying
   * of shared buffers.
   */
  const Array<T,F>& as_const() const {
    return *this;
  }

  /**
   * Raw pointer to underlying buffer.
   */
  T* buf() {
    duplicate();
    return buffer->buf() + offset;
  }
  T* buf() const {
    return buffer->buf() + offset;
  }

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
   * Number of rows. For a one-dimensional array, this is the length.
   */
  auto rows() const {
    return shape.length(0);
  }

  /**
   * Number of columns. For a one-dimensional array, this is 1.
   */
  auto cols() const {
    return F::count() == 1 ? 1 : shape.length(1);
  }

  /**
   * Stride between rows.
   */
  auto rowStride() const {
    return F::count() == 1 ? shape.volume() : shape.stride(0);
  }

  /**
   * Stride between columns.
   */
  auto colStride() const {
    return F::count() == 1 ? shape.stride(0) : shape.stride(1);
  }

  /**
   * @name Iterators
   */
  ///@{
  /**
   * Iterator pointing to the first element.
   *
   * Iterators are used to access the elements of an array sequentially.
   * Elements are visited in the order in which they are stored in memory;
   * the rightmost dimension is the fastest moving (for a matrix, this is
   * "row major" order).
   *
   * There is no `end()` function to retrieve an iterator to
   * one-past-the-last element. This is because a first/last pair must be
   * created atomically for thread safety. Instead use something like:
   *
   *     auto first = begin();
   *     auto last = first + size();
   */
  Iterator<T,F> begin() {
    return Iterator<T,F>(buf(), shape);
  }
  Iterator<T,F> begin() const {
    return Iterator<T,F>(buf(), shape);
  }
  ///@}

  /**
   * @name Slices
   */
  ///@{
  /**
   * Slice operator.
   *
   * @tparam V Slice type.
   *
   * @param slice Slice.
   *
   * @return The new array.
   */
  template<class V, std::enable_if_t<V::rangeCount() != 0,int> = 0>
  auto operator()(const V& slice) {
    duplicate();
    return Array<T,decltype(shape(slice))>(shape(slice),
        buffer, offset + shape.serial(slice));
  }
  template<class V, std::enable_if_t<V::rangeCount() != 0,int> = 0>
  auto operator()(const V& slice) const {
    return Array<T,decltype(shape(slice))>(shape(slice),
        buffer, offset + shape.serial(slice));
  }
  template<class V, std::enable_if_t<V::rangeCount() == 0,int> = 0>
  auto& operator()(const V& slice) {
    return *(buf() + shape.serial(slice));
  }
  template<class V, std::enable_if_t<V::rangeCount() == 0,int> = 0>
  const auto& operator()(const V& slice) const {
    return *(buf() + shape.serial(slice));
  }
  ///@}

  /**
   * @name Resize
   */
  /**
   * Shrink a one-dimensional array in-place.
   *
   * @tparam G Shape type.
   *
   * @param shape New shape.
   */
  template<class G>
  void shrink(const G& shape) {
    static_assert(F::count() == 1, "can only shrink one-dimensional arrays");
    static_assert(G::count() == 1, "can only shrink one-dimensional arrays");
    assert(!isView);
    assert(shape.size() < size());

    lock();
    if (isShared()) {
      Array<T,F> tmp(shape, *this);
      swap(tmp);
    } else {
      if (shape.volume() == 0) {
        release();
      } else {
        auto oldSize = Buffer<T>::size(volume());
        auto newSize = Buffer<T>::size(shape.volume());
        auto iter = as_const().begin();
        auto last = iter + size();
        for (iter += shape.size(); iter != last; ++iter) {
          iter->~T();
        }
        buffer = (Buffer<T>*)libbirch::reallocate(buffer,
            oldSize, buffer->tid, newSize);
      }
      this->shape = shape;
    }
    unlock();
  }

  /**
   * Enlarge a one-dimensional array in-place.
   *
   * @tparam G Shape type.
   *
   * @param shape New shape.
   * @param x Value to assign to new elements.
   */
  template<class G>
  void enlarge(const G& shape, const T& x) {
    static_assert(F::count() == 1, "can only enlarge one-dimensional arrays");
    static_assert(G::count() == 1, "can only enlarge one-dimensional arrays");
    assert(!isView);
    assert(shape.size() > size());

    lock();
    auto n = size();
    if (isShared() || !buffer) {
      Array<T,F> tmp(shape, *this);
      swap(tmp);
    } else {
      auto oldSize = Buffer<T>::size(volume());
      auto newSize = Buffer<T>::size(shape.volume());
      buffer = (Buffer<T>*)libbirch::reallocate(buffer, oldSize,
          buffer->tid, newSize);
      this->shape = shape;
    }
    auto iter = as_const().begin();
    std::uninitialized_fill(iter + n, iter + size(), x);
    unlock();
  }
  ///@}

  /**
   * @name Eigen integration
   */
  ///@{
  template<IS_VALUE(T)>
  operator eigen_type() const {
    return toEigen();
  }

  template<IS_VALUE(T)>
  auto toEigen() {
    return eigen_type(buf(), rows(), cols(),
        eigen_stride_type(rowStride(), colStride()));
  }
  template<IS_VALUE(T)>
  auto toEigen() const {
    return eigen_type(buf(), rows(), cols(),
        eigen_stride_type(rowStride(), colStride()));
  }

  /**
   * Construct from Eigen Matrix expression.
   */
  template<IS_VALUE(T), class EigenType, std::enable_if_t<is_eigen_compatible<this_type,EigenType>::value,int> = 0>
  Array(const Eigen::MatrixBase<EigenType>& o) :
      shape(o.rows(), o.cols()),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    toEigen() = o;
  }

  /**
   * Construct from Eigen DiagonalWrapper expression.
   */
  template<IS_VALUE(T), class EigenType, std::enable_if_t<is_diagonal_compatible<this_type,EigenType>::value,int> = 0>
  Array(const Eigen::DiagonalWrapper<EigenType>& o) :
      shape(o.rows(), o.cols()),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    toEigen() = o;
  }

  /**
   * Construct from Eigen TriangularWrapper expression.
   */
  template<IS_VALUE(T), class EigenType, unsigned Mode, std::enable_if_t<is_triangle_compatible<this_type,EigenType>::value,int> = 0>
  Array(const Eigen::TriangularView<EigenType,Mode>& o)  :
      shape(o.rows(), o.cols()),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    toEigen() = o;
  }
  ///@}

  template<IS_VALUE(T)>
  void freeze() {
    //
  }

  template<IS_NOT_VALUE(T)>
  void freeze() {
    auto iter = begin();
    auto last = iter + size();
    for (; iter != last; ++iter) {
      iter->freeze();
    }
  }

  template<IS_VALUE(T)>
  void thaw(Label* label) {
    //
  }

  template<IS_NOT_VALUE(T)>
  void thaw(Label* label) {
    auto iter = begin();
    auto last = iter + size();
    for (; iter != last; ++iter) {
      iter->thaw(label);
    }
  }

  template<IS_VALUE(T)>
  void finish() {
    //
  }

  template<IS_NOT_VALUE(T)>
  void finish() {
    auto iter = begin();
    auto last = iter + size();
    for (; iter != last; ++iter) {
      iter->finish();
    }
  }

private:
  /**
   * Constructor for forced copy.
   */
  template<class U, class G>
  Array(const F& shape, const Array<U,G>& o) :
      shape(shape.compact()),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    uninitialized_copy(o);
  }

  /**
   * Constructor for views.
   */
  Array(const F& shape, Buffer<T>* buffer, int64_t offset) :
      shape(shape),
      buffer(buffer),
      offset(offset),
      isView(true) {
    //
  }

  /**
   * Is the buffer shared with one or more other arrays?
   */
  bool isShared() const {
    return buffer && buffer->numUsage() > 1u;
  }

  /**
   * Swap with another array.
   */
  void swap(Array<T,F>& o) {
    std::swap(buffer, o.buffer);
    std::swap(shape, o.shape);
    std::swap(offset, o.offset);
  }

  /**
   * Allocate memory for array, leaving uninitialized.
   */
  void allocate() {
    assert(!buffer);
    auto size = Buffer<T>::size(volume());
    if (size > 0) {
      buffer = new (libbirch::allocate(size)) Buffer<T>();
      if (buffer) {
        buffer->incUsage();
      }
      offset = 0;
    }
  }

  /**
   * Duplicate underlying buffer by copy.
   */
  void duplicate() {
    if (!isView) {
      lock();
      if (isShared()) {
        Array<T,F> tmp(shape, *this);
        swap(tmp);
      }
      unlock();
    }
  }

  /**
   * Deallocate memory of array.
   */
  void release() {
    if (buffer && buffer->decUsage() == 0) {
      auto iter = as_const().begin();
      auto last = iter + size();
      for (; iter != last; ++iter) {
        iter->~T();
      }
      size_t size = Buffer<T>::size(volume());
      libbirch::deallocate(buffer, size, buffer->tid);
    }
    buffer = nullptr;
    offset = 0;
  }

  /**
   * Initialize allocated memory.
   *
   * @param args Constructor arguments.
   */
  template<IS_NOT_VALUE(T), class ... Args>
  void initialize(Label* context, Args ... args) {
    auto iter = as_const().begin();
    auto last = iter + size();
    for (; iter != last; ++iter) {
      new (&*iter) T(context, new typename T::value_type(context, args...));
    }
  }

  /**
   * Assign from another array.
   */
  template<IS_VALUE(T), class U, class G>
  void copy(const Array<U,G>& o) {
    assert(!isShared());
    auto n = std::min(size(), o.size());
    auto begin1 = o.begin();
    auto end1 = begin1 + n;
    auto begin2 = as_const().begin();
    auto end2 = begin2 + n;
    if (inside(begin1, end1, begin2)) {
      std::copy_backward(begin1, end1, end2);
    } else {
      std::copy(begin1, end1, begin2);
    }
  }

  /**
   * Assign from another array.
   */
  template<IS_NOT_VALUE(T), class U, class G>
  void copy(Label* context, const Array<U,G>& o) {
    assert(!isShared());
    auto n = std::min(size(), o.size());
    auto begin1 = o.begin();
    auto end1 = begin1 + n;
    auto begin2 = as_const().begin();
    auto end2 = begin2 + n;
    if (inside(begin1, end1, begin2)) {
      for (; end1 != begin1; --end1, --end2) {
        (end2 - 1)->assign(context, *(end1 - 1));
      }
    } else {
      for (; begin1 != end1; ++begin1, ++begin2) {
        begin2->assign(context, *begin1);
      }
    }
  }

  /**
   * Copy from another array.
   */
  template<class U, class G>
  void uninitialized_copy(const Array<U,G>& o) {
    assert(!isShared());
    auto n = std::min(size(), o.size());
    auto begin1 = o.begin();
    auto end1 = begin1 + n;
    auto begin2 = as_const().begin();
    std::uninitialized_copy(begin1, end1, begin2);
  }

  /**
   * Copy from another array.
   */
  template<IS_NOT_VALUE(T), class U, class G>
  void uninitialized_copy(Label* context, const Array<U,G>& o) {
    assert(!isShared());
    auto n = std::min(size(), o.size());
    auto begin1 = o.begin();
    auto end1 = begin1 + n;
    auto begin2 = as_const().begin();
    for (; begin1 != end1; ++begin1, ++begin2) {
      new (&*begin2) T(context, *begin1);
    }
  }

  /**
   * Deep copy from another array.
   */
  template<IS_NOT_VALUE(T), class U, class G>
  void uninitialized_copy(Label* context, Label* label, const Array<U,G>& o) {
    assert(!isShared());
    auto n = std::min(size(), o.size());
    auto begin1 = o.begin();
    auto end1 = begin1 + n;
    auto begin2 = as_const().begin();
    for (; begin1 != end1; ++begin1, ++begin2) {
      new (&*begin2) T(context, label, *begin1);
    }
  }

  /**
   * Release the lock.
   */
  void lock() {
    mutex.set();
  }

  /**
   * Release the lock.
   */
  void unlock() {
    mutex.unset();
  }

  /**
   * Shape.
   */
  F shape;

  /**
   * Buffer.
   */
  Buffer<T>* buffer;

  /**
   * Offset into the buffer. This should be zero when isView is false.
   */
  int64_t offset;

  /**
   * Is this a view of another array? A view has stricter assignment
   * semantics, as it cannot be resized or moved.
   */
  bool isView;

  /**
   * Lock used for copy on write.
   */
  ExclusiveLock mutex;
};

template<class T, class F>
struct is_value<Array<T,F>> {
  static const bool value = is_value<T>::value;
};

}
