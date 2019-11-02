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
#include "libbirch/ReaderWriterLock.hpp"

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
   */
  Array() :
      shape(),
      buffer(nullptr),
      offset(0),
      isView(false) {
    assert(shape.volume() == 0);
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
    std::uninitialized_copy(values.begin(), values.end(), begin());
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
  Array(Label* context, const F& shape,
      const std::initializer_list<U>& values) :
      shape(shape),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    std::uninitialized_copy(values.begin(), values.end(), begin());
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
    if (!isView && buffer) {
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
    o.pin();
    uninitialized_copy(o);
    o.unpin();
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
    o.pin();
    uninitialized_copy(context, o);
    o.unpin();
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
    o.pin();
    uninitialized_copy(context, label, o);
    o.unpin();
  }

  /**
   * Destructor.
   */
  ~Array() {
    release();
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
    o.pin();
    if (isView) {
      libbirch_assert_msg_(o.shape.conforms(shape), "array sizes are different");
      pin();
      copy(o);
      unpin();
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
    o.unpin();
    return *this;
  }

  /**
   * Copy assignment. For a view the shapes of the two arrays must
   * conform, otherwise a resize is permitted.
   */
  template<IS_NOT_VALUE(T)>
  Array<T,F>& assign(Label* context, const Array<T,F>& o) {
    o.pin();
    if (isView) {
      libbirch_assert_msg_(o.shape.conforms(shape), "array sizes are different");
      pin();
      copy(context, o);
      unpin();
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
    o.unpin();
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_NOT_VALUE(T)>
  Array<T,F>& assign(Label* context, Array<T,F>&& o) {
    if (isView) {
      libbirch_assert_msg_(o.shape.conforms(shape), "array sizes are different");
      pin();
      copy(context, o);
      unpin();
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
      pin();
      copy(o);
      unpin();
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
   * @name Element access, caller not responsible for thread safety
   */
  ///@{
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
  auto get(const V& slice) const {
    pin();
    Array<T,decltype(shape(slice))> view(shape(slice), buffer, offset +
        shape.serial(slice));
    unpin();
    return view;
  }

  template<class U, class G, class V, IS_VALUE(T),
      std::enable_if_t<V::rangeCount() != 0,int> = 0>
  auto set(const V& slice, const Array<U,G>& value) {
    pinWrite();
    Array<T,decltype(shape(slice))> view(shape(slice), buffer, offset +
        shape.serial(slice));
    view.copy(value);
    unpin();
    return view;
  }

  template<class U, class G, class V, IS_NOT_VALUE(T),
      std::enable_if_t<V::rangeCount() != 0,int> = 0>
  auto set(const V& slice, Label* context, const Array<U,G>& value) {
    pinWrite();
    Array<T,decltype(shape(slice))> view(shape(slice), buffer, offset +
        shape.serial(slice));
    view.copy(context, value);
    unpin();
    return view;
  }

  template<class V, std::enable_if_t<V::rangeCount() == 0,int> = 0>
  T get(const V& slice) const {
    pin();
    auto view = *(buf() + shape.serial(slice));
    unpin();
    return view;
  }

  template<class V, IS_VALUE(T),
      std::enable_if_t<V::rangeCount() == 0,int> = 0>
  T& set(const V& slice, const T& value) {
    pinWrite();
    auto& view = *(buf() + shape.serial(slice));
    view = value;
    unpin();
    return view;
  }

  template<class V, IS_NOT_VALUE(T),
      std::enable_if_t<V::rangeCount() == 0,int> = 0>
  T& set(const V& slice, Label* context, const T& value) {
    pinWrite();
    T& view = *(buf() + shape.serial(slice));
    view.assign(context, value);
    unpin();
    return view;
  }

  template<class V, IS_VALUE(T),
      std::enable_if_t<V::rangeCount() == 0,int> = 0>
  T& set(const V& slice, T&& value) {
    pinWrite();
    auto& view = *(buf() + shape.serial(slice));
    view = std::move(value);
    unpin();
    return view;
  }

  template<class V, IS_NOT_VALUE(T),
      std::enable_if_t<V::rangeCount() == 0,int> = 0>
  T& set(const V& slice, Label* context, T&& value) {
    pinWrite();
    T& view = *(buf() + shape.serial(slice));
    view.assign(context, std::move(value));
    unpin();
    return view;
  }
///@}

  /**
   * @name Element access, caller responsible for thread safety
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
    assert(!isShared());
    return Iterator<T,F>(buf(), shape);
  }
  Iterator<T,F> begin() const {
    return Iterator<T,F>(buf(), shape);
  }
  Iterator<T,F> end() {
    assert(!isShared());
    return begin() + size();
  }
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
  auto operator()(const V& slice) {
    assert(!isShared());
    return Array<T,decltype(shape(slice))>(shape(slice),
        buffer, offset + shape.serial(slice));
  }
  template<class V, std::enable_if_t<V::rangeCount() != 0,int> = 0>
  auto operator()(const V& slice) const {
    return Array<T,decltype(shape(slice))>(shape(slice),
        buffer, offset + shape.serial(slice));
  }
  template<class V, std::enable_if_t<V::rangeCount() == 0,int> = 0>
  value_type& operator()(const V& slice) {
    assert(!isShared());
    return *(buf() + shape.serial(slice));
  }
  template<class V, std::enable_if_t<V::rangeCount() == 0,int> = 0>
  value_type operator()(const V& slice) const {
    return *(buf() + shape.serial(slice));
  }

  /**
   * Pin the buffer. This prevents substitution of the buffer by
   * copy-on-write operations until unpinned.
   */
  void pin() const {
    const_cast<Array*>(this)->bufferLock.read();
  }

  /**
   * As pin(), but furthermore ensures that the buffer is not shared, and
   * thus its contents eligible for writing. If shared, a copy is performed.
   * This is used to perform copy-on-write (if necessary) before writing the
   * contents of the buffer.
   */
  void pinWrite() {
    assert(!isView);
    bufferLock.write();
    if (isShared()) {
      Array<T,F> tmp(shape, *this);
      swap(tmp);
    }
    bufferLock.downgrade();  // downgrade write lock to read lock
  }

  /**
   * Unpin the buffer.
   */
  void unpin() const {
    const_cast<Array*>(this)->bufferLock.unread();
  }

  /**
   * Lock the buffer. This is used before substitution of the buffer by a
   * copy-on-write operation.
   */
  void lock() {
    bufferLock.write();
  }

  /**
   * Unlock the buffer.
   */
  void unlock() {
    bufferLock.unwrite();
  }
  ///@}

  /**
   * @name Thread-safe resize
   */
  ///@{
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
    assert(shape.size() <= size());

    lock();
    auto oldSize = size();
    auto newSize = shape.size();
    if (newSize < oldSize) {
      if (isShared()) {
        Array<T,F> tmp(shape, *this);
        swap(tmp);
      } else {
        if (newSize == 0) {
          release();
        } else {
          auto iter = begin();
          auto last = end();
          for (iter += shape.size(); iter != last; ++iter) {
            iter->~T();
          }
          // ^ C++17 use std::destroy
          auto oldBytes = Buffer<T>::size(volume());
          auto newBytes = Buffer<T>::size(shape.volume());
          buffer = (Buffer<T>*)libbirch::reallocate(buffer, oldBytes,
              buffer->tid, newBytes);
        }
        this->shape = shape;
      }
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
    assert(shape.size() >= size());

    lock();
    auto oldSize = size();
    auto newSize = shape.size();
    if (newSize > oldSize) {
      if (!buffer || isShared()) {
        Array<T,F> tmp(shape, *this);
        swap(tmp);
      } else {
        auto oldBytes = Buffer<T>::size(volume());
        auto newBytes = Buffer<T>::size(shape.volume());
        buffer = (Buffer<T>*)libbirch::reallocate(buffer, oldBytes,
            buffer->tid, newBytes);
        this->shape = shape;
      }
      std::uninitialized_fill(begin() + oldSize, begin() + newSize, x);
    }
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
    return eigen_type(buf(), rows(), cols(), eigen_stride_type(rowStride(),
        colStride()));
  }

  template<IS_VALUE(T)>
  auto toEigen() const {
    return eigen_type(buf(), rows(), cols(), eigen_stride_type(rowStride(),
        colStride()));
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

  /**
   * Number of rows. For a vectoor, this is the length.
   */
  auto rows() const {
    assert(1 <= F::count() && F::count() <= 2);
    return shape.length(0);
  }

  /**
   * Number of columns. For a vector, this is 1.
   */
  auto cols() const {
    assert(1 <= F::count() && F::count() <= 2);
    return F::count() == 1 ? 1 : shape.length(1);
  }

  /**
   * Stride between rows.
   */
  auto rowStride() const {
    assert(1 <= F::count() && F::count() <= 2);
    return F::count() == 1 ? shape.volume() : shape.stride(0);
  }

  /**
   * Stride between columns.
   */
  auto colStride() const {
    assert(1 <= F::count() && F::count() <= 2);
    return F::count() == 1 ? shape.stride(0) : shape.stride(1);
  }
  ///@}

  template<IS_VALUE(T)>
  void freeze() {
    //
  }

  template<IS_NOT_VALUE(T)>
  void freeze() {
    pin();
    auto iter = begin();
    auto last = end();
    for (; iter != last; ++iter) {
      iter->freeze();
    }
    unpin();
  }

  template<IS_VALUE(T)>
  void thaw(Label* label) {
    //
  }

  template<IS_NOT_VALUE(T)>
  void thaw(Label* label) {
    pin();
    auto iter = begin();
    auto last = end();
    for (; iter != last; ++iter) {
      iter->thaw(label);
    }
    unpin();
  }

  template<IS_VALUE(T)>
  void finish() {
    //
  }

  template<IS_NOT_VALUE(T)>
  void finish() {
    pin();
    auto iter = begin();
    auto last = end();
    for (; iter != last; ++iter) {
      iter->finish();
    }
    unpin();
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
   * Raw pointer to underlying buffer.
   */
  T* buf() const {
    return buffer->buf() + offset;
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
    assert(!isView);
    assert(!o.isView);
    std::swap(buffer, o.buffer);
    std::swap(shape, o.shape);
    std::swap(offset, o.offset);
  }

  /**
   * Allocate memory for array, leaving uninitialized.
   */
  void allocate() {
    assert(!buffer);
    auto bytes = Buffer<T>::size(volume());
    if (bytes > 0u) {
      buffer = new (libbirch::allocate(bytes)) Buffer<T>();
      offset = 0;
    }
  }

  /**
   * Deallocate memory of array.
   */
  void release() {
    if (!isView && buffer && buffer->decUsage() == 0) {
      auto iter = begin();
      auto last = end();
      for (; iter != last; ++iter) {
        iter->~T();
      }
      // ^ C++17 use std::destroy
      size_t bytes = Buffer<T>::size(volume());
      libbirch::deallocate(buffer, bytes, buffer->tid);
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
    auto iter = begin();
    auto last = end();
    for (; iter != last; ++iter) {
      new (&*iter) T(context, new typename T::value_type(context, args...));
    }
  }

  /**
   * Assign from another array.
   */
  template<IS_VALUE(T), class U, class G>
  void copy(const Array<U,G>& o) {
    auto n = std::min(size(), o.size());
    auto begin1 = o.begin();
    auto end1 = begin1 + n;
    auto begin2 = begin();
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
    auto n = std::min(size(), o.size());
    auto begin1 = o.begin();
    auto end1 = begin1 + n;
    auto begin2 = begin();
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
    auto begin2 = begin();
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
    auto begin2 = begin();
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
    auto begin2 = begin();
    for (; begin1 != end1; ++begin1, ++begin2) {
      new (&*begin2) T(context, label, *begin1);
    }
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
   * Lock used for copy-on-write. Read use is obtained when the current
   * buffer must be preserved for either read or write operations. Write use
   * is obtained to substitute the current buffer with another.
   */
  ReaderWriterLock bufferLock;
};

template<class T, class F>
struct is_value<Array<T,F>> {
  static const bool value = is_value<T>::value;
};

}
