/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/type.hpp"
#include "libbirch/Label.hpp"
#include "libbirch/Frame.hpp"
#include "libbirch/Buffer.hpp"
#include "libbirch/Iterator.hpp"
#include "libbirch/Eigen.hpp"
#include "libbirch/ExclusiveLock.hpp"

namespace libbirch {

template<class T, class F>
class ArrayBase {
public:
  /**
   * Number of rows. For a one-dimensional array, this is the length.
   */
  auto rows() const {
    return frame.length(0);
  }

  /**
   * Number of columns. For a one-dimensional array, this is 1.
   */
  auto cols() const {
    return F::count() == 1 ? 1 : this->frame.length(1);
  }

protected:
  /**
   * Constructor.
   */
  ArrayBase(const F& frame = F(), const Buffer<T>* buffer = nullptr,
      const int64_t offset = 0,
      const bool isView = false) :
      frame(frame),
      buffer(buffer),
      offset(offset),
      isView(isView) {
    //
  }

  /**
   * Copy constructor.
   */
  ArrayBase(const ArrayBase<T,F>& o) :
      frame(o.frame),
      buffer(o.buffer),
      offset(o.offset),
      isView(true) {
    //
  }

  /**
   * Move constructor.
   */
  ArrayBase(ArrayBase<T,F> && o) :
      frame(o.frame),
      buffer(o.buffer),
      offset(o.offset),
      isView(o.isView) {
    o.buffer = nullptr;
  }

  /**
   * Raw pointer to underlying buffer.
   */
  T* buf() const {
    return buffer->buf() + offset;
  }

  auto rowStride() const {
    return F::count() == 1 ? this->frame.volume() : this->frame.stride(0);
  }

  auto colStride() const {
    return F::count() == 1 ? this->frame.stride(0) : this->frame.stride(1);
  }

  /**
   * Allocate memory for array, leaving uninitialized.
   */
  void allocate() {
    assert(!buffer);
    auto size = Buffer<T>::size(frame.volume());
    if (size > 0) {
      buffer = new (libbirch::allocate(size)) Buffer<T>();
      buffer->incUsage();
    }
  }

  /**
   * Is the buffer shared with one or more other arrays?
   */
  bool isShared() const {
    return buffer && buffer->numUsage() > 1u;
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
   * Frame.
   */
  F frame;

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

/**
 * A multidimensional array. Combines underlying data and a frame describing
 * the shape of that data. Allows the construction of views of the data,
 * where a view indexes either an individual element or some range of
 * elements.
 *
 * @ingroup libbirch
 *
 * @tparam Type Value type.
 * @tparam Frame Frame type.
 */
template<class T, class F, class Enable = void>
class Array {
  //
};

template<class T, class F>
class Array<T,F,IS_NOT_VALUE(T)> : public ArrayBase<T,F> {
  template<class U, class G, class Enable> friend class Array;
public:
  using this_type = Array<T,F>;
  using value_type = T;
  using frame_type = F;
  using eigen_type = typename eigen_type<this_type>::type;
  using eigen_stride_type = typename eigen_stride_type<this_type>::type;

  Array() = default;
  Array(const Array&) = default;
  Array(Array&&) = default;
  Array& operator=(const Array&) = delete;
  Array& operator=(Array&&) = delete;

  /**
   * Constructor.
   *
   * @param context Current context.
   * @param frame Frame.
   */
  Array(Label* context, const F& frame) {
    this->allocate();
    this->initialize(context);
  }

  /**
   * Constructor.
   *
   * @tparam ...Args Constructor parameter types.
   *
   * @param context Current context.
   * @param frame Frame.
   * @param args Constructor arguments.
   */
  template<class... Args>
  Array(Label* context, const F& frame, Args ... args) :
      ArrayBase<T,F>(frame) {
    this->allocate();
    this->initialize(context, args...);
  }

  /**
   * Constructor.
   *
   * @tparam ...Args Constructor parameter types.
   *
   * @param context Current context.
   * @param frame Frame.
   * @param values Values.
   */
  template<class U>
  Array(Label* context, const F& frame, const std::initializer_list<U>& values) :
      ArrayBase<T,F>(frame) {
    this->allocate();
    this->uninitialized_copy(context, values);
  }

  /**
   * Copy constructor.
   */
  template<class U, class G>
  Array(Label* context, const Array<U,G>& o) :
      ArrayBase<T,F>(o.frame) {
    this->allocate();
    this->uninitialized_copy(context, o);
  }

  /**
   * Deep copy constructor.
   */
  Array(Label* context, Label* label, const Array<T,F>& o) :
      ArrayBase<T,F>(o.frame) {
    this->allocate();
    this->uninitialized_copy(context, label, o);
  }

  /**
   * Destructor.
   */
  ~Array() {
    this->release();
  }

  /**
   * Copy assignment. For a view the frames of the two arrays must
   * conform, otherwise a resize is permitted.
   */
  template<class U, class G>
  Array<T,F>& assign(Label* context, const Array<U,G>& o) {
    if (this->isView) {
      this->copy(context, o);
    } else {
      this->lock();
      this->frame = o.frame;
      if (o.isView) {
        this->release();
        this->allocate();
        this->uninitialized_copy(context, o);
      } else {
        this->buffer = o.buffer;
        this->buffer->incUsage();
      }
      this->unlock();
    }
    return *this;
  }

  /**
   * Move assignment.
   */
  Array<T,F>& assign(Label* context, Array<T,F> && o) {
    if (this->isView) {
      this->copy(context, o);
    } else {
      this->lock();
      this->frame = o.frame;
      if (o.isView) {
        this->release();
        this->allocate();
        this->uninitialized_copy(context, o);
      } else {
        this->buffer = o.buffer;
        o.buffer = nullptr;
      }
      this->unlock();
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
   * Shrink a one-dimensional array in-place.
   *
   * @tparam G Frame type.
   *
   * @param frame New frame.
   */
  template<class G>
  void shrink(const G& frame) {
    static_assert(F::count() == 1, "can only shrink one-dimensional arrays");
    static_assert(G::count() == 1, "can only shrink one-dimensional arrays");
    assert(!this->isView);
    assert(frame.size() < this->size());

    this->lock();
    if (this->isShared()) {
      Array<T,F> o1(std::move(*this));
      this->frame = frame;
      this->allocate();
      this->copy(o1);
    } else {
      auto oldSize = Buffer<T>::size(this->frame.volume());
      auto newSize = Buffer<T>::size(frame.volume());
      this->frame = frame;
      if (this->frame.size() == 0) {
        release();
      } else {
        auto iter = Iterator<T,F>(this->buf(), this->frame);
        // ^ don't use begin() as inside a locked region
        auto last = iter + this->size();
        for (iter += frame.size(); iter != last; ++iter) {
          iter->~T();
        }
        this->buffer = (Buffer<T>*)libbirch::reallocate(this->buffer,
            oldSize, this->buffer->tid, newSize);
      }
    }
    this->unlock();
  }

  /**
   * Enlarge a one-dimensional array in-place.
   *
   * @tparam G Frame type.
   *
   * @param frame New frame.
   * @param x Value to assign to new elements.
   */
  template<class G>
  void enlarge(const G& frame, const T& x) {
    static_assert(F::count() == 1, "can only enlarge one-dimensional arrays");
    static_assert(G::count() == 1, "can only enlarge one-dimensional arrays");
    assert(!this->isView);
    assert(frame.size() > this->size());

    this->lock();
    auto n = this->frame.size();
    if (this->isShared() || !this->buffer) {
      Array<T,F> o1(std::move(*this));
      this->frame = frame;
      this->allocate();
      this->copy(o1);
    } else {
      auto oldSize = Buffer<T>::size(this->frame.volume());
      auto newSize = Buffer<T>::size(frame.volume());
      this->frame = frame;
      this->buffer = (Buffer<T>*)libbirch::reallocate(this->buffer, oldSize,
          this->buffer->tid, newSize);
    }
    Iterator<T,F> iter(this->buf(), this->frame);
    // ^ don't use begin() as we have obtained the lock already
    std::uninitialized_fill(iter + n, iter + this->frame.size(), x);
    this->unlock();
  }

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
    return Iterator<T,F>(this->duplicate()->buf() + this->offset, this->frame);
  }
  Iterator<T,F> begin() const {
    return Iterator<T,F>(this->buf(), this->frame);
  }

  /**
   * View operator.
   *
   * @tparam View1 View type.
   *
   * @param o View.
   *
   * @return The new array.
   */
  template<class View1, std::enable_if_t<View1::rangeCount() != 0,int> = 0>
  auto operator()(const View1& view) {
    return Array<T,decltype(this->frame(view))>(this->frame(view),
        this->duplicate(), this->offset + this->frame.serial(view), true);
  }
  template<class View1, std::enable_if_t<View1::rangeCount() != 0,int> = 0>
  auto operator()(const View1& view) const {
    return Array<T,decltype(this->frame(view))>(this->frame(view),
        this->buffer, this->offset + this->frame.serial(view), true);
  }
  template<class View1, std::enable_if_t<View1::rangeCount() == 0,int> = 0>
  auto& operator()(const View1& view) {
    return *(this->duplicate()->buf() + this->offset +
        this->frame.serial(view));
  }
  template<class View1, std::enable_if_t<View1::rangeCount() == 0,int> = 0>
  const auto& operator()(const View1& view) const {
    return *(this->buf() + this->frame.serial(view));
  }

  void freeze() {
    auto iter = this->begin();
    auto last = iter + this->size();
    for (; iter != last; ++iter) {
      iter->freeze();
    }
  }

  void thaw(Label* label) {
    auto iter = this->begin();
    auto last = iter + this->size();
    for (; iter != last; ++iter) {
      iter->thaw(label);
    }
  }

  void finish() {
    auto iter = this->begin();
    auto last = iter + this->size();
    for (; iter != last; ++iter) {
      iter->finish();
    }
  }

protected:
  /**
   * Constructor.
   */
  Array(const F& frame, const Buffer<T>* buffer, const int64_t offset,
      const bool isView) : ArrayBase<T,F>(frame, buffer, offset, isView) {
    //
  }

  /**
   * Duplicate underlying buffer by copy.
   */
  Buffer<T>* duplicate() {
    if (!this->isView) {
      this->lock();
      if (this->isShared()) {
        Array<T,F> o1(std::move(*this));
        this->allocate();
        this->uninitialized_copy(o1);
      }
      assert(!this->isShared());
      this->unlock();
    }
    return this->buffer;
  }

  /**
   * Deallocate memory of array.
   */
  void release() {
    if (this->buffer && this->buffer->decUsage() == 0) {
      Iterator<T,F> iter(this->buffer->buf() + this->offset, this->frame);
      // ^ just erased buffer, so can't use begin()
      auto last = iter + this->frame.size();
      for (; iter != last; ++iter) {
        iter->~T();
      }
      auto size = Buffer<T>::size(this->frame.volume());
      libbirch::deallocate(this->buffer, size, this->buffer->tid);
      this->buffer = nullptr;
    }
  }

  /**
   * Initialize allocated memory.
   *
   * @param args Constructor arguments.
   */
  template<class ... Args>
  void initialize(Args ... args) {
    auto iter = this->begin();
    auto last = iter + this->frame.size();
    for (; iter != last; ++iter) {
      new (&*iter) T(args...);
    }
  }

  /**
   * Uninitialized copy from another array.
   */
  template<class U, class G>
  void uninitialized_copy(const Array<U,G>& o) {
    assert(!this->isShared());
    libbirch_assert_msg_(o.frame.conforms(this->frame), "array sizes are different");
    auto first = o.begin();
    auto last = first + o.size();
    std::uninitialized_copy(first, last, this->begin());
  }

  /**
   * Deep copy from another array.
   */
  template<class U, class G>
  void uninitialized_copy(Label* label, const Array<U,G>& o) {
    assert(!this->isShared());
    libbirch_assert_msg_(o.frame.conforms(this->frame), "array sizes are different");
    auto n = std::min(this->size(), o.size());
    auto iter1 = o.begin();
    auto last1 = iter1 + n;
    auto iter2 = begin();
    for (; iter1 != last1; ++iter1, ++iter2) {
      new (&*iter2) T(label, *iter1);
    }
  }

  /**
   * Copy from another array.
   */
  template<class U, class G>
  void copy(const Array<U,G>& o) {
    assert(!this->isShared());
    libbirch_assert_msg_(o.frame.conforms(this->frame), "array sizes are different");
    auto n = std::min(this->size(), o.size());
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

};

template<class T, class F>
class Array<T,F,IS_VALUE(T)> : public ArrayBase<T,F> {
  template<class U, class G, class Enable> friend class Array;
public:
  Array() = default;
  Array(const Array&) = default;
  Array(Array&&) = default;

  using this_type = Array<T,F>;
  using value_type = T;
  using frame_type = F;
  using eigen_type = typename eigen_type<this_type>::type;
  using eigen_stride_type = typename eigen_stride_type<this_type>::type;

  /**
   * Constructor.
   *
   * @param frame Frame.
   */
  Array(const F& frame) : ArrayBase<T,F>(frame) {
    this->allocate();
  }

  /**
   * Constructor.
   *
   * @param frame Frame.
   * @param values Values.
   */
  Array(const F& frame, const std::initializer_list<T>& values) :
      ArrayBase<T,F>(frame) {
    this->allocate();
    std::uninitialized_copy(values.begin(), values.end(), this->begin());
  }

  /**
   * Copy constructor.
   */
  template<class U, class G>
  Array(const Array<U,G>& o) : ArrayBase<T,F>(o.frame) {
    this->allocate();
    this->copy(o);
  }

  /**
   * Destructor.
   */
  ~Array() {
    this->release();
  }

  /**
   * Copy assignment. For a view the frames of the two arrays must
   * conform, otherwise a resize is permitted.
   */
  template<class U, class G>
  Array<T,F>& assign(const Array<U,G>& o) {
    if (this->isView) {
      this->copy(o);
    } else {
      this->lock();
      this->frame = o.frame;
      if (o.isView) {
        this->release();
        this->allocate();
        this->copy(o);
      } else {
        this->buffer = o.buffer;
        this->buffer->incUsage();
      }
      this->unlock();
    }
    return *this;
  }

  /**
   * Move assignment.
   */
  Array<T,F>& assign(Array<T,F> && o) {
    if (this->isView) {
      this->copy(o);
    } else {
      this->lock();
      this->frame = o.frame;
      if (o.isView) {
        this->release();
        this->allocate();
        this->copy(o);
      } else {
        this->buffer = o.buffer;
        o.buffer = nullptr;
      }
      this->unlock();
    }
    return *this;
  }

  /**
   * Copy assignment.
   */
  Array<T,F>& operator=(const Array<T,F>& o) {
    return assign(o);
  }

  /**
   * Move assignment.
   */
  Array<T,F>& operator=(Array<T,F>&& o) {
    return assign(std::move(o));
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
   * Shrink a one-dimensional array in-place.
   *
   * @tparam G Frame type.
   *
   * @param frame New frame.
   */
  template<class G>
  void shrink(const G& frame) {
    static_assert(F::count() == 1, "can only shrink one-dimensional arrays");
    static_assert(G::count() == 1, "can only shrink one-dimensional arrays");
    assert(!this->isView);
    assert(frame.size() < this->size());

    this->lock();
    if (this->isShared()) {
      Array<T,F> o1(std::move(*this));
      this->frame = frame;
      this->allocate();
      this->copy(o1);
    } else {
      auto oldSize = Buffer<T>::size(this->frame.volume());
      auto newSize = Buffer<T>::size(frame.volume());
      this->frame = frame;
      if (this->frame.size() == 0) {
        release();
      } else {
        this->buffer = (Buffer<T>*)libbirch::reallocate(this->buffer,
            oldSize, this->buffer->tid, newSize);
      }
    }
    this->unlock();
  }

  /**
   * Enlarge a one-dimensional array in-place.
   *
   * @tparam G Frame type.
   *
   * @param frame New frame.
   * @param x Value to assign to new elements.
   */
  template<class G>
  void enlarge(const G& frame, const T& x) {
    static_assert(F::count() == 1, "can only enlarge one-dimensional arrays");
    static_assert(G::count() == 1, "can only enlarge one-dimensional arrays");
    assert(!this->isView);
    assert(frame.size() > this->size());

    this->lock();
    auto n = this->frame.size();
    if (this->isShared() || !this->buffer) {
      Array<T,F> o1(std::move(*this));
      this->frame = frame;
      this->allocate();
      this->copy(o1);
    } else {
      auto oldSize = Buffer<T>::size(this->frame.volume());
      auto newSize = Buffer<T>::size(frame.volume());
      this->frame = frame;
      this->buffer = (Buffer<T>*)libbirch::reallocate(this->buffer, oldSize,
          this->buffer->tid, newSize);
    }
    Iterator<T,F> iter(this->buf(), this->frame);
    // ^ don't use begin() as we have obtained the lock already
    std::fill(iter + n, iter + this->frame.size(), x);
    this->unlock();
  }

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
    return Iterator<T,F>(this->duplicate()->buf() + this->offset, this->frame);
  }
  Iterator<T,F> begin() const {
    return Iterator<T,F>(this->buf(), this->frame);
  }

  /**
   * View operator.
   *
   * @tparam View1 View type.
   *
   * @param o View.
   *
   * @return The new array.
   */
  template<class View1, std::enable_if_t<View1::rangeCount() != 0,int> = 0>
  auto operator()(const View1& view) {
    return Array<T,decltype(this->frame(view))>(this->frame(view),
        this->duplicate(), this->offset + this->frame.serial(view), true);
  }
  template<class View1, std::enable_if_t<View1::rangeCount() != 0,int> = 0>
  auto operator()(const View1& view) const {
    return Array<T,decltype(this->frame(view))>(this->frame(view),
        this->buffer, this->offset + this->frame.serial(view), true);
  }
  template<class View1, std::enable_if_t<View1::rangeCount() == 0,int> = 0>
  auto& operator()(const View1& view) {
    return *(this->duplicate()->buf() + this->offset + this->frame.serial(view));
  }
  template<class View1, std::enable_if_t<View1::rangeCount() == 0,int> = 0>
  const auto& operator()(const View1& view) const {
    return *(this->buf() + this->frame.serial(view));
  }

  /**
   * @name Eigen integration
   *
   * These functions and operators permit the implicit conversion between
   * Birch Array types and Eigen Matrix types.
   */
  ///@{
  operator eigen_type() const {
    return this->toEigen();
  }

  auto toEigen() {
    return eigen_type(this->duplicate()->buf() + this->offset, this->rows(),
        this->cols(), eigen_stride_type(this->rowStride(), this->colStride()));
  }
  auto toEigen() const {
    return eigen_type(this->buf() + this->offset, this->rows(), this->cols(),
        eigen_stride_type(this->rowStride(), this->colStride()));
  }

  /**
   * Construct from Eigen Matrix expression.
   */
  template<class EigenType, std::enable_if_t<is_eigen_compatible<this_type,EigenType>::value,int> = 0>
  Array(const Eigen::MatrixBase<EigenType>& o) :
      ArrayBase<T,F>(F(o.rows(), o.cols())) {
    this->allocate();
    this->toEigen() = o;
  }

  /**
   * Construct from Eigen DiagonalWrapper expression.
   */
  template<class EigenType, std::enable_if_t<is_diagonal_compatible<this_type,EigenType>::value,int> = 0>
  Array(const Eigen::DiagonalWrapper<EigenType>& o) :
      ArrayBase<T,F>(F(o.rows(), o.cols())) {
    this->allocate();
    this->toEigen() = o;
  }

  /**
   * Construct from Eigen TriangularWrapper expression.
   */
  template<class EigenType, unsigned Mode, std::enable_if_t<is_triangle_compatible<this_type,EigenType>::value,int> = 0>
  Array(const Eigen::TriangularView<EigenType,Mode>& o) :
      ArrayBase<T,F>(F(o.rows(), o.cols())) {
    this->allocate();
    this->toEigen() = o;
  }
  ///@}

  void freeze() {
    //
  }

  void thaw(Label* label) {
    //
  }

  void finish() {
    //
  }

protected:
  /**
   * Constructor.
   */
  Array(const F& frame, const Buffer<T>* buffer, const int64_t offset,
      const bool isView) : ArrayBase<T,F>(frame, buffer, offset, isView) {
    //
  }

  /**
   * Duplicate underlying buffer by copy.
   */
  Buffer<T>* duplicate() {
    if (!this->isView) {
      this->lock();
      if (this->isShared()) {
        Array<T,F> o1(std::move(*this));
        this->allocate();
        this->copy(o1);
      }
      assert(!this->isShared());
      this->unlock();
    }
    return buffer;
  }

  /**
   * Deallocate memory of array.
   */
  void release() {
    if (this->buffer && this->buffer->decUsage() == 0) {
      libbirch::deallocate(this->buffer, this->size, this->buffer->tid);
      this->buffer = nullptr;
    }
  }

  /**
   * Copy from another array.
   */
  template<class U, class G>
  void copy(const Array<U,G>& o) {
    assert(!this->isShared());
    libbirch_assert_msg_(o.frame.conforms(this->frame), "array sizes are different");
    auto n = std::min(this->size(), o.size());
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
};

template<class T, class F>
struct is_value<Array<T,F>> {
  static const bool value = is_value<T>::value;
};

}
