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

/**
 * Array.
 *
 * @tparam T Value type.
 * @tparam F Frame type.
 */
template<class T, class F>
class Array {
public:
  using this_type = Array<T,F>;
  using value_type = T;
  using frame_type = F;
  using eigen_type = typename eigen_type<this_type>::type;
  using eigen_stride_type = typename eigen_stride_type<this_type>::type;

  /**
   * Constructor.
   */
  Array(const F& frame = F(), const Buffer<T>* buffer = nullptr,
      const int64_t offset = 0,
      const bool isView = false) :
      frame(frame),
      buffer(buffer),
      offset(offset),
      isView(isView) {
    //
  }

  /**
   * Constructor.
   *
   * @param frame Frame.
   */
  template<IS_VALUE(T)>
  Array(const F& frame) : Array<T,F>(frame) {
    this->allocate();
    this->initialize();
  }

  /**
   * Constructor.
   *
   * @param context Current context.
   * @param frame Frame.
   */
  template<IS_NOT_VALUE(T)>
  Array(Label* context, const F& frame) : Array(frame) {
    this->allocate();
    this->initialize(context);
  }

  /**
   * Constructor.
   *
   * @param frame Frame.
   * @param values Values.
   */
  template<IS_VALUE(T)>
  Array(const F& frame, const std::initializer_list<T>& values) :
      Array<T,F>(frame) {
    this->allocate();
    std::uninitialized_copy(values.begin(), values.end(), this->begin());
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
  template<IS_NOT_VALUE(T), class U>
  Array(Label* context, const F& frame, const std::initializer_list<U>& values) :
      Array(frame) {
    this->allocate();
    std::uninitialized_copy(values.begin(), values.end(), this->begin());
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
  template<IS_NOT_VALUE(T), class... Args>
  Array(Label* context, const F& frame, Args ... args) :
      Array(frame) {
    this->allocate();
    this->initialize(context, args...);
  }

  /**
   * Copy constructor.
   */
  template<IS_VALUE(T), class U, class G>
  Array(const Array<U,G>& o) : Array<T,F>(o.frame) {
    this->allocate();
    this->uninitialized_copy(o);
  }

  /**
   * Copy constructor.
   */
  template<IS_NOT_VALUE(T), class U, class G>
  Array(Label* context, const Array<U,G>& o) :
      Array(o.frame) {
    this->allocate();
    this->uninitialized_copy(context, o);
  }

  /**
   * Deep copy constructor.
   */
  template<IS_NOT_VALUE(T)>
  Array(Label* context, Label* label, const Array<T,F>& o) :
      Array(o.frame) {
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
  template<IS_VALUE(T), class U, class G>
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
   * Copy assignment. For a view the frames of the two arrays must
   * conform, otherwise a resize is permitted.
   */
  template<IS_NOT_VALUE(T), class U, class G>
  Array<T,F>& assign(Label* context, const Array<U,G>& o) {
    if (this->isView) {
      this->copy(context, o);
    } else {
      this->lock();
      this->frame = o.frame;
      if (o.isView) {
        this->release();
        this->allocate();
        this->copy(context, o);
      } else {
        this->buffer = o.buffer;
        this->buffer->incUsage();
      }
      this->unlock();
    }
    return *this;
  }

  /**
   * Copy assignment.
   */
  template<IS_VALUE(T)>
  Array<T,F>& operator=(const Array<T,F>& o) {
    return assign(o);
  }

  /**
   * Move assignment.
   */
  template<IS_NOT_VALUE(T)>
  Array<T,F>& assign(Label* context, Array<T,F> && o) {
    if (this->isView) {
      this->copy(context, o);
    } else {
      this->lock();
      this->frame = o.frame;
      if (o.isView) {
        this->release();
        this->allocate();
        this->copy(context, o);
      } else {
        this->buffer = o.buffer;
        o.buffer = nullptr;
      }
      this->unlock();
    }
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_VALUE(T)>
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
   * Move assignment.
   */
  template<IS_VALUE(T)>
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
   * Raw pointer to underlying buffer.
   */
  T* buf() const {
    return buffer->buf() + offset;
  }

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

  /**
   * Stride between rows.
   */
  auto rowStride() const {
    return F::count() == 1 ? this->frame.volume() : this->frame.stride(0);
  }

  /**
   * Stride between columns.
   */
  auto colStride() const {
    return F::count() == 1 ? this->frame.stride(0) : this->frame.stride(1);
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
    return Iterator<T,F>(this->duplicate()->buf() + this->offset, this->frame);
  }
  Iterator<T,F> begin() const {
    return Iterator<T,F>(this->buf(), this->frame);
  }
  ///@}

  /**
   * @name Views
   */
  ///@{
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
  ///@}

  /**
   * @name Resize
   */
  /**
   * Shrink a one-dimensional array in-place.
   *
   * @tparam G Frame type.
   *
   * @param frame New frame.
   */
  template<IS_VALUE(T), class G>
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
      this->uninitialized_copy(o1);
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
   * Shrink a one-dimensional array in-place.
   *
   * @tparam G Frame type.
   *
   * @param frame New frame.
   */
  template<IS_NOT_VALUE(T), class G>
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
      this->uninitialized_copy(o1);
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
  template<IS_VALUE(T), class G>
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
      this->uninitialized_copy(o1);
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
   * Enlarge a one-dimensional array in-place.
   *
   * @tparam G Frame type.
   *
   * @param frame New frame.
   * @param x Value to assign to new elements.
   */
  template<IS_NOT_VALUE(T), class G>
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
      this->uninitialized_copy(o1);
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
  ///@}

  /**
   * @name Eigen integration
   */
  ///@{
  template<IS_VALUE(T)>
  operator eigen_type() const {
    return this->toEigen();
  }

  template<IS_VALUE(T)>
  auto toEigen() {
    return eigen_type(this->duplicate()->buf() + this->offset, this->rows(),
        this->cols(), eigen_stride_type(this->rowStride(), this->colStride()));
  }
  template<IS_VALUE(T)>
  auto toEigen() const {
    return eigen_type(this->buf() + this->offset, this->rows(), this->cols(),
        eigen_stride_type(this->rowStride(), this->colStride()));
  }

  /**
   * Construct from Eigen Matrix expression.
   */
  template<IS_VALUE(T), class EigenType, std::enable_if_t<is_eigen_compatible<this_type,EigenType>::value,int> = 0>
  Array(const Eigen::MatrixBase<EigenType>& o) :
      Array<T,F>(F(o.rows(), o.cols())) {
    this->allocate();
    this->toEigen() = o;
  }

  /**
   * Construct from Eigen DiagonalWrapper expression.
   */
  template<IS_VALUE(T), class EigenType, std::enable_if_t<is_diagonal_compatible<this_type,EigenType>::value,int> = 0>
  Array(const Eigen::DiagonalWrapper<EigenType>& o) :
      Array<T,F>(F(o.rows(), o.cols())) {
    this->allocate();
    this->toEigen() = o;
  }

  /**
   * Construct from Eigen TriangularWrapper expression.
   */
  template<IS_VALUE(T), class EigenType, unsigned Mode, std::enable_if_t<is_triangle_compatible<this_type,EigenType>::value,int> = 0>
  Array(const Eigen::TriangularView<EigenType,Mode>& o) :
      Array<T,F>(F(o.rows(), o.cols())) {
    this->allocate();
    this->toEigen() = o;
  }
  ///@}
  template<IS_VALUE(T)>
  void freeze() {
    //
  }

  template<IS_NOT_VALUE(T)>
  void freeze() {
    auto iter = this->begin();
    auto last = iter + this->size();
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
    auto iter = this->begin();
    auto last = iter + this->size();
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
    auto iter = this->begin();
    auto last = iter + this->size();
    for (; iter != last; ++iter) {
      iter->finish();
    }
  }

private:
  /**
   * Is the buffer shared with one or more other arrays?
   */
  bool isShared() const {
    return buffer && buffer->numUsage() > 1u;
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
   * Duplicate underlying buffer by copy.
   */
  template<IS_VALUE(T)>
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
    return buffer;
  }

  /**
   * Duplicate underlying buffer by copy.
   */
  template<IS_NOT_VALUE(T)>
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
  template<IS_NOT_VALUE(T), class ... Args>
  void initialize(Args ... args) {
    auto iter = this->begin();
    auto last = iter + this->frame.size();
    for (; iter != last; ++iter) {
      new (&*iter) T(args...);
    }
  }

  /**
   * Copy from another array.
   */
  template<IS_VALUE(T), class U, class G>
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

  /**
   * Copy from another array.
   */
  template<IS_NOT_VALUE(T), class U, class G>
  void copy(Label* context, const Array<U,G>& o) {
    assert(!this->isShared());
    libbirch_assert_msg_(o.frame.conforms(this->frame), "array sizes are different");
    auto n = std::min(this->size(), o.size());
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
  template<IS_VALUE(T), class U, class G>
  void uninitialized_copy(const Array<U,G>& o) {
    assert(!this->isShared());
    libbirch_assert_msg_(o.frame.conforms(this->frame), "array sizes are different");
    auto n = std::min(this->size(), o.size());
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
    assert(!this->isShared());
    libbirch_assert_msg_(o.frame.conforms(this->frame), "array sizes are different");
    auto n = std::min(this->size(), o.size());
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
    assert(!this->isShared());
    libbirch_assert_msg_(o.frame.conforms(this->frame), "array sizes are different");
    auto n = std::min(this->size(), o.size());
    auto begin1 = o.begin();
    auto end1 = begin1 + n;
    auto begin2 = begin();
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

template<class T, class F>
struct is_value<Array<T,F>> {
  static const bool value = is_value<T>::value;
};

}
