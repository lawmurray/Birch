/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/Frame.hpp"
#include "libbirch/Buffer.hpp"
#include "libbirch/Allocator.hpp"
#include "libbirch/Iterator.hpp"
#include "libbirch/SharedCOW.hpp"
#include "libbirch/Sequence.hpp"
#include "libbirch/Eigen.hpp"
#include "libbirch/ExclusiveLock.hpp"

namespace bi {
/**
 * Array. Combines underlying data and a frame describing the shape of that
 * data. Allows the construction of views of the data, where a view indexes
 * either an individual element or some range of elements.
 *
 * @ingroup libbirch
 *
 * @tparam Type Value type.
 * @tparam Frame Frame type.
 *
 * @todo Review in light of new context attribute of pointers.
 */
template<class T, class F = EmptyFrame>
class Array {
  template<class U, class G>
  friend class Array;
public:
  /**
   * Default constructor.
   */
  Array() :
      frame(),
      buffer(nullptr),
      offset(0),
      isView(false) {
    assert(frame.volume() == 0);
  }

  /**
   * Constructor.
   *
   * @param frame Frame.
   */
  Array(const F& frame) :
      frame(frame),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    initialize();
  }

  /**
   * Constructor.
   *
   * @tparam ...Args Constructor parameter types.
   *
   * @param frame Frame.
   * @param args Constructor arguments for each element.
   */
  template<class ... Args>
  Array(const F& frame, Args ... args) :
      frame(frame),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    initialize(args...);
  }

  /**
   * Copy constructor.
   *
   * @param canShare Is it fine for the new array to share an underlying
   * buffer with o (in a copy on write manner)?
   */
  Array(const Array<T,F>& o, const bool canShare = true);

  /**
   * Generic copy constructor.
   */
  template<class U, class G>
  Array(const Array<U,G>& o) :
      frame(o.frame),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    copy(o);
  }

  /**
   * Move constructor.
   */
  Array(Array<T,F> && o) :
      frame(o.frame),
      buffer(o.buffer.load()),
      offset(o.offset),
      isView(o.isView) {
    o.buffer = nullptr;
    o.offset = 0;
  }

  /**
   * Sequence constructor.
   *
   * @param o Sequence.
   */
  Array(const typename sequence_type<T,F::count()>::type& o) :
      frame(sequence_frame(o)),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    copy(o);
  }

  /**
   * Destructor.
   */
  ~Array() {
    release();
  }

  /**
   * Copy assignment. For a view the frames of the two arrays must conform,
   * otherwise a resize is permitted.
   */
  Array<T,F>& operator=(const Array<T,F>& o) {
    if (!isView || !frame.conforms(o.frame)) {
      rebase(o);
    } else if (lockIfShared()) {
      rebase(o);
      unlock();
    } else {
      assign(o);
    }
    return *this;
  }

  /**
   * Generic copy assignment. For a view the frames of the two arrays must
   * conform, otherwise a resize is permitted.
   */
  template<class U, class G>
  Array<T,F>& operator=(const Array<U,G>& o) {
    if (!isView || !frame.conforms(o.frame)) {
      rebase(o);
    } else if (lockIfShared()) {
      rebase(o);
      unlock();
    } else {
      assign(o);
    }
    return *this;
  }

  /**
   * Move assignment.
   */
  Array<T,F>& operator=(Array<T,F> && o) {
    if (!isView || !frame.conforms(o.frame)) {
      rebase(o);
    } else if (lockIfShared()) {
      rebase(o);
      unlock();
    } else {
      assign(o);
    }
  }

  /**
   * Generic sequence assignment. For a view the frames of array must
   * conform to that of the sequence, otherwise a resize is permitted.
   */
  Array<T,F>& operator=(const typename sequence_type<T,F::count()>::type& o) {
    if (!isView || !frame.conforms(o.frame)) {
      rebase(o);
    } else if (lockIfShared()) {
      rebase(o);
      unlock();
    } else {
      assign(o);
    }
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
  template<class View1, typename = std::enable_if_t<View1::rangeCount() != 0>>
  auto operator()(const View1& view) {
    duplicate();  // copy on write
    return Array<T,decltype(frame(view))>(buffer, offset + frame.serial(view),
        frame(view));
  }
  template<class View1, typename = std::enable_if_t<View1::rangeCount() != 0>>
  auto operator()(const View1& view) const {
    return Array<T,decltype(frame(view))>(buffer, offset + frame.serial(view),
        frame(view));
  }
  template<class View1, typename = std::enable_if_t<View1::rangeCount() == 0>>
  auto& operator()(const View1& view) {
    duplicate();  // copy on write
    return *(buf() + frame.serial(view));
  }
  template<class View1, typename = std::enable_if_t<View1::rangeCount() == 0>>
  const auto& operator()(const View1& view) const {
    return *(buf() + frame.serial(view));
  }

  /**
   * Equal comparison.
   */
  template<class G>
  bool operator==(const Array<T,G>& o) const {
    ///@todo Could optimize for arrays sharing the same buffer
    return frame.conforms(o.frame) && std::equal(begin(), end(), o.begin());
  }

  /**
   * Not equal comparison.
   */
  template<class G>
  bool operator!=(const Array<T,G>& o) const {
    return !(*this == o);
  }

  /**
   * @name Eigen integration
   *
   * These functions and operators permit the implicit conversion between
   * Birch Array types and Eigen Matrix types.
   */
  //@{
  /**
   * Compatibility check.
   */
  template<class DerivedType>
  struct is_eigen_compatible {
    static const bool value =
        std::is_same<T,typename DerivedType::value_type>::value
            && ((F::count() == 1 && DerivedType::ColsAtCompileTime == 1)
                || (F::count() == 2
                    && DerivedType::ColsAtCompileTime == Eigen::Dynamic));
  };

  /**
   * Appropriate Eigen Matrix type for this Birch Array type.
   */
  using EigenType = typename std::conditional<F::count() == 2,
  EigenMatrixMap<T>,
  typename std::conditional<F::count() == 1,
  EigenVectorMap<T>,
  void>::type>::type;

  using EigenStrideType = typename std::conditional<F::count() == 2,
  EigenMatrixStride,
  typename std::conditional<F::count() == 1,
  EigenVectorStride,
  void>::type>::type;

  /**
   * Convert to Eigen Matrix type.
   */
  EigenType toEigen() {
    duplicate();  // copy on write
    return EigenType(buf(), length(0), (F::count() == 1 ? 1 : length(1)),
        (F::count() == 1 ?
            EigenStrideType(stride(0), 1) :
            EigenStrideType(stride(0), stride(1))));
  }
  EigenType toEigen() const {
    return EigenType(buf(), length(0), (F::count() == 1 ? 1 : length(1)),
        (F::count() == 1 ?
            EigenStrideType(stride(0), 1) :
            EigenStrideType(stride(0), stride(1))));
  }

  /**
   * Construct with new allocation and copy in existing array from Eigen
   * Matrix expression.
   *
   * @param o Existing array.
   * @param frame Frame.
   *
   * Memory is allocated for the array, and is freed on destruction. After
   * allocation, the contents of the existing array are copied in.
   */
  template<class DerivedType, typename = std::enable_if_t<
      is_eigen_compatible<DerivedType>::value>>
  Array(const Eigen::MatrixBase<DerivedType>& o, const F& frame) :
      frame(frame),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    toEigen() = o;  // buffer uninitialized, but okay as type is primitive
  }

  /**
   * Construct from Eigen Matrix expression.
   */
  template<class DerivedType, typename = std::enable_if_t<
      is_eigen_compatible<DerivedType>::value>>
  Array(const Eigen::MatrixBase<DerivedType>& o) :
      frame(o.rows(), o.cols()),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    toEigen() = o;  // buffer uninitialized, but okay as type is primitive
  }

  /**
   * Assign from Eigen Matrix expression.
   */
  template<class DerivedType, typename = std::enable_if_t<
      is_eigen_compatible<DerivedType>::value>>
  Array<T,F>& operator=(const Eigen::MatrixBase<DerivedType>& o) {
    if (!isView || !frame.conforms(o.rows(), o.cols()) || isShared()) {
      rebase(o);
    } else {
      toEigen() = o;
    }
    return *this;
  }
  //@}

  /**
   * @name Queries
   */
  //@{
  /**
   * Get the length of the @p i th dimension.
   */
  int64_t length(const int i) const {
    return frame.length(i);
  }

  /**
   * Get the stride of the @p i th dimension.
   */
  int64_t stride(const int i) const {
    return frame.stride(i);
  }
  //@}

  /**
   * @name Iteration
   */
  //@{
  /**
   * Iterator pointing to the first element.
   *
   * Iterators are used to access the elements of an array sequentially.
   * Elements are visited in the order in which they are stored in memory;
   * the rightmost dimension is the fastest moving (for a matrix, this is
   * "row major" order).
   *
   * The idiom of iterator usage is as for the STL.
   */
  /**
   * Iterator pointing to the first element.
   */
  Iterator<T,F> begin() {
    duplicate();
    return Iterator<T,F>(buf(), frame);
  }
  Iterator<T,F> begin() const {
    return Iterator<T,F>(buf(), frame);
  }

  /**
   * Iterator pointing to one past the last element.
   */
  Iterator<T,F> end() {
    duplicate();
    return begin() + frame.size();
  }
  Iterator<T,F> end() const {
    return begin() + frame.size();
  }

  /**
   * Raw pointer to underlying buffer.
   */
  T* buf() {
    auto buffer = this->buffer.load();
    return buffer ? buffer->get() + offset : nullptr;
  }

  /**
   * Raw pointer to underlying buffer.
   */
  T* const buf() const {
    auto buffer = this->buffer.load();
    return buffer ? buffer->get() + offset : nullptr;
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
    assert(!isView);
    assert(buffer);
    assert(frame.size() < this->frame.size());

    if (lockIfShared()) {
      Array<T,F> o(*this);
      release();
      this->frame.resize(frame);
      allocate();
      std::uninitialized_copy(o.begin(), o.begin() + frame.size(), begin());
      unlock();
    } else {
      for (auto iter = begin() + frame.size(); iter != end(); ++iter) {
        iter->~T();
      }
      if (frame.size() == 0) {
        release();
      } else {
        buffer = (Buffer<T>*)bi::reallocate(buffer,
            Buffer<T>::size(this->frame.volume()),
            Buffer<T>::size(frame.volume()));
      }
      this->frame.resize(frame);
    }
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
    assert(!isView);
    assert(frame.size() > this->frame.size());

    auto nelements = this->frame.size();
    if (lockIfShared()) {
      Array<T,F> o(*this);
      release();
      this->frame.resize(frame);
      allocate();
      std::uninitialized_copy(o.begin(), o.end(), begin());
      unlock();
    } else {
      auto oldSize = Buffer<T>::size(this->frame.volume());
      auto newSize = Buffer<T>::size(frame.volume());
      this->frame.resize(frame);
      if (buffer) {
        buffer = (Buffer<T>*)bi::reallocate(buffer, oldSize, newSize);
      } else {
        allocate();
      }
    }
    std::uninitialized_fill(begin() + nelements, end(), x);
  }

private:
  /**
   * Constructor for view.
   *
   * @tparam Frame Frame type.
   *
   * @param buffer Buffer.
   * @param offset Offset.
   * @param frame Frame.
   */
  Array(Buffer<T>* buffer, const ptrdiff_t offset, const F& frame) :
      frame(frame),
      buffer(buffer),
      offset(offset),
      isView(true) {
    //
  }

  /**
   * Allocate memory for array, leaving uninitialized.
   */
  void allocate() {
    assert(!buffer);
    auto size = Buffer<T>::size(frame.volume());
    if (size > 0) {
      auto buffer = new (bi::allocate(size)) Buffer<T>();
      buffer->incUsage();
      this->buffer = buffer;
    } else {
      buffer = nullptr;
    }
  }

  /**
   * Duplicate underlying buffer by copy.
   */
  void duplicate() {
    if (lockIfShared()) {
      assert(!isView);
      rebase(std::move(Array<T,F>(*this, false)));
      unlock();
    }
  }

  /**
   * Rebase to match the contents of an existing array (possibly sharing a
   * buffer with it, using copy on write).
   */
  void rebase(const Array<T,F>& o) {
    assert(!isView);
    rebase(std::move(Array<T,F>(o)));
  }

  /**
   * Rebase to match the contents of an existing array (possibly sharing a
   * buffer with it, using copy on write).
   */
  void rebase(Array<T,F>&& o) {
    assert(!isView && offset == 0);
    assert(!o.isView && o.offset == 0);

    std::swap(frame, o.frame);

    /* can't use std::swap on atomics */
    auto buffer = this->buffer.load();
    this->buffer = o.buffer.load();
    o.buffer = buffer;
  }

  /**
   * Deallocate memory of array.
   */
  void release() {
    auto buffer = this->buffer.load();
    if (!isView && buffer && buffer->decUsage() == 0) {
      for (auto iter = begin(); iter != end(); ++iter) {
        iter->~T();
      }
      size_t size = Buffer<T>::size(frame.volume());
      bi::deallocate(buffer, size);
    }
    this->buffer = nullptr;
    offset = 0;
  }

  /**
   * Initialize allocated memory.
   *
   * @param args Constructor arguments.
   */
  template<class ... Args>
  void initialize(Args ... args) {
    for (auto iter = begin(); iter != end(); ++iter) {
      emplace(*iter, args...);
    }
  }

  /**
   * Copy from another array.
   */
  template<class U, class G>
  void copy(const Array<U,G>& o) {
    assert(!isShared());
    bi_assert_msg(o.frame.conforms(frame), "array sizes are different");
    std::uninitialized_copy(o.begin(), o.end(), begin());
  }

  void copy(const typename sequence_type<T,F::count()>::type& o) {
    assert(!isShared());
    bi_assert_msg(frame.conforms(sequence_frame(o)),
        "array size and sequence size are different");
    auto iter = begin();
    sequence_copy(iter, o);
  }

  /**
   * Assign from another array.
   */
  template<class U, class G>
  void assign(const Array<U,G>& o) {
    assert(!isShared());
    bi_assert_msg(o.frame.conforms(frame), "array sizes are different");

    auto begin1 = o.begin();
    auto end1 = o.end();
    auto begin2 = begin();
    auto end2 = end();
    if (inside(begin1, end1, begin2)) {
      std::copy_backward(begin1, end1, end2);
    } else {
      std::copy(begin1, end1, begin2);
    }
  }

  void assign(const typename sequence_type<T,F::count()>::type& o) {
    assert(!isShared());
    bi_assert_msg(frame.conforms(sequence_frame(o)),
        "array size and sequence size are different");
    auto iter = begin();
    sequence_assign(iter, o);
  }

  /**
   * Construct element of value type in place.
   *
   * @param o Element.
   * @param args Constructor arguments.
   */
  template<class U, class ... Args>
  static void emplace(U& o, Args ... args) {
    new (&o) U(args...);
  }

  /**
   * Construct element of shared pointer type in place.
   *
   * @param o Element.
   * @param args Constructor arguments.
   */
  template<class U, class ... Args>
  static void emplace(SharedCOW<U>& o, Args ... args) {
    new (&o) SharedCOW<U>(U::create(args...));
  }

  /**
   * Is the buffer shared with one or more other arrays?
   */
  bool isShared() const {
    auto buffer = this->buffer.load();
    return buffer && buffer->numUsage() > 1u;
  }

  /**
   * Obtain the lock.
   */
  bool lockIfShared() {
    if (isShared()) {
      mutex.keep();
      if (isShared()) {
        return true;
      } else {
        mutex.unkeep();
        return false;
      }
    } else {
      return false;
    }
  }

  /**
   * Release the lock.
   */
  void unlock() {
    mutex.unkeep();
  }

  /**
   * Frame.
   */
  F frame;

  /**
   * Buffer.
   */
  std::atomic<Buffer<T>*> buffer;

  /**
   * Offset into the buffer. This should be zero when isView is false.
   */
  ptrdiff_t offset;

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
}

#include "libbirch/value.hpp"

template<class T, class F>
bi::Array<T,F>::Array(const Array<T,F>& o, const bool canShare) :
    frame(o.frame),
    buffer(o.buffer.load()),
    offset(o.offset),
    isView(o.isView) {
  if (!canShare || (cloneUnderway && !is_value<T>::value)) {
    /* either the caller has explicitly requested a copy (canShare), or we
     * are cloning an array that is not of purely value type, in which case
     * we must copy for correct bookkeeping under the lazy deep clone
     * rules */
    buffer = nullptr;  // hadn't increment count yet anyway
    offset = 0u;
    allocate();
    copy(o);
  } else if (!isView && buffer) {
    /* views do not increment the buffer use count, as they are meant to be
     * temporary and should not outlive the buffer itself */
    buffer.load()->incUsage();
  }
}
