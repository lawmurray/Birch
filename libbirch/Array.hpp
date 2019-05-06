/**
 * @file
 */
#pragma once

#include "libbirch/memory.hpp"
#include "libbirch/Frame.hpp"
#include "libbirch/Buffer.hpp"
#include "libbirch/Iterator.hpp"
#include "libbirch/Shared.hpp"
#include "libbirch/Sequence.hpp"
#include "libbirch/Eigen.hpp"
#include "libbirch/ExclusiveLock.hpp"

#include <atomic>

namespace libbirch {
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
      buffer(o.buffer.load(std::memory_order_relaxed)),
      offset(o.offset),
      isView(o.isView) {
    o.buffer.store(nullptr, std::memory_order_relaxed);
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
    if (!isView && (!frame.conforms(o.frame) || isShared())) {
      lock();
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
    if (!isView && (!frame.conforms(o.frame) || isShared())) {
      lock();
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
    if (!isView && (!frame.conforms(o.frame) || isShared())) {
      lock();
      if (o.isView) {
        Array<T,F> o1(o, false);
        rebase(std::move(o1));
      } else {
        rebase(std::move(o));
      }
      unlock();
    } else {
      assign(o);
    }
    return *this;
  }

  /**
   * Generic sequence assignment. For a view the frames of array must
   * conform to that of the sequence, otherwise a resize is permitted.
   */
  Array<T,F>& operator=(const typename sequence_type<T,F::count()>::type& o) {
    if (!isView && (!frame.conforms(o.frame) || isShared())) {
      lock();
      rebase(o);
      unlock();
    } else {
      assign(o);
    }
  }

  /**
   * Return const reference to the array. This can be used to ensure that the
   * array is being accessed in a const context, to avoid unnecessary copying
   * of shared buffers.
   */
  const Array<T,F>& as_const() {
    return *this;
  }
  const Array<T,F>& as_const() const {
    return *this;
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
    return Array<T,decltype(frame(view))>(duplicate(),
        offset + frame.serial(view), frame(view));
  }
  template<class View1, typename = std::enable_if_t<View1::rangeCount() != 0>>
  auto operator()(const View1& view) const {
    return Array<T,decltype(frame(view))>(buffer, offset + frame.serial(view),
        frame(view));
  }
  template<class View1, typename = std::enable_if_t<View1::rangeCount() == 0>>
  auto& operator()(const View1& view) {
    return *(duplicate()->buf() + offset + frame.serial(view));
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
    auto first = begin();
    auto last = first + size();
    return frame.conforms(o.frame) && std::equal(first, last, o.begin());
  }

  /**
   * Not equal comparison.
   */
  template<class G>
  bool operator!=(const Array<T,G>& o) const {
    return !(*this == o);
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
    return Iterator<T,F>(duplicate()->buf() + offset, frame);
  }
  Iterator<T,F> begin() const {
    return Iterator<T,F>(buf(), frame);
  }

  /**
   * Size (product of lengths).
   */
  int64_t size() const {
    return frame.size();
  }

  /**
   * Volume (number of elements allocated in buffer).
   */
  int64_t volume() const {
    return frame.volume();
  }

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

  /**
   * Raw pointer to underlying buffer.
   */
  T* buf() {
    return buffer.load(std::memory_order_relaxed)->buf() + offset;
  }

  /**
   * Raw pointer to underlying buffer.
   */
  T* buf() const {
    return buffer.load(std::memory_order_relaxed)->buf() + offset;
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
    assert(frame.size() < size());

    lock();
    if (isShared()) {
      rebase(Array<T,F>(*this, frame));
    } else {
      Iterator<T,F> iter(buf(), frame);
      // ^ don't use begin() as we have obtained the lock already
      auto last = iter + size();
      for (iter += frame.size(); iter != last; ++iter) {
        iter->~T();
      }
      if (frame.size() == 0) {
        release();
      } else {
        auto oldBuffer = buffer.load(std::memory_order_relaxed);
        auto oldSize = Buffer<T>::size(volume());
        auto newSize = Buffer<T>::size(frame.volume());
        buffer.store(
            (Buffer<T>*)libbirch::reallocate(oldBuffer, oldSize, oldBuffer->tid,
                newSize), std::memory_order_relaxed);
      }
      this->frame.resize(frame);
    }
    unlock();
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
    assert(frame.size() > size());

    lock();
    if (isShared() || !buffer) {
      rebase(Array<T,F>(*this, frame, x));
    } else {
      auto oldVolume = this->frame.volume();
      this->frame.resize(frame);
      auto oldBuffer = buffer.load(std::memory_order_relaxed);
      auto oldSize = Buffer<T>::size(oldVolume);
      auto newSize = Buffer<T>::size(frame.volume());
      buffer.store(
          (Buffer<T>*)libbirch::reallocate(oldBuffer, oldSize, oldBuffer->tid,
              newSize), std::memory_order_relaxed);
      Iterator<T,F> iter(buf(), frame);
      // ^ don't use begin() as we have obtained the lock already
      std::uninitialized_fill(iter + oldVolume, iter + frame.size(), x);
    }
    unlock();
  }

  /**
   * @name Eigen integration
   *
   * These functions and operators permit the implicit conversion between
   * Birch Array types and Eigen Matrix types.
   */
  ///@{
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
   * Explicitly convert to Eigen matrix type.
   */
  EigenType toEigen() {
    return EigenType(duplicate()->buf() + offset, length(0),
        (F::count() == 1 ? 1 : length(1)),
        (F::count() == 1 ?
            EigenStrideType(volume(), stride(0)) :
            EigenStrideType(stride(0), stride(1))));
  }
  EigenType toEigen() const {
    return EigenType(buf(), length(0),
        (F::count() == 1 ? 1 : length(1)),
        (F::count() == 1 ?
            EigenStrideType(volume(), stride(0)) :
            EigenStrideType(stride(0), stride(1))));
  }

  /**
   * Implicitly convert to Eigen matrix type.
   */
  operator EigenType() {
    return toEigen();
  }
  operator EigenType() const {
    return toEigen();
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
    if (!isView && (!frame.conforms(o.rows(), o.cols()) || isShared())) {
      lock();
      rebase(o);
      unlock();
    } else {
      toEigen() = o;
    }
    return *this;
  }
  ///@}

private:
  /**
   * Constructor for view.
   *
   * @param buffer Buffer.
   * @param offset Offset.
   * @param frame Frame.
   */
  Array(Buffer<T>* buffer, const int64_t offset, const F& frame) :
      frame(frame),
      buffer(buffer),
      offset(offset),
      isView(true) {
    //
  }

  /**
   * Constructor for enlarging a one dimensional array.
   *
   * @tparam G Frame type.
   *
   * @param o Original array.
   * @param frame Frame.
   * @param x Value to assign to new elements.
   */
  template<class G>
  Array(const Array<T,F>& o, const G& frame, const T& x) :
      frame(frame),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    auto first = o.begin();
    auto iter = begin();
    std::uninitialized_copy(first, first + o.frame.size(), iter);
    std::uninitialized_fill(iter + o.frame.size(), iter + frame.size(), x);
  }

  /**
   * Constructor for shrinking a one dimensional array.
   *
   * @tparam G Frame type.
   *
   * @param o Original array.
   * @param frame Frame.
   */
  template<class G>
  Array(const Array<T,F>& o, const G& frame) :
      frame(frame),
      buffer(nullptr),
      offset(0),
      isView(false) {
    allocate();
    auto first = o.begin();
    auto last = first + frame.size();
    auto iter = begin();
    std::uninitialized_copy(first, last, iter);
  }

  /**
   * Allocate memory for array, leaving uninitialized.
   */
  void allocate() {
    assert(!buffer);
    auto size = Buffer<T>::size(frame.volume());
    if (size > 0) {
      auto tmp = new (libbirch::allocate(size)) Buffer<T>();
      tmp->incUsage();
      buffer.store(tmp, std::memory_order_relaxed);
    }
  }

  /**
   * Duplicate underlying buffer by copy.
   */
  Buffer<T>* duplicate() {
    if (!isView) {
      lock();
      if (isShared()) {
        Array<T,F> o1(*this, false);
        rebase(std::move(o1));
      }
      assert(!isShared());
      auto ptr = buffer.load(std::memory_order_relaxed);
      unlock();
      return ptr;
    } else {
      return buffer.load(std::memory_order_relaxed);
    }
  }

  /**
   * Rebase to match the contents of an existing array (possibly sharing a
   * buffer with it, using copy on write).
   */
  void rebase(const Array<T,F>& o) {
    assert(!isView);
    Array<T,F> o1(o);
    // ^ the copy must be named like this, not a temporary, to avoid copy
    //   elision, i.e. rebase(std::move(Array<T,F>(o))) may elide
    rebase(std::move(o1));
  }

  /**
   * Rebase to match the contents of an existing array (possibly sharing a
   * buffer with it, using copy on write).
   */
  void rebase(Array<T,F> && o) {
    assert(!isView);
    assert(!o.isView);
    std::swap(frame, o.frame);
    o.buffer.store(
        buffer.exchange(o.buffer.load(std::memory_order_relaxed),
            std::memory_order_relaxed), std::memory_order_relaxed);  // can't std::swap atomics
  }

  /**
   * Deallocate memory of array.
   */
  void release() {
    auto tmp = buffer.exchange(nullptr);
    if (!isView) {
      if (tmp && tmp->decUsage() == 0) {
        Iterator<T,F> iter(tmp->buf() + offset, frame);
        // ^ just erased buffer, so can't use begin()
        auto last = iter + size();
        for (; iter != last; ++iter) {
          iter->~T();
        }
        size_t size = Buffer<T>::size(frame.volume());
        libbirch::deallocate(tmp, size, tmp->tid);
      }
    }
  }

  /**
   * Initialize allocated memory.
   *
   * @param args Constructor arguments.
   */
  template<class ... Args>
  void initialize(Args ... args) {
    auto iter = begin();
    auto last = iter + size();
    for (; iter != last; ++iter) {
      emplace(*iter, args...);
    }
  }

  /**
   * Copy from another array.
   */
  template<class U, class G>
  void copy(const Array<U,G>& o) {
    assert(!isShared());
    libbirch_assert_msg_(o.frame.conforms(frame), "array sizes are different");
    auto first = o.begin();
    auto last = first + o.size();
    std::uninitialized_copy(first, last, begin());
  }

  void copy(const typename sequence_type<T,F::count()>::type& o) {
    assert(!isShared());
    libbirch_assert_msg_(frame.conforms(sequence_frame(o)),
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
    libbirch_assert_msg_(o.frame.conforms(frame), "array sizes are different");

    auto begin1 = o.begin();
    auto end1 = begin1 + o.size();
    auto begin2 = begin();
    auto end2 = begin2 + size();
    if (inside(begin1, end1, begin2)) {
      /* copy backward */
      while (begin1 != end1) {
        --end1;
        --end2;
        *end2 = static_cast<T>(*end1);
      }
    } else {
      /* copy forward */
      while (begin1 != end1) {
        *begin2 = static_cast<T>(*begin1);
        ++begin1;
        ++begin2;
      }
    }
    // ^ std::copy_backward and std::copy could be used here, but explicit
    //   cast required when Shared<T> on left and some other U on right
  }

  void assign(const typename sequence_type<T,F::count()>::type& o) {
    assert(!isShared());
    libbirch_assert_msg_(frame.conforms(sequence_frame(o)),
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
  static void emplace(Shared<U>& o, Args ... args) {
    new (&o) Shared<U>(U::create_(args...));
  }

  /**
   * Is the buffer shared with one or more other arrays?
   */
  bool isShared() const {
    auto tmp = buffer.load(std::memory_order_relaxed);
    auto result = tmp && tmp->numUsage() > 1u;
    return result;
  }

  /**
   * Release the lock.
   */
  void lock() {
    mutex.keep();
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
}

#include "libbirch/value.hpp"

template<class T, class F>
libbirch::Array<T,F>::Array(const Array<T,F>& o, const bool canShare) :
    frame(o.frame),
    buffer(nullptr),
    offset(0),
    isView(false) {
  if (!canShare || (cloneUnderway && !is_value<T>::value)) {
    /* either the caller has explicitly requested a copy (canShare), or we
     * are cloning an array that is not of purely value type, in which case
     * we must copy for correct bookkeeping under the lazy deep clone
     * rules */
    allocate();
    copy(o);
  } else {
    auto tmp = o.buffer.load(std::memory_order_relaxed);
    if (tmp && !o.isView) {
      /* views do not increment the buffer use count, as they are meant to be
       * temporary and should not outlive the buffer itself */
      tmp->incUsage();
    }
    buffer.store(tmp, std::memory_order_relaxed);
    offset = o.offset;
    isView = o.isView;
  }
}
