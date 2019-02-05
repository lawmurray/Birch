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
   */
  Array(const Array<T,F>& o);

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
      buffer(o.buffer),
      offset(o.offset),
      isView(o.isView) {
    o.isView = true;  // prevents decrement of buffer usage count
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
    ///@todo Optimize to just copy pointer to buffer where possible
    if (isView) {
      assign(o);
    } else if (!frame.conforms(o.frame)) {
      lock();
      Array<T,F> o1(*this);
      frame.resize(o.frame);
      allocate();
      copy(o);
      unlock();
    } else if (lockIfShared()) {
      Array<T,F> o1(*this);
      allocate();
      copy(o);
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
    ///@todo Optimize to just copy pointer to buffer where possible
    if (isView) {
      assign(o);
    } else if (!frame.conforms(o.frame)) {
      lock();
      Array<T,F> o1(*this);
      frame.resize(o.frame);
      allocate();
      copy(o);
      unlock();
    } else if (lockIfShared()) {
      Array<T,F> o1(*this);
      allocate();
      copy(o);
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
    if (isView) {
      assign(o);
    } else if (o.isView) {
      if (!frame.conforms(o.frame)) {
        lock();
        Array<T,F> o1(*this);
        frame.resize(o.frame);
        allocate();
        copy(o);
        unlock();
      } else if (lockIfShared()) {
        Array<T,F> o1(*this);
        allocate();
        copy(o);
        unlock();
      } else {
        assign(o);
      }
    } else {
      lock();
      std::swap(frame, o.frame);
      std::swap(buffer, o.buffer);
      std::swap(offset, o.offset);
      std::swap(isView, o.isView);
      unlock();
    }
    return *this;
  }

  /**
   * Generic sequence assignment. For a view the frames of array must
   * conform to that of the sequence, otherwise a resize is permitted.
   */
  Array<T,F>& operator=(const typename sequence_type<T,F::count()>::type& o) {
    if (isView) {
      assign(o);
    } else {
      auto frame1 = sequence_frame(o);
      if (!frame.conforms(frame1)) {
        lock();
        Array<T,F> o1(*this);
        frame = frame1;
        allocate();
        copy(o);
        unlock();
      } else if (lockIfShared()) {
        Array<T,F> o1(*this);
        allocate();
        copy(o);
        unlock();
      } else {
        assign(o);
      }
    }
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
    if (!isView) {
      if (!frame.conforms(o.rows(), o.cols())) {
        lock();
        Array<T,F> o1(*this);
        frame.resize(o.rows(), o.cols());
        allocate();
        unlock();
      } else if (lockIfShared()) {
        Array<T,F> o1(*this);
        allocate();
        unlock();
      }
    }
    toEigen() = o;  // buffer may be uninitialized, but okay for primitive type
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
  Iterator<T,F> begin() const {
    return Iterator<T,F>(buf(), frame);
  }

  /**
   * Iterator pointing to one past the last element.
   */
  Iterator<T,F> end() const {
    return begin() + frame.size();
  }

  /**
   * Raw pointer to underlying buffer.
   */
  T* buf() {
    return buffer ? buffer->get() + offset : nullptr;
  }

  /**
   * Raw pointer to underlying buffer.
   */
  T* const buf() const {
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
      this->frame.resize(frame);
      allocate();
      std::uninitialized_copy(o.begin(), o.begin() + frame.size(), begin());
      unlock();
    } else {
      lock();
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
      unlock();
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
      this->frame.resize(frame);
      allocate();
      std::uninitialized_copy(o.begin(), o.end(), begin());
    } else {
      lock();
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
    unlock();
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
   * Allocate memory for array.
   */
  void allocate() {
    if (buffer) {
      release();
    }
    auto size = Buffer<T>::size(frame.volume());
    if (size > 0) {
      buffer = new (bi::allocate(size)) Buffer<T>();
      buffer->incUsage();
    } else {
      buffer = nullptr;
    }
  }

  /**
   * Deallocate memory of array.
   */
  void release() {
    if (!isView && buffer && buffer->decUsage() == 0) {
      for (auto iter = begin(); iter != end(); ++iter) {
        iter->~T();
      }
      size_t size = Buffer<T>::size(frame.volume());
      bi::deallocate(buffer, size);
      buffer = nullptr;
      offset = 0;
    }
  }

  /**
   * If the buffer is shared, copy it for writing.
   */
  void duplicate() {
    if (lockIfShared()) {
      assert(!isView);  // if view, should have duplicated when it was made
      Array<T,F> o(*this);
      allocate();
      copy(o);
      unlock();
    }
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
    assert(!buffer || buffer->numUsage() == 1);
    bi_assert_msg(o.frame.conforms(frame), "array sizes are different");
    std::uninitialized_copy(o.begin(), o.end(), begin());
  }

  void copy(const typename sequence_type<T,F::count()>::type& o) {
    assert(!buffer || buffer->numUsage() == 1);
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
    assert(!buffer || buffer->numUsage() == 1);
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
    assert(!buffer || buffer->numUsage() == 1);
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
   * Obtain the lock.
   */
  void lock() {
    mutex.keep();
  }

  /**
   * Is the buffer shared? If the array is locked, first waits on the lock.
   * Then, if shared, obtains the lock and returns true (the caller should
   * release with unlock()), while if not shared returns false.
   */
  bool lockIfShared() {
    //if (buffer && buffer->numUsage() > 1u) {
    mutex.keep();
    if (buffer && buffer->numUsage() > 1u) {
      return true;
    } else {
      mutex.unkeep();
      return false;
    }
    //} else {
    //  return false;
    //}
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
  Buffer<T>* buffer;

  /**
   * Offset into the buffer.
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
bi::Array<T,F>::Array(const Array<T,F>& o) :
    frame(o.frame),
    buffer(o.buffer),
    offset(o.offset),
    isView(o.isView) {
  if (!isView && buffer) {
    buffer->incUsage();
  }
  if (cloneUnderway && !is_value<T>::value) {
    /* arrays other than those with purely value types must be copied here
     * for correct bookkeeping with lazy deep clone */
    allocate();
    copy(o);
  }
}
