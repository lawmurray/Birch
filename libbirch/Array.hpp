/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/Frame.hpp"
#include "libbirch/Iterator.hpp"
#include "libbirch/SharedCOW.hpp"
#include "libbirch/Allocator.hpp"
#include "libbirch/Sequence.hpp"
#include "libbirch/Eigen.hpp"

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
      ptr(nullptr),
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
      isView(false) {
    allocate();
    initialize(args...);
  }

  /**
   * Copy constructor.
   */
  Array(const Array<T,F>& o) :
      frame(o.frame),
      isView(false) {
    allocate();
    copy(o);
  }

  /**
   * Generic copy constructor.
   */
  template<class U, class G>
  Array(const Array<U,G>& o) :
      frame(o.frame),
      isView(false) {
    allocate();
    copy(o);
  }

  /**
   * Move constructor.
   */
  Array(Array<T,F> && o) :
      frame(o.frame),
      ptr(o.ptr),
      isView(o.isView) {
    o.isView = true;  // prevents deletion of ptr
  }

  /**
   * Sequence constructor.
   *
   * @param o Sequence.
   */
  Array(const typename sequence_type<T,F::count()>::type& o) :
      frame(sequence_frame(o)),
      isView(false) {
    allocate();
    copy(o);
  }

  /**
   * Destructor.
   */
  ~Array() {
    deallocate();
  }

  /**
   * Copy assignment. For a view the frames of the two arrays must conform,
   * otherwise a resize is permitted.
   */
  Array<T,F>& operator=(const Array<T,F>& o) {
    if (isView) {
      assign(o);
    } else {
      if (!frame.conforms(o.frame)) {
        deallocate();
        frame.resize(o.frame);
        allocate();
        copy(o);
      } else {
        assign(o);
      }
    }
    return *this;
  }

  /**
   * Generic copy assignment. For a view the frames of the two arrays must
   * conform, otherwise a resize is permitted.
   */
  template<class U, class G>
  Array<T,F>& operator=(const Array<U,G>& o) {
    if (isView) {
      assign(o);
    } else {
      if (!frame.conforms(o.frame)) {
        deallocate();
        frame.resize(o.frame);
        allocate();
        copy(o);
      } else {
        assign(o);
      }
    }
    return *this;
  }

  /**
   * Move assignment.
   */
  Array<T,F>& operator=(Array<T,F> && o) {
    if (isView) {
      assign(o);
    } else {
      if (o.isView) {
        if (!frame.conforms(o.frame)) {
          deallocate();
          frame.resize(o.frame);
          allocate();
          copy(o);
        } else {
          assign(o);
        }
      } else {
        deallocate();
        frame = std::move(o.frame);
        ptr = std::move(o.ptr);
        o.isView = true;  // prevents deletion of ptr
      }
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
        deallocate();
        frame = frame1;
        allocate();
        copy(o);
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
    return Array<T,decltype(frame(view))>(buf() + frame.serial(view),
        frame(view));
  }
  template<class View1, typename = std::enable_if_t<View1::rangeCount() != 0>>
  auto operator()(const View1& view) const {
    return Array<T,decltype(frame(view))>(buf() + frame.serial(view),
        frame(view));
  }
  template<class View1, typename = std::enable_if_t<View1::rangeCount() == 0>>
  auto& operator()(const View1& view) {
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
        std::is_same<T,typename DerivedType::value_type>::value &&
        ((F::count() == 1 && DerivedType::ColsAtCompileTime == 1) ||
        (F::count() == 2 && DerivedType::ColsAtCompileTime == Eigen::Dynamic));
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
  template<class DerivedType, typename = std::enable_if_t<is_eigen_compatible<DerivedType>::value>>
  Array(const Eigen::MatrixBase<DerivedType>& o, const F& frame) :
      frame(frame),
      isView(false) {
    allocate();
    toEigen() = o;
  }

  /**
   * Construct from Eigen Matrix expression.
   */
  template<class DerivedType, typename = std::enable_if_t<is_eigen_compatible<DerivedType>::value>>
  Array(const Eigen::MatrixBase<DerivedType>& o) :
      frame(o.rows(), o.cols()),
      isView(false) {
    allocate();
    toEigen() = o;
  }

  /**
   * Assign from Eigen Matrix expression.
   */
  template<class DerivedType, typename = std::enable_if_t<is_eigen_compatible<DerivedType>::value>>
  Array<T,F>& operator=(const Eigen::MatrixBase<DerivedType>& o) {
    if (!isView && !frame.conforms(o.rows(), o.cols())) {
      deallocate();
      frame.resize(o.rows(), o.cols());
      allocate();
    }
    toEigen() = o;
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

  /**
   * Get this. Used for compatibility with Shared<Array<...>>.
   */
  auto& get() {
    return *this;
  }
  auto& get() const {
    return *this;
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
    return Iterator<T,F>(ptr, frame);
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
    return ptr;
  }

  /**
   * Raw pointer to underlying buffer.
   */
  T* const buf() const {
    return ptr;
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
    assert(frame.size() < this->frame.size());

    /* destroy elements that will be removed */
    for (auto iter = begin() + frame.size(); iter != end(); ++iter) {
      iter->~T();
    }

    int64_t oldVol = this->frame.volume();
    this->frame.resize(frame);
    int64_t newVol = this->frame.volume();
    ptr = alloc.reallocate(ptr, oldVol, newVol);
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

    int64_t oldSize = this->frame.size();  // old size
    int64_t oldVol = this->frame.volume();
    this->frame.resize(frame);
    int64_t newVol = this->frame.volume();
    ptr = alloc.reallocate(ptr, oldVol, newVol);
    std::uninitialized_fill(begin() + oldSize, end(), x);
  }

private:
  /**
   * Constructor for view.
   *
   * @tparam Frame Frame type.
   *
   * @param ptr Existing allocation.
   * @param frame Frame.
   */
  Array(T* ptr, const F& frame) :
      frame(frame),
      ptr(ptr),
      isView(true) {
    //
  }

  /**
   * Allocate memory for array.
   */
  void allocate() {
    ptr = alloc.allocate(frame.volume());
  }

  /**
   * Deallocate memory of array.
   */
  void deallocate() {
    if (!isView) {
      for (auto iter = begin(); iter != end(); ++iter) {
        iter->~T();
      }
      alloc.deallocate(ptr, frame.volume());
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
    bi_assert_msg(o.frame.conforms(frame), "array sizes are different");

    std::uninitialized_copy(o.begin(), o.end(), begin());
  }

  void copy(const typename sequence_type<T,F::count()>::type& o) {
    bi_assert_msg(frame.conforms(sequence_frame(o)), "array size and sequence size are different");
    auto iter = begin();
    sequence_copy(iter, o);
  }

  /**
   * Assign from another array.
   */
  template<class U, class G>
  void assign(const Array<U,G>& o) {
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
    //if (frame.size() > 0) {
      //auto iter1 = begin();
      //auto end1 = end();
      //auto iter2 = o.begin();

      //int64_t block1 = frame.block();
      //int64_t block2 = o.frame.block();
      //int64_t block = gcd(block1, block2);

      //for (; iter1 != end1; iter1 += block, iter2 += block) {
      //  std::memmove(&(*iter1), &(*iter2), block * sizeof(T));
      //  // ^ memory regions may overlap, so avoid memcpy
      //}
      //bi_assert(iter2 == o.end());
    //}
  }

  void assign(const typename sequence_type<T,F::count()>::type& o) {
    bi_assert_msg(frame.conforms(sequence_frame(o)), "array size and sequence size are different");
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
   * Greatest common divisor of two positive integers.
   */
  static int64_t gcd(const int64_t a, const int64_t b) {
    /* pre-condition */
    assert(a > 0);
    assert(b > 0);

    int64_t a1 = a, b1 = b;
    while (a1 != b1 && b1 != 0) {
      a1 = a1 % b1;
      std::swap(a1, b1);
    }
    return a1;
  }

  /**
   * Frame.
   */
  F frame;

  /**
   * Buffer.
   */
  T* ptr;

  /**
   * Is this a view of another array? A view has stricter assignment
   * semantics, as it cannot be resized or moved.
   */
  bool isView;

  /**
   * Allocator.
   */
  Allocator<T> alloc;
};
}
