/**
 * @file
 */
#pragma once

#include "libbirch/Frame.hpp"
#include "libbirch/Iterator.hpp"
#include "libbirch/SharedPointer.hpp"
#include "libbirch/Sequence.hpp"
#include "libbirch/Eigen.hpp"
#include "libbirch/global.hpp"

#include <cstring>

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
 */
template<class T, class F = EmptyFrame>
class Array {
  template<class U, class G>
  friend class Array;
public:
  /**
   * Constructor.
   *
   * @param frame Frame.
   */
  Array(const F& frame = F()) :
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
   * Move constructor.
   */
  Array(Array<T,F> && o) :
      frame(o.frame),
      ptr(o.ptr),
      isView(o.isView) {
    o.isView = false;  // prevents deletion of ptr
  }

  /**
   * Sequence constructor.
   *
   * @param o Sequence.
   */
  template<class U>
  Array(const Sequence<U>& o) :
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
        frame.resize(o.frame);
        deallocate();
        allocate();
        copy(o);
      } else {
        assign(o);
      }
    }
    return *this;
  }

  /**
   * Move assignment. The frames of the two arrays must conform.
   */
  Array<T,F>& operator=(Array<T,F> && o) {
    if (isView) {
      assign(o);
    } else {
      if (o.isView) {
        if (!frame.conforms(o.frame)) {
          frame.resize(o.frame);
          deallocate();
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
   * Sequence assignment.
   *
   * @param o Sequence.
   */
  template<class U>
  Array<T,F>& operator=(const Sequence<U>& o) {
    assign(o);
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
    static const bool value = (F::count() == 1
        && DerivedType::ColsAtCompileTime == 1)
        || (F::count() == 2
            && DerivedType::ColsAtCompileTime == Eigen::Dynamic);
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
  template<class DerivedType, typename = std::enable_if_t<
      is_eigen_compatible<DerivedType>::value>>
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
      frame.resize(o.rows(), o.cols());
      deallocate();
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
  size_t length(const int i) const {
    return frame.length(i);
  }

  /**
   * Get the stride of the @p i th dimension.
   */
  size_t stride(const int i) const {
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
  Iterator<T,F> begin() {
    return Iterator<T,F>(ptr, frame);
  }

  /**
   * Iterator pointing to the first element.
   */
  Iterator<const T,F> begin() const {
    return Iterator<const T,F>(ptr, frame);
  }

  /**
   * Iterator pointing to one beyond the last element.
   */
  Iterator<T,F> end() {
    return begin() + frame.size();
  }

  /**
   * Iterator pointing to one beyond the last element.
   */
  Iterator<const T,F> end() const {
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
    ptr = (T*)std::malloc(sizeof(T)*frame.volume());
    assert(ptr);
  }

  /**
   * Deallocate memory of array.
   */
  void deallocate() {
    if (!isView) {
      std::free(ptr);
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
  template<class G>
  void copy(const Array<T,G>& o) {
    /* pre-condition */
    assert(o.frame.conforms(frame));

    if (frame.size() > 0) {
      auto iter1 = begin();
      auto end1 = end();
      auto iter2 = o.begin();
      auto end2 = o.end();

      for (; iter1 != end1; ++iter1, ++iter2) {
        new (&(*iter1)) T(*iter2);
      }
      assert(iter2 == end2);
    }
  }

  template<class U>
  void copy(const Sequence<U>& o) {
    assert(F::count() == sequence_depth<Sequence<U>>::value);

    size_t sizes[F::count()];
    frame.lengths(sizes);
    assert(sequence_conforms(sizes, o));
    auto iter = begin();
    sequence_copy(iter, o);
  }

  /**
   * Assign from another array.
   */
  template<class G>
  void assign(const Array<T,G>& o) {
    /* pre-condition */
    assert(o.frame.conforms(frame));

    if (frame.size() > 0) {
      auto iter1 = begin();
      auto end1 = end();

      auto iter2 = o.begin();
      auto end2 = o.end();

      //size_t block1 = frame.block();
      //size_t block2 = o.frame.block();
      //size_t block = gcd(block1, block2);

      //for (; iter1 != end1; iter1 += block, iter2 += block) {
      //  std::memmove(&(*iter1), &(*iter2), block * sizeof(T));
      //  // ^ memory regions may overlap, so avoid memcpy
      //}
      for (; iter1 != end1; ++iter1, ++iter2) {
        *iter1 = *iter2;
      }
      assert(iter2 == end2);
    }
  }

  template<class U>
  void assign(const Sequence<U>& o) {
    assert(F::count() == sequence_depth<Sequence<U>>::value);

    size_t sizes[F::count()];
    frame.lengths(sizes);
    assert(sequence_conforms(sizes, o));
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
   * Construct element of smart pointer type in place.
   *
   * @param o Element.
   * @param args Constructor arguments.
   */
  template<class U, class ... Args>
  static void emplace(SharedPointer<U>& o, Args ... args) {
    new (&o) SharedPointer<U>(new U(args...));
  }

  /**
   * Greatest common divisor of two positive integers.
   */
  static size_t gcd(const size_t a, const size_t b) {
    /* pre-condition */
    assert(a > 0);
    assert(b > 0);

    size_t a1 = a, b1 = b;
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
};
}
