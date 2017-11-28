/**
 * @file
 */
#pragma once

#include "libbirch/Frame.hpp"
#include "libbirch/Iterator.hpp"
#include "libbirch/Pointer.hpp"
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
      ptr(allocate(frame.volume())),
      isView(false) {
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
      ptr(allocate(frame.volume())),
      isView(false) {
    initialize(args...);
  }

  /**
   * Copy constructor.
   */
  Array(const Array<T,F>& o) :
      frame(o.frame),
      ptr(allocate(frame.volume())),
      isView(false) {
    copy(o);
  }

  /**
   * Move constructor.
   */
  Array(Array<T,F> && o) = default;

  /**
   * Sequence constructor.
   *
   * @param o Sequence.
   */
  template<class U>
  Array(const Sequence<U>& o) :
      frame(sequence_frame(o)),
      ptr(allocate(frame.volume())),
      isView(false) {
    copy(o);
  }

  /**
   * Copy assignment. For a view the frames of the two arrays must conform,
   * otherwise a resize is permitted.
   */
  Array<T,F>& operator=(const Array<T,F>& o) {
    if (!isView && !frame.conforms(o.frame)) {
      frame.resize(o.frame);
      ptr = allocate(frame.volume());
    }
    copy(o);
    return *this;
  }

  /**
   * Move assignment. The frames of the two arrays must conform.
   */
  Array<T,F>& operator=(Array<T,F> && o) {
    if (!isView) {
      if (!o.isView) {
        /* move */
        frame = o.frame;
        ptr = o.ptr;
      } else {
        if (!frame.conforms(o.frame)) {
          /* resize */
          frame.resize(o.frame);
          ptr = allocate(frame.volume());
        }
        copy(o);
      }
    } else {
      copy(o);
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
    copy(o);
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
  auto operator()(const View1& view) const {
    return Array<T,decltype(frame(view))>(ptr + frame.serial(view),
        frame(view));
  }
  template<class View1, typename = std::enable_if_t<View1::rangeCount() == 0>>
  auto& operator()(const View1& view) {
    return *(ptr + frame.serial(view));
  }
  template<class View1, typename = std::enable_if_t<View1::rangeCount() == 0>>
  const auto& operator()(const View1& view) const {
    return *(ptr + frame.serial(view));
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
      is_eigen_compatible<DerivedType>::value>>Array(const Eigen::EigenBase<DerivedType>& o, const F& frame) :
  frame(frame),
  ptr(allocate(frame.volume())),
  isView(false) {
    toEigen() = o;
  }

  /**
   * Construct from Eigen Matrix expression.
   */
  template<class DerivedType, typename = std::enable_if_t<is_eigen_compatible<DerivedType>::value>>
  Array(const Eigen::EigenBase<DerivedType>& o) :
  frame(o.rows(), o.cols()),
  ptr(allocate(frame.volume())),
  isView(false) {
    toEigen() = o;
  }

  /**
   * Assign from Eigen Matrix expression.
   */
  template<class DerivedType, typename = std::enable_if_t<is_eigen_compatible<DerivedType>::value>>
  Array<T,F>& operator=(const Eigen::EigenBase<DerivedType>& o) {
    if (!isView && !frame.conforms(o.rows(), o.cols())) {
      frame.resize(o.rows(), o.cols());
      ptr = allocate(frame.volume());
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
  //@}

  /**
   * @name Collections
   */
  //@{
  /**
   * Get lengths.
   *
   * @tparam Integer Integer type.
   *
   * @param[out] out Array assumed to have at least count() elements.
   */
  template<class Integer>
  void lengths(Integer* out) const {
    frame.lengths(out);
  }

  /**
   * Get strides.
   *
   * @tparam Integer Integer type.
   *
   * @param[out] out Array assumed to have at least count() elements.
   */
  template<class Integer>
  void strides(Integer* out) const {
    frame.strides(out);
  }
  //@}

  /**
   * @name Reductions
   */
  //@{
  /**
   * Number of spans in the frame.
   */
  static constexpr int count() {
    return F::count();
  }

  /**
   * Are all elements stored contiguously in memory?
   */
  bool contiguous() const {
    return frame.contiguous();
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
   *
   * @tparam U Element type.
   *
   * @param size Number of elements to allocate.
   */
  static T* allocate(const size_t n) {
    T* raw = (T*)GC_MALLOC(sizeof(T) * n);
    assert(raw);
    return raw;
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
  static void emplace(Pointer<U>& o, Args ... args) {
    auto raw = new (GC) U(args...);
    new (&o) Pointer<U>(raw);
  }

  /**
   * Copy from another array.
   */
  template<class G>
  void copy(const Array<T,G>& o) {
    /* pre-condition */
    assert(o.frame.conforms(frame));

    if (frame.size() > 0) {
      size_t block1 = frame.block();
      auto iter1 = begin();
      auto end1 = end();

      size_t block2 = o.frame.block();
      auto iter2 = o.begin();
      auto end2 = o.end();

      size_t block = gcd(block1, block2);
      for (; iter1 != end1; iter1 += block, iter2 += block) {
        std::memmove(&(*iter1), &(*iter2), block * sizeof(T));
        // ^ memory regions may overlap, so avoid memcpy
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
    sequence_copy(o, iter);
    assert(iter == end());
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

/**
 * Default array for `D` dimensions.
 */
template<class T, int D>
using DefaultArray = Array<T,typename DefaultFrame<D>::type>;
}
