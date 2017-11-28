/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"
#include "libbirch/Span.hpp"
#include "libbirch/Index.hpp"
#include "libbirch/Range.hpp"
#include "libbirch/View.hpp"
#include "libbirch/Eigen.hpp"

#include <cstddef>

namespace bi {
/**
 * Empty frame.
 *
 * @ingroup libbirch
 *
 * @see NonemptyFrame
 */
struct EmptyFrame {
  EmptyFrame() {
    //
  }

  /**
   * Special constructor for Eigen integration where all matrices and vectors
   * are treated as matrices, with row and column counts. If this constructor,
   * is reached, it means the array is a vector, and @p length should be one
   * (and is checked for this).
   */
  EmptyFrame(const Eigen::Index cols) {
    assert(cols == 1);
  }

  EmptyFrame operator()(const EmptyView& o) const {
    return EmptyFrame();
  }

  bool conforms(const EmptyFrame& o) const {
    return true;
  }

  template<class Frame1>
  bool conforms(const Frame1& o) const {
    return false;
  }

  bool conforms(const Eigen::Index rows, const Eigen::Index cols) {
    return false;
  }

  bool conforms(const Eigen::Index rows) {
    return false;
  }

  void resize(const EmptyFrame& o) {
    //
  }

  template<class T1>
  void resize(T1* in) {
    //
  }

  void resize(const Eigen::Index cols) {
    // special case for vector represented as N x 1 matrix in Eigen.
    assert(cols == 1);
  }

  ptrdiff_t offset(const ptrdiff_t n) {
    return 0;
  }

  size_t length(const int i) const {
    assert(false);
  }

  size_t stride(const int i) const {
    assert(false);
  }

  template<class T1>
  void lengths(T1* out) const {
    //
  }

  template<class T1>
  void strides(T1* out) const {
    //
  }

  static constexpr int count() {
    return 0;
  }

  static constexpr size_t size() {
    return 1;
  }

  static constexpr size_t volume() {
    return 1;
  }

  static constexpr size_t block() {
    return 1;
  }

  template<class View>
  ptrdiff_t serial(const View& o) const {
    return 0;
  }

  bool contiguous() const {
    return true;
  }

  bool operator==(const EmptyFrame& o) const {
    return true;
  }

  template<class Frame1>
  bool operator==(const Frame1& o) const {
    return false;
  }

  template<class Frame1>
  bool operator!=(const Frame1& o) const {
    return !operator==(o);
  }

  EmptyFrame& operator*=(const ptrdiff_t n) {
    return *this;
  }

  EmptyFrame operator*(const ptrdiff_t n) const {
    EmptyFrame result(*this);
    result *= n;
    return result;
  }
};

/**
 * Nonempty frame.
 *
 * @ingroup libbirch
 *
 * @tparam Tail Frame type.
 * @tparam Head Span type.
 *
 * A frame describes the `D` dimensions of an array. It consists of a
 * @em head span describing the first dimension, and a tail @em tail frame
 * describing the remaining `D - 1` dimensions, recursively. The tail frame
 * is EmptyFrame for the last dimension.
 */
template<class Head, class Tail>
struct NonemptyFrame {
  /**
   * Head type.
   */
  typedef Head head_type;

  /**
   * Tail type.
   */
  typedef Tail tail_type;

  /**
   * Default constructor (for zero-size frame).
   */
  NonemptyFrame() {
    //
  }

  /*
   * Special constructor for Eigen integration where all matrices and vectors
   * are treated as matrices, with row and column counts.
   */
  NonemptyFrame(const Eigen::Index rows, const Eigen::Index cols = 1) :
      head(rows, cols),
      tail(cols) {
    //
  }

  /**
   * Generic constructor.
   */
  template<class Head1, class Tail1>
  NonemptyFrame(const Head1 head, const Tail1 tail) :
      head(head),
      tail(tail) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<class Head1, class Tail1>
  NonemptyFrame(const NonemptyFrame<Head1,Tail1>& o) :
      head(o.head),
      tail(o.tail) {
    //
  }

  /**
   * View operator.
   */
  template<ptrdiff_t offset_value1, size_t length_value1, class Tail1>
  auto operator()(
      const NonemptyView<Range<offset_value1,length_value1>,Tail1>& o) const {
    /* pre-conditions */
    assert(o.head.offset >= 0);
    assert(o.head.offset + o.head.length <= head.length);

    return NonemptyFrame<decltype(head(o.head)),decltype(tail(o.tail))>(
        head(o.head), tail(o.tail));
  }

  /**
   * View operator.
   */
  template<ptrdiff_t offset_value1, class Tail1>
  auto operator()(const NonemptyView<Index<offset_value1>,Tail1>& o) const {
    /* pre-condition */
    assert(o.head.offset >= 0 && o.head.offset < head.length);

    return o.head.offset * head.stride + tail(o.tail);
  }

  /**
   * Does this frame conform to another? Two frames conform if their spans
   * conform.
   */
  bool conforms(const EmptyFrame& o) const {
    return false;
  }
  template<class Frame1>
  bool conforms(const Frame1& o) const {
    return head.conforms(o.head) && tail.conforms(o.tail);
  }
  bool conforms(const Eigen::Index rows, const Eigen::Index cols) {
    return head.conforms(rows) && tail.conforms(cols);
  }
  bool conforms(const Eigen::Index rows) {
    return head.conforms(rows);
  }

  /**
   * Resize this frame to conform to another.
   */
  template<class Frame1>
  void resize(const Frame1& o) {
    tail.resize(o.tail);
    head.length = o.head.length;
    head.stride = tail.volume();
  }
  template<class T1>
  void resize(T1* in) {
    tail.resize(in + 1);
    head.length = *in;
    head.stride = tail.volume();
  }
  void resize(const Eigen::Index rows, const Eigen::Index cols) {
    tail.resize(cols);
    head.length = rows;
    head.stride = tail.volume();
  }
  void resize(const Eigen::Index rows) {
    head.length = rows;
    head.stride = tail.volume();
  }

  /**
   * Compute the offset to the @p n th element in storage order.
   *
   * @param n Element number.
   */
  ptrdiff_t offset(const ptrdiff_t n) {
    ptrdiff_t q = n / head.length;
    ptrdiff_t r = n % head.length;

    return r * head.stride + tail.offset(q);
  }

  /**
   * @name Getters
   */
  //@{
  /**
   * Get the length of the @p i th dimension.
   */
  size_t length(const int i) const {
    /* pre-condition */
    assert(i >= 0 && i < count());

    if (i == 0) {
      return head.length;
    } else {
      return tail.length(i - 1);
    }
  }

  /**
   * Get the stride of the @p i th dimension.
   */
  size_t stride(const int i) const {
    /* pre-condition */
    assert(i >= 0 && i < count());

    if (i == 0) {
      return head.stride;
    } else {
      return tail.stride(i - 1);
    }
  }

  /**
   * Get lengths.
   *
   * @tparam T1 Integer type.
   *
   * @param[out] out Array assumed to have at least count() elements.
   */
  template<class T1>
  void lengths(T1* out) const {
    *out = head.length;
    tail.lengths(out + 1);
  }

  /**
   * Get strides.
   *
   * @tparam T1 Integer type.
   *
   * @param[out] out Array assumed to have at least count() elements.
   */
  template<class T1>
  void strides(T1* out) const {
    *out = head.stride;
    tail.strides(out + 1);
  }
  //@}

  /**
   * @name Reductions
   */
  //@{
  /**
   * Number of dimensions.
   */
  static constexpr int count() {
    return 1 + Tail::count();
  }

  /**
   * Product of all lengths.
   */
  size_t size() const {
    return head.length*tail.size();
  }

  /**
   * Product of all strides.
   */
  size_t volume() const {
    return head.length*head.stride;
  }

  /**
   * Size of contiguous blocks.
   */
  size_t block() const {
    size_t block = tail.block();
    return head.stride == block ? head.length*head.stride : block;
  }

  /**
   * Serial offset for a view.
   */
  template<class View>
  ptrdiff_t serial(const View& o) const {
    return o.head.offset * head.stride + tail.serial(o.tail);
  }

  /**
   * Are all elements stored contiguously in memory?
   */
  bool contiguous() const {
    return volume() == size();
  }
  //@}

  /**
   * Equality operator.
   */
  template<class Head1, class Tail1>
  bool operator==(const NonemptyFrame<Head1,Tail1>& o) const {
    return head == o.head && tail == o.tail;
  }

  /**
   * Equality operator.
   */
  bool operator==(const EmptyFrame& o) const {
    return false;
  }

  /**
   * Unequality operator.
   */
  template<class Frame1>
  bool operator!=(const Frame1& o) const {
    return !(*this == o);
  }

  /**
   * Head.
   */
  Head head;

  /**
   * Tail
   */
  Tail tail;
};

/**
 * Default frame for `D` dimensions.
 */
template<int D>
struct DefaultFrame {
  typedef NonemptyFrame<Span<>,typename DefaultFrame<D - 1>::type> type;
};
template<>
struct DefaultFrame<0> {
  typedef EmptyFrame type;
};
}
