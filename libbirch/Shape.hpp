/**
 * @file
 */
#pragma once

#include "libbirch/Span.hpp"
#include "libbirch/Index.hpp"
#include "libbirch/Range.hpp"
#include "libbirch/View.hpp"
#include "libbirch/Eigen.hpp"

namespace libbirch {
/**
 * Empty shape.
 *
 * @ingroup libbirch
 *
 * @see Shape
 */
struct EmptyShape {
  EmptyShape() = default;

  /**
   * Special constructor for Eigen integration where all matrices and vectors
   * are treated as matrices, with row and column counts. If this constructor,
   * is reached, it means the array is a vector, and @p length should be one
   * (and is checked for this).
   */
  explicit EmptyShape(const Eigen::Index cols) {
    assert(cols == 1);
  }

  EmptyShape operator()(const EmptyView& o) const {
    return EmptyShape();
  }

  EmptyShape compact() const {
    return EmptyShape();
  }

  bool conforms(const EmptyShape& o) const {
    return true;
  }

  template<class G>
  bool conforms(const G& o) const {
    return false;
  }

  bool conforms(const Eigen::Index rows, const Eigen::Index cols) {
    return false;
  }

  bool conforms(const Eigen::Index rows) {
    return false;
  }

  int64_t offset(const int64_t n) const {
    return 0;
  }

  int64_t length(const int i) const {
    assert(false);
    return 0;
  }

  int64_t stride(const int i) const {
    assert(false);
    return 0;
  }

  static constexpr int count() {
    return 0;
  }

  static constexpr int64_t size() {
    return 1;
  }

  static constexpr int64_t volume() {
    return 1;
  }

  template<class V>
  int64_t serial(const V& o) const {
    return 0;
  }
};

/**
 * Shape of an array.
 *
 * @ingroup libbirch
 *
 * @tparam Head Span type.
 * @tparam Tail Shape type.
 *
 * A shape describes the `D` dimensions of an array. It consists of a
 * @em head span describing the first dimension, and a tail @em tail shape
 * describing the remaining `D - 1` dimensions, recursively. The tail shape
 * is EmptyShape for the last dimension.
 */
template<class Head, class Tail>
struct Shape {
  /**
   * Default constructor (for zero-size shape).
   */
  Shape() {
    //
  }

  /*
   * Special constructor for Eigen integration where all matrices and vectors
   * are treated as matrices, with row and column counts.
   */
  explicit Shape(const Eigen::Index rows, const Eigen::Index cols = 1) :
      head(rows, cols),
      tail(cols) {
    //
  }

  /**
   * Generic constructor.
   */
  template<class Head1, class Tail1>
  explicit Shape(const Head1 head, const Tail1 tail) :
      head(head),
      tail(tail) {
    //
  }

  /**
   * Copy constructor.
   */
  Shape(const Shape<Head,Tail>& o) = default;

  /**
   * Generic copy constructor.
   */
  template<class Head1, class Tail1>
  Shape(const Shape<Head1,Tail1>& o) :
      head(o.head),
      tail(o.tail) {
    //
  }

  /**
   * View operator.
   */
  template<int64_t offset_value1, int64_t length_value1, class Tail1>
  auto operator()(
      const View<Range<offset_value1,length_value1>,Tail1>& o) const {
    /* pre-conditions */
    libbirch_assert_msg_(
        o.head.offset >= 0 && o.head.offset + o.head.length <= head.length,
        "range is " << (o.head.offset + 1) << ".."
            << (o.head.offset + o.head.length) << " for dimension of length "
            << head.length);

    return Shape<decltype(head(o.head)),decltype(tail(o.tail))>(
        head(o.head), tail(o.tail));
  }

  /**
   * View operator.
   */
  template<int64_t offset_value1, class Tail1>
  auto operator()(const View<Index<offset_value1>,Tail1>& o) const {
    /* pre-condition */
    libbirch_assert_msg_(o.head.offset >= 0 && o.head.offset < head.length,
        "index is " << (o.head.offset + 1) << " for dimension of length "
            << head.length);

    return tail(o.tail);
  }

  /**
   * Compact the shape to produce a new shape of the same size, but with
   * contiguous storage.
   */
  Shape<Head,Tail> compact() const {
    auto tail = this->tail.compact();
    auto head = Head(this->head.length, tail.size());
    return Shape<Head,Tail>(head, tail);
  }

  /**
   * Does this shape conform to another? Two shapes conform if their spans
   * conform.
   */
  bool conforms(const EmptyShape& o) const {
    return false;
  }
  template<class G>
  bool conforms(const G& o) const {
    return head.conforms(o.head) && tail.conforms(o.tail);
  }
  bool conforms(const Eigen::Index rows, const Eigen::Index cols) {
    return head.conforms(rows)
        && (tail.conforms(cols)
            || (std::is_same<Tail,EmptyShape>::value && cols == 1));
  }
  bool conforms(const Eigen::Index rows) {
    return head.conforms(rows);
  }

  /**
   * Compute the offset to the @p n th element in storage order.
   *
   * @param n Element number.
   */
  int64_t offset(const int64_t n) const {
    int64_t s = tail.size();
    if (s > 0) {
      int64_t q = n / s;
      int64_t r = n % s;
      return q * head.stride + tail.offset(r);
    } else {
      return 0;
    }
  }

  /**
   * @name Getters
   */
  //@{
  /**
   * Get the length of the @p i th dimension.
   */
  int64_t length(const int i) const {
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
  int64_t stride(const int i) const {
    /* pre-condition */
    assert(i >= 0 && i < count());

    if (i == 0) {
      return head.stride;
    } else {
      return tail.stride(i - 1);
    }
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
  int64_t size() const {
    return head.length * tail.size();
  }

  /**
   * Product of all strides.
   */
  int64_t volume() const {
    return head.length * head.stride;
  }

  /**
   * Serial offset for a view.
   */
  template<class V>
  int64_t serial(const V& o) const {
    libbirch_assert_msg_(o.head.offset >= 0 && o.head.offset < head.length,
        "index is " << (o.head.offset + 1) << " for dimension of length "
            << head.length);
    return o.head.offset * head.stride + tail.serial(o.tail);
  }
  //@}

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
 * Default shape for `D` dimensions.
 */
template<int D>
struct DefaultShape {
  typedef Shape<Span<>,typename DefaultShape<D - 1>::type> type;
};
template<>
struct DefaultShape<0> {
  typedef EmptyShape type;
};
}
