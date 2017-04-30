/**
 * @file
 */
#pragma once

#include "bi/data/constant.hpp"
#include "bi/data/Span.hpp"
#include "bi/data/Index.hpp"
#include "bi/data/Range.hpp"
#include "bi/data/View.hpp"

#include <cstddef>

namespace bi {
/**
 * Empty frame.
 *
 * @ingroup library
 *
 * @see NonemptyFrame
 */
struct EmptyFrame {
  EmptyFrame() {
    //
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

  template<class Frame1>
  Frame1 prepend(const Frame1& o) const {
    return o;
  }

  int_t offset(const int_t n) {
    return 0;
  }

  int_t length(const int i) const {
    assert(false);
  }

  int_t stride(const int i) const {
    assert(false);
  }

  int_t lead(const int i) const {
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

  template<class T1>
  void leads(T1* out) const {
    //
  }

  static constexpr int count() {
    return 0;
  }

  static constexpr int_t size() {
    return 1;
  }

  static constexpr int_t volume() {
    return 1;
  }

  static constexpr int_t block() {
    return 1;
  }

  template<class View>
  int_t serial(const View& o) const {
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

  EmptyFrame& operator*=(const int_t n) {
    return *this;
  }

  EmptyFrame operator*(const int_t n) const {
    EmptyFrame result(*this);
    result *= n;
    return result;
  }
};

/**
 * Nonempty frame.
 *
 * @ingroup library
 *
 * @tparam Tail Frame type.
 * @tparam Head Span type.
 *
 * A frame describes the `D` dimensions of an array. It consists of a
 * @em tail frame describing the first `D - 1` dimensions, recursively, and a
 * @em head Span describing the remaining dimension. The tail frame is
 * EmptyFrame for the first dimension.
 */
template<class Tail, class Head>
struct NonemptyFrame {
  /**
   * Tail type.
   */
  typedef Tail tail_type;

  /**
   * Head type.
   */
  typedef Head head_type;

  /**
   * Generic constructor.
   */
  template<class Tail1, class Head1>
  NonemptyFrame(const Tail1 tail, const Head1 head) :
      tail(tail),
      head(head) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<class Tail1, class Head1>
  NonemptyFrame(const NonemptyFrame<Tail1,Head1>& o) :
      tail(o.tail),
      head(o.head) {
    //
  }

  /**
   * View operator.
   */
  template<class Tail1, int_t other_offset_value, int_t other_length_value,
      int_t other_stride_value>
  auto operator()(
      const NonemptyView<Tail1,
          Range<other_offset_value,other_length_value,other_stride_value>>& o) const {
    return NonemptyFrame<decltype(tail(o.tail)),decltype(head(o.head))>(
        tail(o.tail), head(o.head));
  }

  /**
   * View operator.
   */
  template<class Tail1, int_t other_offset_value>
  auto operator()(
      const NonemptyView<Tail1,Index<other_offset_value>>& o) const {
    return tail(o.tail) * head.lead;
  }

  /**
   * Does this frame conform to another? Two frames conform if their spans
   * conform.
   */
  bool conforms(const EmptyFrame& o) const {
    return false;
  }

  /**
   * Does this frame conform to another? Two frames conform if their spans
   * conform.
   */
  template<class Frame1>
  bool conforms(const Frame1& o) const {
    return tail.conforms(o.tail)/* && head.conforms(o.head)*/;
  }

  /**
   * Append dimensions from another frame to the right of this one.
   */
  template<class Frame1>
  auto prepend(const Frame1& o) const {
    auto tail = this->tail.prepend(o);
    return NonemptyFrame<decltype(tail),Head>(tail, head);
  }

  /**
   * Compute the offset to the @p n th element in storage order.
   *
   * @param n Element number.
   */
  int_t offset(const int_t n) {
    int_t q = n / head.length;
    int_t r = n % head.length;

    return tail.offset(q)*head.lead + r*head.stride;
  }

  /**
   * @name Queries
   */
  //@{
  /**
   * Get the length of the @p i th dimension.
   */
  int_t length(const int i) const {
    /* pre-condition */
    assert(i >= 0 && i < count());
    if (i == count() - 1) {
      return head.length;
    } else {
      return tail.length(i);
    }
  }

  /**
   * Get the stride of the @p i th dimension.
   */
  int_t stride(const int i) const {
    /* pre-condition */
    assert(i >= 0 && i < count());

    if (i == count() - 1) {
      return head.stride;
    } else {
      return tail.stride(i);
    }
  }

  /**
   * Get the lead of the @p i th dimension.
   */
  int_t lead(const int i) const {
    /* pre-condition */
    assert(i >= 0 && i < count());

    if (i == count() - 1) {
      return head.lead;
    } else {
      return tail.lead(i);
    }
  }
  //@}

  /**
   * @name Collections
   */
  //@{
  /**
   * Get lengths.
   *
   * @tparam T1 Integer type.
   *
   * @param[out] out Array assumed to have at least count() elements.
   */
  template<class T1>
  void lengths(T1* out) const {
    tail.lengths(out);
    *(out + Tail::count()) = head.length;
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
    tail.strides(out);
    *(out + Tail::count()) = head.stride;
  }

  /**
   * Get leads.
   *
   * @tparam T1 Integer type.
   *
   * @param[out] out Array assumed to have at least count() elements.
   */
  template<class T1>
  void leads(T1* out) const {
    tail.leads(out);
    *(out + Tail::count()) = head.lead;
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
    return Tail::count() + 1;
  }

  /**
   * Product of all lengths.
   */
  int_t size() const {
    return tail.size() * head.length;
  }

  /**
   * Product of all leads.
   */
  int_t volume() const {
    return tail.volume() * head.lead;
  }

  /**
   * Size of contiguous blocks.
   */
  int_t block() const {
    if (head.stride == 1) {
      if (head.lead == head.length) {
        return tail.block()*head.length;
      } else {
        return head.length;
      }
    } else {
      return 1;
    }
  }

  /**
   * Serial offset for a view.
   */
  template<class View>
  int_t serial(const View& o) const {
    return tail.serial(o.tail) * head.lead + head.stride * o.head.offset;
  }

  /**
   * Are all elements stored contiguously in memory?
   */
  bool contiguous() const {
    return size() == 1 || (count() == 1 && head.stride == 1)
        || volume() == size();
  }
  //@}

  /**
   * Equality operator.
   */
  template<class Head1, class Tail1>
  bool operator==(const NonemptyFrame<Head1,Tail1>& o) const {
    return tail == o.tail && head == o.head;
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
   * Multiply stride of rightmost dimension.
   *
   * @todo Take index instead of n.
   */
  NonemptyFrame<Tail,Head>& operator*=(const int_t n) {
    head *= n;
    return *this;
  }

  /**
   * Multiply stride of rightmost dimension.
   *
   * @todo Take index instead of n.
   */
  auto operator*(const int_t n) const {
    auto head = this->head * n;
    return NonemptyFrame<Tail,decltype(head)>(this->tail, head);
  }

  /**
   * Tail
   */
  Tail tail;

  /**
   * Head.
   */
  Head head;
};

/**
 * Default frame for `D` dimensions.
 */
template<int D>
struct DefaultFrame {
  typedef NonemptyFrame<typename DefaultFrame<D - 1>::type,Span<>> type;
};
template<>
struct DefaultFrame<0> {
  typedef EmptyFrame type;
};
}
