/**
 * @file
 */
#pragma once

#include "bi/data/Frame.hpp"

namespace bi {
/**
 * Array. Combines underlying data and a frame describing the shape of that
 * data. Allows the construction of views of the data, where a view indexes
 * either an individual element or some range of elements.
 *
 * @ingroup library
 *
 * @tparam Type Value type.
 * @tparam Frame Frame type.
 */
template<class Type, class Frame = EmptyFrame>
class Array {
public:
  /**
   * Constructor with new allocation.
   *
   * @tparam ...Args Arbitrary types.
   *
   * @param frame Frame.
   * @param args Optional constructor arguments.
   *
   * Memory is allocated for the array, and is freed on destruction.
   */
  template<class ... Args>
  Array(const Frame& frame, Args ... args) :
      frame(frame) {
    create(&ptr, frame);
    fill(ptr, frame, args...);
  }

  /**
   * Constructor with existing allocation.
   *
   * @tparam Frame Frame type.
   *
   * @param ptr Existing allocation.
   * @param frame Frame.
   */
  Array(Type* ptr, const Frame& frame) :
      frame(frame),
      ptr(ptr),
      own(false) {
    //
  }

  /**
   * Copy constructor.
   */
  Array(const Array<Type,Frame>& o) :
      frame(o.frame) {
    create(&ptr, frame);
    fill(ptr, frame);
    own = true;
    *this = o;
  }

  /**
   * Move constructor.
   */
  Array(Array<Type,Frame> && o) :
      frame(o.frame),
      ptr(o.ptr),
      own(o.own) {
    o.own = false;  // ownership moves
  }

  /**
   * Destructor.
   */
  ~Array() {
    if (own) {
      release(ptr, frame);
    }
  }

  /**
   * Copy assignment. The frames of the two arrays must conform.
   */
  Array<Type,Frame>& operator=(const Array<Type,Frame>& o) {
    /* pre-condition */
    assert(frame.conforms(o.frame));

    copy(*this, o);
    return *this;
  }

  /**
   * Move assignment. The frames of the two arrays must conform.
   */
  Array<Type,Frame>& operator=(Array<Type,Frame> && o) {
    /* pre-condition */
    assert(frame.conforms(o.frame));

    if (ptr == o.ptr) {
      /* just take ownership */
      if (!own) {
        std::swap(own, o.own);
      }
    } else {
      /* copy assignment */
      own = true;
      create(&ptr, frame);
      fill(ptr, frame);
      *this = o;
    }
    return *this;
  }

  /**
   * Generic assignment. The frames of the two arrays must conform.
   */
  template<class Type1, class Frame1>
  Array<Type,Frame>& operator=(const Array<Type1,Frame1>& o) {
    /* pre-condition */
    assert(frame.conforms(o.frame));

    copy(*this, o);
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
  template<class View1>
  auto& operator()(const View1& view) {
    return viewReturn(view, frame(view));
  }

  /**
   * View operator.
   */
  template<class View1>
  auto& operator()(const View1& view) const {
    return viewReturn(view, frame(view));
  }

  /**
   * @name Selections
   */
  //@{
  /**
   * Access the first element.
   */
  Type& front() {
    return *ptr;
  }

  /**
   * Access the first element.
   */
  const Type& front() const {
    return *ptr;
  }

  /**
   * Access the last element.
   */
  Type& back() {
    return ptr + frame.lead - 1;
  }

  /**
   * Access the last element.
   */
  const Type& back() const {
    return ptr + frame.lead - 1;
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

  /**
   * Get leads.
   *
   * @tparam Integer Integer type.
   *
   * @param[out] out Array assumed to have at least count() elements.
   */
  template<class Integer>
  void leads(Integer* out) const {
    frame.leads(out);
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
    return Frame::count();
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
   * the leftmost dimension is the fastest moving (for a matrix, this is
   * "column major" order).
   *
   * The idiom of iterator usage is as for the STL.
   */
  template<class View = typename DefaultView<Frame::count()>::type>
  auto begin(const View& view = typename DefaultView<Frame::count()>::type());

  /**
   * Iterator pointing to the first element.
   */
  template<class View = typename DefaultView<Frame::count()>::type>
  auto begin(const View& view =
      typename DefaultView<Frame::count()>::type()) const;

  /**
   * Iterator pointing to one beyond the last element.
   */
  template<class View = typename DefaultView<Frame::count()>::type>
  auto end(const View& view = typename DefaultView<Frame::count()>::type());

  /**
   * Iterator pointing to one beyond the last element.
   */
  template<class View = typename DefaultView<Frame::count()>::type>
  auto end(const View& view =
      typename DefaultView<Frame::count()>::type()) const;

protected:
  /**
   * Frame.
   */
  Frame frame;

  /**
   * Value.
   */
  Type* ptr;

  /**
   * Do we own the underlying buffer?
   */
  bool own;

  /**
   * Return value of view when result is an array.
   */
  template<class View1, class Frame1>
  auto viewReturn(const View1& view, const Frame1& frame) const {
    return Array<Type,Frame1>(ptr + frame.serial(view), frame);
  }

  /**
   * Return value of view when result is a scalar.
   */
  template<class View1>
  auto& viewReturn(const View1& view, const EmptyFrame& frame) const {
    return *(ptr + frame.serial(view));
  }
};

/**
 * Default array for `D` dimensions.
 */
template<class Type, int D>
using DefaultArray = Array<Type,typename DefaultFrame<D>::type>;
}

#include "bi/data/Iterator.hpp"

template<class Type, class Frame>
template<class View>
auto bi::Array<Type,Frame>::begin(const View& view) {
  return Iterator<Type,Frame,View>(ptr, frame, view);
}

template<class Type, class Frame>
template<class View>
auto bi::Array<Type,Frame>::begin(const View& view) const {
  return Iterator<const Type,Frame,View>(ptr, frame, view);
}

template<class Type, class Frame>
template<class View>
auto bi::Array<Type,Frame>::end(const View& view) {
  return Iterator<Type,Frame,View>(ptr, frame, view, frame.length);
}

template<class Type, class Frame>
template<class View>
auto bi::Array<Type,Frame>::end(const View& view) const {
  return Iterator<const Type,Frame,View>(ptr, frame, view, frame.length);
}
