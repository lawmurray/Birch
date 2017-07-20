/**
 * @file
 *
 * Convenience functions for creating spans, ranges, frames, views, arrays,
 * objects, etc.
 */
#include "bi/lib/Span.hpp"
#include "bi/lib/Index.hpp"
#include "bi/lib/Range.hpp"
#include "bi/lib/Frame.hpp"
#include "bi/lib/View.hpp"
#include "bi/lib/Array.hpp"

#include <gc.h>

namespace bi {
/**
 * Make a span.
 *
 * @ingroup library
 *
 * @param length Length.
 */
inline auto make_span(const size_t length) {
  return Span<mutable_value,1,mutable_value>(length);
}

/**
 * Make an index.
 *
 * @ingroup library
 *
 * @param index Index.
 */
inline auto make_index(const ptrdiff_t i) {
  return Index<mutable_value>(i);
}

/**
 * Make a range.
 *
 * @ingroup library
 *
 * @param start First index.
 * @param end Last index.
 */
inline auto make_range(const ptrdiff_t start, const ptrdiff_t end) {
  return Range<mutable_value,mutable_value,1>(start, end - start + 1);
}

/**
 * Make a frame.
 *
 * @ingroup library
 */
//@{
/*
 * No arguments.
 */
inline auto make_frame() {
  return EmptyFrame();
}

/*
 * Single argument.
 */
template<size_t length_value, ptrdiff_t stride_value, size_t lead_value>
auto make_frame(const Span<length_value,stride_value,lead_value>& arg) {
  auto tail = make_frame();
  auto head = arg;
  return NonemptyFrame<decltype(tail),decltype(head)>(tail, head);
}

inline auto make_frame(const size_t arg) {
  return make_frame(make_span(arg));
}

/*
 * Multiple arguments.
 */
template<size_t length_value, ptrdiff_t stride_value, size_t lead_value,
    class ... Args>
auto make_frame(const Span<length_value,stride_value,lead_value>& arg,
    Args ... args) {
  return make_frame(make_frame(arg), args...);
}

template<class ... Args>
auto make_frame(const size_t arg, Args ... args) {
  return make_frame(make_frame(arg), args...);
}

/*
 * Tail plus single argument.
 */
template<class Tail, class Head, size_t length_value, ptrdiff_t stride_value,
size_t lead_value>
auto make_frame(const NonemptyFrame<Tail,Head>& frame,
    const Span<length_value,stride_value,lead_value>& arg) {
  auto tail = frame;
  auto head = arg;
  return NonemptyFrame<decltype(tail),decltype(head)>(tail, head);
}

template<class Tail, class Head>
auto make_frame(const NonemptyFrame<Tail,Head>& frame, const size_t arg) {
  return make_frame(frame, make_span(arg));
}

/*
 * Tail plus multiple arguments.
 */
template<class Tail, class Head, class Arg, class ... Args>
auto make_frame(const NonemptyFrame<Tail,Head>& tail, const Arg& arg,
    Args ... args) {
  return make_frame(make_frame(tail, arg), args...);
}

template<class Tail, class Head, class ... Args>
auto make_frame(const NonemptyFrame<Tail,Head>& tail, const size_t arg,
    Args ... args) {
  return make_frame(make_frame(tail, make_span(arg)), args...);
}
//@}

/**
 * Make a view.
 *
 * @ingroup library
 */
//@{
/*
 * No arguments.
 */
inline auto make_view() {
  return EmptyView();
}

/*
 * Single argument.
 */
template<ptrdiff_t offset_value, size_t length_value, ptrdiff_t stride_value>
auto make_view(const Range<offset_value,length_value,stride_value>& arg) {
  auto tail = make_view();
  auto head = arg;
  return NonemptyView<decltype(tail),decltype(head)>(tail, head);
}

template<ptrdiff_t offset_value>
auto make_view(const Index<offset_value>& arg) {
  auto tail = make_view();
  auto head = arg;
  return NonemptyView<decltype(tail),decltype(head)>(tail, head);
}

inline auto make_view(const ptrdiff_t arg) {
  return make_view(make_index(arg));
}

/*
 * Multiple arguments.
 */
template<ptrdiff_t offset_value, size_t length_value, ptrdiff_t stride_value,
    class ... Args>
auto make_view(const Range<offset_value,length_value,stride_value>& arg,
    Args ... args) {
  return make_view(make_view(arg), args...);
}

template<ptrdiff_t offset_value, class ... Args>
auto make_view(const Index<offset_value>& arg, Args ... args) {
  return make_view(make_view(arg), args...);
}

template<class ... Args>
auto make_view(const ptrdiff_t arg, Args ... args) {
  return make_view(make_index(arg), args...);
}

/*
 * Tail plus single argument.
 */
template<class Tail, class Head, ptrdiff_t offset_value, size_t length_value,
    ptrdiff_t stride_value>
auto make_view(const NonemptyView<Tail,Head>& view,
    const Range<offset_value,length_value,stride_value>& arg) {
  auto tail = view;
  auto head = arg;
  return NonemptyView<decltype(tail),decltype(head)>(tail, head);
}

template<class Tail, class Head, ptrdiff_t offset_value>
auto make_view(const NonemptyView<Tail,Head>& view,
    const Index<offset_value>& arg) {
  auto tail = view;
  auto head = arg;
  return NonemptyView<decltype(tail),decltype(head)>(tail, head);
}

template<class Tail, class Head>
auto make_view(const NonemptyView<Tail,Head>& view, const ptrdiff_t arg) {
  return make_view(view, make_index(arg));
}

/*
 * Tail plus multiple arguments.
 */
template<class Tail, class Head, class Arg, class ... Args>
auto make_view(const NonemptyView<Tail,Head>& tail, Arg arg, Args ... args) {
  return make_view(make_view(tail, arg), args...);
}

/**
 * Make an array.
 *
 * @ingroup library
 *
 * @tparam Type Value type.
 * @tparam Frame Frame type.
 *
 * @param frame Frame.
 *
 * @return The array.
 */
template<class Type, class Frame = EmptyFrame>
auto make_array(const Frame& frame = EmptyFrame()) {
  return Array<Type,Frame>(frame);
}

/**
 * Make an object.
 *
 * @ingroup library
 *
 * @tparam Type Value type.
 * @tparam Args Argument types.
 *
 * @param args Constructor arguments.
 *
 * @return Pointer to the object.
 */
template<class Type, class... Args>
Pointer<Type> make_object(Args... args) {
  auto raw = new (GC_MALLOC(sizeof(Type))) Type(args...);
  return Pointer<Type>(raw);
}
}
