/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/type.hpp"

namespace libbirch {
/**
 * Tuple.
 *
 * @tparam Head First element type.
 * @tparam ...Tail Remaining element types.
 *
 * Tuples do not support lvalue reference types. If all types are lvalue
 * reference types, Tie may be used instead.
 */
template<class Head, class ...Tail>
class Tuple {
  template<class Head1, class... Tail1> friend class Tuple;
  template<class Head1, class... Tail1> friend class Tie;

  static_assert(!std::is_lvalue_reference<Head>::value,
      "Tuple does not support lvalue reference types, try Tie instead.");
public:
  Tuple() = default;

  /**
   * Constructor.
   */
  template<IS_VALUE1(Head),IS_VALUE2(Tuple<Tail...>)>
  Tuple(const Head& head, const Tail&... tail) :
      head(head),
      tail(tail...) {
    //
  }

  /**
   * Constructor.
   */
  template<IS_VALUE1(Head),IS_NOT_VALUE2(Tuple<Tail...>)>
  Tuple(Label* context, const Head& head, const Tail&... tail) :
      head(head),
      tail(context, tail...) {
    //
  }

  /**
   * Constructor.
   */
  template<IS_NOT_VALUE1(Head),IS_VALUE2(Tuple<Tail...>)>
  Tuple(Label* context, const Head& head, const Tail&... tail) :
      head(context, head),
      tail(tail...) {
    //
  }

  /**
   * Constructor.
   */
  template<IS_NOT_VALUE1(Head),IS_NOT_VALUE2(Tuple<Tail...>)>
  Tuple(Label* context, const Head& head, const Tail&... tail) :
      head(context, head),
      tail(context, tail...) {
    //
  }

  /**
   * Copy constructor.
   */
  Tuple(const Tuple& o) = default;

  /**
   * Copy constructor.
   */
  template<class Head1, class... Tail1>
  Tuple(const Tuple<Head1,Tail1...>& o) :
      head(o.head),
      tail(o.tail) {
    //
  }

  /**
   * Copy constructor.
   */
  template<IS_VALUE1(Head),IS_NOT_VALUE2(Tuple<Tail...>), class Head1,
      class... Tail1>
  Tuple(Label* context, const Tuple<Head1,Tail1...>& o) :
      head(o.head),
      tail(context, o.tail) {
    //
  }

  /**
   * Copy constructor.
   */
  template<IS_NOT_VALUE1(Head),IS_VALUE2(Tuple<Tail...>), class Head1,
      class... Tail1>
  Tuple(Label* context, const Tuple<Head1,Tail1...>& o) :
      head(context, o.head),
      tail(o.tail) {
    //
  }

  /**
   * Copy constructor.
   */
  template<IS_NOT_VALUE1(Head),IS_NOT_VALUE2(Tuple<Tail...>), class Head1,
      class... Tail1>
  Tuple(Label* context, const Tuple<Head1,Tail1...>& o) :
      head(context, o.head),
      tail(context, o.tail) {
    //
  }

  /**
   * Deep copy constructor.
   */
  template<IS_VALUE1(Head),IS_NOT_VALUE2(Tuple<Tail...>)>
  Tuple(Label* context, Label* label, const Tuple& o) :
      head(o.head),
      tail(context, label, o.tail) {
    //
  }

  /**
   * Deep copy constructor.
   */
  template<IS_NOT_VALUE1(Head),IS_VALUE2(Tuple<Tail...>)>
  Tuple(Label* context, Label* label, const Tuple& o) :
      head(context, label, o.head),
      tail(o.tail) {
    //
  }

  /**
   * Deep copy constructor.
   */
  template<IS_NOT_VALUE1(Head),IS_NOT_VALUE2(Tuple<Tail...>)>
  Tuple(Label* context, Label* label, const Tuple& o) :
      head(context, label, o.head),
      tail(context, label, o.tail) {
    //
  }

  /**
   * Move constructor.
   */
  Tuple(Tuple&& o) = default;

  /**
   * Move constructor.
   */
  template<class Head1, class... Tail1>
  Tuple(Tuple<Head1,Tail1...>&& o) :
      head(std::move(o.head)),
      tail(std::move(o.tail)) {
    //
  }

  /**
   * Move constructor.
   */
  template<IS_VALUE1(Head),IS_NOT_VALUE2(Tuple<Tail...>), class Head1,
      class... Tail1>
  Tuple(Label* context, Tuple<Head1,Tail1...>&& o) :
      head(std::move(o.head)),
      tail(context, std::move(o.tail)) {
    //
  }

  /**
   * Move constructor.
   */
  template<IS_NOT_VALUE1(Head),IS_VALUE2(Tuple<Tail...>), class Head1,
      class... Tail1>
  Tuple(Label* context, Tuple<Head1,Tail1...>&& o) :
      head(context, std::move(o.head)),
      tail(std::move(o.tail)) {
    //
  }

  /**
   * Move constructor.
   */
  template<IS_NOT_VALUE1(Head),IS_NOT_VALUE2(Tuple<Tail...>), class Head1,
      class... Tail1>
  Tuple(Label* context, Tuple<Head1,Tail1...>&& o) :
      head(context, std::move(o.head)),
      tail(context, std::move(o.tail)) {
    //
  }

  /**
   * Copy assignment.
   */
  template<IS_VALUE1(Head),IS_VALUE2(Tuple<Tail...>)>
  Tuple& assign(const Tuple<Head,Tail...>& o) {
    head = o.head;
    tail = o.tail;
    return *this;
  }

  /**
   * Copy assignment.
   */
  template<IS_VALUE1(Head),IS_NOT_VALUE2(Tuple<Tail...>)>
  Tuple& assign(Label* context, const Tuple<Head,Tail...>& o) {
    head = o.head;
    tail.assign(context, o.tail);
    return *this;
  }

  /**
   * Copy assignment.
   */
  template<IS_NOT_VALUE1(Head),IS_VALUE2(Tuple<Tail...>)>
  Tuple& assign(Label* context, const Tuple<Head,Tail...>& o) {
    head.assign(context, o.head);
    tail = o.tail;
    return *this;
  }

  /**
   * Copy assignment.
   */
  template<IS_NOT_VALUE1(Head),IS_NOT_VALUE2(Tuple<Tail...>)>
  Tuple& assign(Label* context, const Tuple<Head,Tail...>& o) {
    head.assign(context, o.head);
    tail.assign(context, o.tail);
    return *this;
  }

  /**
   * Copy assignment operator.
   */
  Tuple& operator=(const Tuple<Head,Tail...>& o) {
    return assign(o);
  }

  /**
   * Copy assignment operator.
   */
  template<IS_VALUE1(Head),IS_VALUE2(Tuple<Tail...>)>
  Tuple& operator=(const Tuple<Head,Tail...>& o) {
    return assign(o);
  }

  /**
   * Move assignment.
   */
  template<IS_VALUE1(Head),IS_VALUE2(Tuple<Tail...>)>
  Tuple& assign(Tuple<Head,Tail...>&& o) {
    head = std::move(o.head);
    tail = std::move(o.tail);
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_VALUE1(Head),IS_NOT_VALUE2(Tuple<Tail...>)>
  Tuple& assign(Label* context, Tuple<Head,Tail...>&& o) {
    head = std::move(o.head);
    tail.assign(context, std::move(o.tail));
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_NOT_VALUE1(Head),IS_VALUE2(Tuple<Tail...>)>
  Tuple& assign(Label* context, Tuple<Head,Tail...>&& o) {
    head.assign(context, std::move(o.head));
    tail = std::move(o.tail);
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_NOT_VALUE1(Head),IS_NOT_VALUE2(Tuple<Tail...>)>
  Tuple& assign(Label* context, Tuple<Head,Tail...>&& o) {
    head.assign(context, std::move(o.head));
    tail.assign(context, std::move(o.tail));
    return *this;
  }

  /**
   * Move assignment operator.
   */
  Tuple& operator=(Tuple<Head,Tail...>&& o) {
    return assign(std::move(o));
  }

  /**
   * Move assignment operator.
   */
  template<IS_VALUE1(Head),IS_VALUE2(Tuple<Tail...>)>
  Tuple& operator=(Tuple<Head,Tail...>&& o) {
    return assign(std::move(o));
  }

  template<IS_VALUE1(Head)>
  void freeze() {
    tail.freeze();
  }

  template<IS_NOT_VALUE1(Head)>
  void freeze() {
    head.freeze();
    tail.freeze();
  }

  template<IS_VALUE1(Head)>
  void thaw(Label* label) {
    tail.thaw(label);
  }

  template<IS_NOT_VALUE1(Head)>
  void thaw(Label* label) {
    head.thaw(label);
    tail.thaw(label);
  }

  template<IS_VALUE1(Head)>
  void finish() {
    tail.finish();
  }

  template<IS_NOT_VALUE1(Head)>
  void finish() {
    head.finish();
    tail.finish();
  }

private:
  /**
   * First element.
   */
  Head head;

  /**
   * Remaining elements.
   */
  Tuple<Tail...> tail;
};

/*
 * Tuple with a single element.
 *
 * @tparam Head Element type.
 */
template<class Head>
class Tuple<Head> {
  template<class Head1, class... Tail1> friend class Tuple;
  template<class Head1, class... Tail1> friend class Tie;

  static_assert(!std::is_lvalue_reference<Head>::value,
      "Tuple does not support lvalue reference types, try Tie instead.");
public:
  Tuple() = default;

  /**
   * Constructor.
   */
  template<IS_VALUE(Head)>
  Tuple(const Head& head) :
      head(head) {
    //
  }

  /**
   * Constructor.
   */
  template<IS_NOT_VALUE(Head)>
  Tuple(Label* context, const Head& head) :
      head(context, head) {
    //
  }

  /**
   * Copy constructor.
   */
  Tuple(const Tuple& o) :
      head(o.head) {
    //
  }

  /**
   * Copy constructor.
   */
  template<class Head1>
  Tuple(const Tuple<Head1>& o) :
      head(o.head) {
    //
  }

  /**
   * Copy constructor.
   */
  template<IS_NOT_VALUE(Head), class Head1>
  Tuple(Label* context, const Tuple<Head1>& o) :
      head(context, o.head) {
    //
  }

  /**
   * Move constructor.
   */
  Tuple(Tuple&& o) :
      head(std::move(o.head)) {
    //
  }

  /**
   * Move constructor.
   */
  template<class Head1>
  Tuple(Tuple<Head1>&& o) :
      head(std::move(o.head)) {
    //
  }

  /**
   * Move constructor.
   */
  template<IS_NOT_VALUE(Head), class Head1>
  Tuple(Label* context, Tuple<Head1>&& o) :
      head(context, std::move(o.head)) {
    //
  }

  /**
   * Deep copy constructor.
   */
  template<IS_NOT_VALUE(Head)>
  Tuple(Label* context, Label* label, const Tuple& o) :
      head(context, label, o.head) {
    //
  }

  /**
   * Copy assignment.
   */
  template<IS_VALUE(Head)>
  Tuple& assign(const Tuple<Head>& o) {
    head = o.head;
    return *this;
  }

  /**
   * Copy assignment.
   */
  template<IS_NOT_VALUE(Head)>
  Tuple& assign(Label* context, const Tuple<Head>& o) {
    head.assign(context, o.head);
    return *this;
  }

  /**
   * Copy assignment operator.
   */
  template<IS_VALUE(Head)>
  Tuple& operator=(const Tuple<Head>& o) {
    return assign(o);
  }

  /**
   * Move assignment.
   */
  template<IS_VALUE(Head)>
  Tuple& assign(Tuple<Head>&& o) {
    head = std::move(o.head);
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_NOT_VALUE(Head)>
  Tuple& assign(Label* context, Tuple<Head>&& o) {
    head.assign(context, std::move(o.head));
    return *this;
  }

  /**
   * Move assignment operator.
   */
  template<IS_VALUE(Head)>
  Tuple& operator=(Tuple<Head>&& o) {
    return assign(std::move(o));
  }

  template<IS_VALUE(Head)>
  void freeze() {
    //
  }

  template<IS_NOT_VALUE(Head)>
  void freeze() {
    head.freeze();
  }

  template<IS_VALUE(Head)>
  void thaw(Label* label) {
    //
  }

  template<IS_NOT_VALUE(Head)>
  void thaw(Label* label) {
    head.thaw(label);
  }

  template<IS_VALUE(Head)>
  void finish() {
    //
  }

  template<IS_NOT_VALUE(Head)>
  void finish() {
    head.finish();
  }

private:
  /**
   * First element.
   */
  Head head;
};

template<class Head, class ...Tail>
struct is_value<Tuple<Head,Tail...>> {
  static const bool value = is_value<Head>::value && is_value<Tuple<Tail...>>::value;
};
template<class Head>
struct is_value<Tuple<Head>> {
  static const bool value = is_value<Head>::value;
};
}
