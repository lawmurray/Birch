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
 * @ingroup libbirch
 */
template<class Head, class Enable = void, class... Tail>
class TupleBase {
  //
};

/**
 * TupleBase with a value head type.
 *
 * @tparam Head Head type.
 * @tparam Tail Tail types.
 */
template<class Head, class ...Tail>
class TupleBase<Head,IS_VALUE(Head),Tail...> {
  template<class Head1, class Enable1, class... Tail1> friend class TupleBase;
public:
  TupleBase() = default;
  TupleBase(const TupleBase&) = default;
  TupleBase(TupleBase&&) = default;
  TupleBase& operator=(const TupleBase&) = delete;
  TupleBase& operator=(TupleBase&&) = delete;

  /**
   * Constructor.
   */
  TupleBase(Head head, Tail... tail) :
      head(std::forward(head)),
      tail(std::forward(tail...)) {
    //
  }

  /**
   * Constructor.
   */
  TupleBase(Label* context, Head head, Tail... tail) :
      head(std::forward(head)),
      tail(context, std::forward(tail...)) {
    //
  }

  /**
   * Copy constructor.
   */
  TupleBase(Label* context, const TupleBase& o) :
      head(o.head),
      tail(context, o.tail) {
    //
  }

  /**
   * Move constructor.
   */
  TupleBase(Label* context, TupleBase&& o) :
      head(std::move(o.head)),
      tail(context, std::move(o.tail)) {
    //
  }

  /**
   * Deep copy constructor.
   */
  TupleBase(Label* context, Label* label, const TupleBase& o) :
      head(o.head),
      tail(context, label, o.tail) {
    //
  }

  /**
   * Copy assignment.
   */
  template<class Head1, class ... Tail1>
  TupleBase& assign(Label* context, const TupleBase<Head1,Tail1...>& o) {
    head = o.head;
    tail.assign(context, o.tail);
    return *this;
  }

  /**
   * Move assignment.
   */
  template<class Head1, class ... Tail1>
  TupleBase& assign(Label* context, TupleBase<Head1,Tail1...>&& o) {
    head = std::move(o.head);
    tail.assign(context, std::move(o.tail));
    return *this;
  }

  void freeze() {
    tail.freeze();
  }

  void thaw(Label* label) {
    tail.thaw(label);
  }

  void finish() {
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
  TupleBase<Tail...> tail;
};

/**
 * TupleBase with a non-value head type.
 *
 * @tparam Head Head type.
 * @tparam Tail Tail types.
 */
template<class Head, class ...Tail>
class TupleBase<Head,IS_NOT_VALUE(Head),Tail...> {
  template<class Head1, class Enable1, class... Tail1> friend class TupleBase;
public:
  TupleBase() = default;
  TupleBase(const TupleBase&) = default;
  TupleBase(TupleBase&&) = default;
  TupleBase& operator=(const TupleBase&) = delete;
  TupleBase& operator=(TupleBase&&) = delete;

  /**
   * Constructor.
   */
  TupleBase(Label* context, Head head, Tail... tail) :
      head(context, std::forward(head)),
      tail(context, std::forward(tail...)) {
    //
  }

  /**
   * Copy constructor.
   */
  TupleBase(Label* context, const TupleBase& o) :
      head(context, o.head),
      tail(context, o.tail) {
    //
  }

  /**
   * Move constructor.
   */
  TupleBase(Label* context, TupleBase&& o) :
      head(context, std::move(o.head)),
      tail(context, std::move(o.tail)) {
    //
  }

  /**
   * Deep copy constructor.
   */
  TupleBase(Label* context, Label* label, const TupleBase& o) :
      head(context, label, o.head),
      tail(context, label, o.tail) {
    //
  }

  /**
   * Copy assignment.
   */
  template<class Head1, class ... Tail1>
  TupleBase& assign(Label* context, const TupleBase<Head1,Tail1...>& o) {
    head.assign(context, o.head);
    tail.assign(context, o.tail);
    return *this;
  }

  /**
   * Move assignment.
   */
  template<class Head1, class ... Tail1>
  TupleBase& assign(Label* context, TupleBase<Head1,Tail1...>&& o) {
    head.assign(context, std::move(o.head));
    tail.assign(context, std::move(o.tail));
    return *this;
  }

  void freeze() {
    head.freeze();
    tail.freeze();
  }

  void thaw(Label* label) {
    head.thaw(label);
    tail.thaw(label);
  }

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
  TupleBase<Tail...> tail;
};

/**
 * TupleBase with a value head type, and no tail.
 *
 * @param Head Value type.
 */
template<class Head>
class TupleBase<Head,IS_VALUE(Head)> {
  template<class Head1, class Enable1, class... Tail1> friend class TupleBase;
public:
  TupleBase() = default;
  TupleBase(const TupleBase&) = default;
  TupleBase(TupleBase&&) = default;
  TupleBase& operator=(const TupleBase&) = default;
  TupleBase& operator=(TupleBase&&) = default;

  /**
   * Constructor.
   */
  TupleBase(Head head) :
      head(std::forward(head)) {
    //
  }

  /**
   * Constructor.
   */
  TupleBase(Label* context, Head head) :
      head(std::forward(head)) {
    //
  }

  /**
   * Copy constructor.
   */
  TupleBase(Label* context, const TupleBase& o) :
      head(o.head) {
    //
  }

  /**
   * Move constructor.
   */
  TupleBase(Label* context, TupleBase&& o) :
      head(std::move(o.head)) {
    //
  }

  /**
   * Deep copy constructor.
   */
  TupleBase(Label* context, Label* label, const TupleBase& o) :
      head(o.head) {
    //
  }

  /**
   * Copy assignment.
   */
  template<class Head1>
  TupleBase& assign(Label* context, const TupleBase<Head1>& o) {
    head = o.head;
    return *this;
  }

  /**
   * Move assignment.
   */
  template<class Head1>
  TupleBase& assign(Label* context, TupleBase<Head1>&& o) {
    head = std::move(o.head);
    return *this;
  }

  void freeze() {
    //
  }

  void thaw(Label* label) {
    //
  }

  void finish() {
    //
  }

private:
  /**
   * First element.
   */
  Head head;
};

/**
 * TupleBase with a non-value head type, and no tail.
 *
 * @param Head Value type.
 */
template<class Head>
class TupleBase<Head,IS_NOT_VALUE(Head)> {
  template<class Head1, class Enable1, class... Tail1> friend class TupleBase;
public:
  TupleBase() = default;
  TupleBase(const TupleBase&) = default;
  TupleBase(TupleBase&&) = default;
  TupleBase& operator=(const TupleBase&) = delete;
  TupleBase& operator=(TupleBase&&) = delete;

  /**
   * Constructor.
   */
  TupleBase(Label* context, Head head) :
      head(context, std::forward(head)) {
    //
  }

  /**
   * Copy constructor.
   */
  TupleBase(Label* context, const TupleBase& o) :
      head(context, o.head) {
    //
  }

  /**
   * Move constructor.
   */
  TupleBase(Label* context, TupleBase&& o) :
      head(context, std::move(o.head)) {
    //
  }

  /**
   * Deep copy constructor.
   */
  TupleBase(Label* context, Label* label, const TupleBase& o) :
      head(context, label, o.head) {
    //
  }

  /**
   * Copy assignment.
   */
  template<class Head1>
  TupleBase& assign(Label* context, const TupleBase<Head1>& o) {
    head.assign(context, o.head);
    return *this;
  }

  /**
   * Move assignment.
   */
  template<class Head1>
  TupleBase& assign(Label* context, TupleBase<Head1>&& o) {
    head.assign(context, std::move(o.head));
    return *this;
  }

  void freeze() {
    head.freeze();
  }

  void thaw(Label* label) {
    head.thaw(label);
  }

  void finish() {
    head.finish();
  }

private:
  /**
   * First element.
   */
  Head head;
};


template<class Head, class... Tail>
using Tuple = TupleBase<Head,void,Tail...>;

template<class Head>
struct is_value<Tuple<Head>> {
  static const bool value = is_value<Head>::value;
};

template<class Head, class ... Tail>
struct is_value<Tuple<Head,Tail...>> {
  static const bool value = is_value<Head>::value && is_value<TupleBase<Tail...>>::value;
};

}
