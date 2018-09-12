/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/primitive/poset.hpp"
#include "bi/primitive/definitely.hpp"

namespace bi {
/**
 * Overloaded object. Groups all overloads into one object.
 *
 * @ingroup common
 */
class Overloaded {
public:
  /**
   * Constructor.
   *
   * @param o First overload.
   */
  Overloaded(Parameterised* o);

  /**
   * Does this contain the given overload?
   *
   * @param o The overload.
   */
  bool contains(Parameterised* o);

  /**
   * Get the given overload.
   *
   * @param o The overload.
   */
  Parameterised* get(Parameterised* o);

  /**
   * Add an overload.
   *
   * @param o The overload.
   */
  void add(Parameterised* o);

  /**
   * Number of overloads.
   */
  int size() const;

  /**
   * Iterators.
   */
  auto begin() const {
    return overloads.begin();
  }
  auto end() const {
    return overloads.end();
  }

  /**
   * First overload.
   */
  auto front() const {
    return *begin();
  }

  /**
   * Overloads.
   */
  poset<Parameterised*,definitely> overloads;
};
}
