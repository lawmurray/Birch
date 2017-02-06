/**
 * @file
 */
#pragma once

#include "bi/visitor/Visitor.hpp"

namespace bi {
/**
 * Gather all possible matches for function references that will need to be
 * checked at runtime.
 *
 * @ingroup compiler_visitor
 */
class DispatchGatherer : public Visitor {
public:
  /**
   * Constructor.
   *
   * @param scope Scope for finding parents of functions.
   */
  DispatchGatherer(Scope* scope);

  /**
   * Begin iterator over gathered objects.
   */
  auto begin() {
    return gathered.begin();
  }

  /**
   * End iterator over gathered objects.
   */
  auto end() {
    return gathered.end();
  }

  /**
   * Number of items gathered.
   */
  auto size() {
    return gathered.size();
  }

  virtual void visit(const FuncReference* o);
  virtual void visit(const RandomInit* o);

private:
  /**
   * Insert function.
   */
  void insert(const FuncParameter* o);

  /**
   * Scope for finding parents of functions.
   */
  Scope* scope;

  /**
   * Gathered matches.
   */
  std::set<const FuncParameter*> gathered;
};
}
