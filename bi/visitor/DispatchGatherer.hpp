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

  virtual void visit(const FuncReference* o);

  /**
   * Gathered matches.
   */
  std::set<const FuncParameter*> gathered;

private:
  /**
   * Insert function.
   */
  void insert(const FuncParameter* o);

  /**
   * Scope for finding parents of functions.
   */
  Scope* scope;
};
}
