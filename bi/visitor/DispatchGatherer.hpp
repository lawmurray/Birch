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
  virtual void visit(const FuncReference* o);

  /**
   * Gathered possible matches.
   */
  std::set<const FuncParameter*> gathered;
};
}
