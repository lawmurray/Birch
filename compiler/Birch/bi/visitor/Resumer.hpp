/**
 * @file
 */
#pragma once

#include "bi/visitor/Cloner.hpp"

namespace bi {
/**
 * Creates the resume function for a particular yield point.
 *
 * @ingroup visitor
 *
 * To create the resume function, rules are applied to determine potentially
 * reachable blocks of code from the yield point:
 *
 * @li Any statement is reachable if it follows that yield statement in
 * execution order.
 * @li If the yield statement is contained in a loop, all statements in the
 * loop are reachable, as it may iterate again. In this case all statements
 * following the yield in the body of the loop (i.e. reachable in the current
 * iteration) are unrolled from the loop, and the whole loop follows them
 * again.
 */
class Resumer : public Cloner {
public:
  /**
   * Constructor.
   *
   * @param yield The yield for which to construct a resume function. If
   * null, constructs the start function.
   */
  Resumer(const Yield* yield = nullptr);

  /**
   * Destructor.
   */
  virtual ~Resumer();

  virtual Statement* clone(const Fiber* o) override;
  virtual Statement* clone(const MemberFiber* o) override;
  virtual Statement* clone(const LocalVariable* o) override;
  virtual Statement* clone(const Yield* o) override;
  virtual Statement* clone(const Return* o) override;
  virtual Statement* clone(const ExpressionStatement* o) override;
  virtual Statement* clone(const Assert* o) override;
  virtual Statement* clone(const Raw* o) override;
  virtual Statement* clone(const StatementList* o) override;
  virtual Statement* clone(const If* o) override;
  virtual Statement* clone(const For* o) override;
  virtual Statement* clone(const Parallel* o) override;
  virtual Statement* clone(const While* o) override;
  virtual Statement* clone(const DoWhile* o) override;
  virtual Statement* clone(const Block* o) override;

private:
  /**
   * The yield point.
   */
  const Yield* yield;

  /**
   * Has the yield point been encountered yet?
   */
  bool foundYield;
};
}
