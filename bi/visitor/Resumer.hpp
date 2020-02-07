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
 * @li Any statement is reachable if it is in the same scope as the yield
 * statement and follows it.
 * @li A loop statement that contains one or more reachable statements, at
 * any depth, are also reachable.
 */
class Resumer : public Cloner {
public:
  /**
   * Constructor.
   */
  Resumer(const Yield* yield = nullptr);

  /**
   * Destructor.
   */
  virtual ~Resumer();

  virtual Statement* clone(const Fiber* o) override;
  virtual Statement* clone(const MemberFiber* o) override;
  virtual Statement* clone(const Yield* o) override;
  virtual Statement* clone(const If* o) override;
  virtual Statement* clone(const StatementList* o) override;
  virtual Statement* clone(const While* o) override;
  virtual Statement* clone(const DoWhile* o) override;

private:
  /**
   * The yield point.
   */
  const Yield* yield;

  /**
   * Has the yield point been encountered yet?
   */
  bool foundYield;

  /**
   * Are we in a loop?
   */
  int inLoop;
};
}
