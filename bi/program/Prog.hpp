/**
 * @file
 */
#pragma once

#include "bi/common/Located.hpp"

namespace bi {
class Cloner;
class Modifier;
class Visitor;

class ProgParameter;
class ProgReference;

/**
 * Program.
 *
 * @ingroup compiler_program
 */
class Prog: public Located {
public:
  /**
   * Constructor.
   */
  Prog(shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Prog() = 0;

  /**
   * Accept cloning visitor.
   *
   * @param v The visitor.
   *
   * @return Cloned (and potentially modified) statement.
   */
  virtual Prog* accept(Cloner* visitor) const = 0;

  /**
   * Accept modifying visitor.
   *
   * @param v The visitor.
   *
   * @return Modified statement.
   */
  virtual Prog* accept(Modifier* visitor) = 0;

  /**
   * Accept read-only visitor.
   *
   * @param v The visitor.
   */
  virtual void accept(Visitor* visitor) const = 0;

  /*
   * Double-dispatch partial order comparisons.
   */
  virtual bool definitely(Prog& o);
  virtual bool dispatchDefinitely(Prog& o) = 0;
  virtual bool definitely(ProgParameter& o);
  virtual bool definitely(ProgReference& o);

  virtual bool possibly(Prog& o);
  virtual bool dispatchPossibly(Prog& o) = 0;
  virtual bool possibly(ProgParameter& o);
  virtual bool possibly(ProgReference& o);
};
}
