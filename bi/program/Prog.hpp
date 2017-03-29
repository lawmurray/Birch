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
  virtual bool definitely(const Prog& o) const;
  virtual bool dispatchDefinitely(const Prog& o) const = 0;
  virtual bool definitely(const ProgParameter& o) const;
  virtual bool definitely(const ProgReference& o) const;

  virtual bool possibly(const Prog& o) const;
  virtual bool dispatchPossibly(const Prog& o) const = 0;
  virtual bool possibly(const ProgParameter& o) const;
  virtual bool possibly(const ProgReference& o) const;
};
}
