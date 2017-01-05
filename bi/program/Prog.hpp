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
   * Partial order comparison operator for comparing programs in terms of
   * specialisation. These double-dispatch to the #le, #gt, #eq and #ne
   * functions below, which can be implemented for specific types in derived
   * classes.
   */
  bool operator<=(Prog& o);
  bool operator==(Prog& o);
  virtual bool dispatch(Prog& o) = 0;
  virtual bool le(ProgParameter& o);
  virtual bool le(ProgReference& o);
};
}
