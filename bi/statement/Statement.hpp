/**
 * @file
 */
#pragma once

#include "bi/common/Located.hpp"
#include "bi/primitive/possibly.hpp"

namespace bi {
class Cloner;
class Modifier;
class Visitor;

class Conditional;
template<class T> class Declaration;
class EmptyStatement;
class ExpressionStatement;
class Import;
template<class T> class List;
class Loop;
class Raw;

class VarParameter;
class FuncParameter;
class ModelParameter;
class ProgParameter;

/**
 * Statement.
 *
 * @ingroup compiler_statement
 */
class Statement: public Located {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  Statement(shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Statement() = 0;

  /**
   * Accept cloning visitor.
   *
   * @param v The visitor.
   *
   * @return Cloned (and potentially modified) statement.
   */
  virtual Statement* accept(Cloner* visitor) const = 0;

  /**
   * Accept modifying visitor.
   *
   * @param v The visitor.
   *
   * @return Modified statement.
   */
  virtual Statement* accept(Modifier* visitor) = 0;

  /**
   * Accept read-only visitor.
   *
   * @param v The visitor.
   */
  virtual void accept(Visitor* visitor) const = 0;

  /*
   * Is statement empty?
   */
  virtual bool isEmpty() const;

  /*
   * Partial order comparison operators for comparing statements in terms of
   * specialisation. These double-dispatch to the #le, #gt, #eq and #ne
   * functions below, which can be implemented for specific types in derived
   * classes.
   */
  possibly operator<=(Statement& o);
  possibly operator==(Statement& o);
  virtual possibly dispatch(Statement& o) = 0;
  virtual possibly le(Conditional& o);
  virtual possibly le(Declaration<VarParameter>& o);
  virtual possibly le(Declaration<FuncParameter>& o);
  virtual possibly le(Declaration<ProgParameter>& o);
  virtual possibly le(Declaration<ModelParameter>& o);
  virtual possibly le(EmptyStatement& o);
  virtual possibly le(ExpressionStatement& o);
  virtual possibly le(Import& o);
  virtual possibly le(List<Statement>& o);
  virtual possibly le(Loop& o);
  virtual possibly le(Raw& o);
};
}
