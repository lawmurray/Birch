/**
 * @file
 */
#pragma once

#include "bi/common/Located.hpp"

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
   * Double-dispatch partial order comparisons.
   */
  virtual bool definitely(Statement& o);
  virtual bool dispatchDefinitely(Statement& o) = 0;
  virtual bool definitely(Conditional& o);
  virtual bool definitely(Declaration<VarParameter>& o);
  virtual bool definitely(Declaration<FuncParameter>& o);
  virtual bool definitely(Declaration<ProgParameter>& o);
  virtual bool definitely(Declaration<ModelParameter>& o);
  virtual bool definitely(EmptyStatement& o);
  virtual bool definitely(ExpressionStatement& o);
  virtual bool definitely(Import& o);
  virtual bool definitely(List<Statement>& o);
  virtual bool definitely(Loop& o);
  virtual bool definitely(Raw& o);

  virtual bool possibly(Statement& o);
  virtual bool dispatchPossibly(Statement& o) = 0;
  virtual bool possibly(Conditional& o);
  virtual bool possibly(Declaration<VarParameter>& o);
  virtual bool possibly(Declaration<FuncParameter>& o);
  virtual bool possibly(Declaration<ProgParameter>& o);
  virtual bool possibly(Declaration<ModelParameter>& o);
  virtual bool possibly(EmptyStatement& o);
  virtual bool possibly(ExpressionStatement& o);
  virtual bool possibly(Import& o);
  virtual bool possibly(List<Statement>& o);
  virtual bool possibly(Loop& o);
  virtual bool possibly(Raw& o);
};
}
