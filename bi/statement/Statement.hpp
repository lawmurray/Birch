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
   * Partial order comparison operators for comparing statements in terms of
   * specialisation. These double-dispatch to the #le, #gt, #eq and #ne
   * functions below, which can be implemented for specific types in derived
   * classes.
   */
  bool operator<=(Statement& o);
  bool operator==(Statement& o);
  virtual bool dispatch(Statement& o) = 0;
  virtual bool le(Conditional& o);
  virtual bool le(Declaration<VarParameter>& o);
  virtual bool le(Declaration<FuncParameter>& o);
  virtual bool le(Declaration<ProgParameter>& o);
  virtual bool le(Declaration<ModelParameter>& o);
  virtual bool le(EmptyStatement& o);
  virtual bool le(ExpressionStatement& o);
  virtual bool le(Import& o);
  virtual bool le(List<Statement>& o);
  virtual bool le(Loop& o);
  virtual bool le(Raw& o);
};
}
