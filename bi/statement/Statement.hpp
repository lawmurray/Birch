/**
 * @file
 */
#pragma once

#include "bi/common/Located.hpp"

namespace bi {
class Cloner;
class Modifier;
class Visitor;

class EmptyStatement;
class ExpressionStatement;
class For;
class If;
class Import;
template<class T> class List;
class Raw;
class Return;
class While;

class Expression;
class Type;

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
  virtual bool definitely(const Statement& o) const;
  virtual bool dispatchDefinitely(const Statement& o) const = 0;
  virtual bool definitely(const EmptyStatement& o) const;
  virtual bool definitely(const ExpressionStatement& o) const;
  virtual bool definitely(const For& o) const;
  virtual bool definitely(const If& o) const;
  virtual bool definitely(const Import& o) const;
  virtual bool definitely(const List<Statement>& o) const;
  virtual bool definitely(const Return& o) const;
  virtual bool definitely(const Raw& o) const;
  virtual bool definitely(const While& o) const;

  virtual bool possibly(const Statement& o) const;
  virtual bool dispatchPossibly(const Statement& o) const = 0;
  virtual bool possibly(const EmptyStatement& o) const;
  virtual bool possibly(const ExpressionStatement& o) const;
  virtual bool possibly(const For& o) const;
  virtual bool possibly(const If& o) const;
  virtual bool possibly(const Import& o) const;
  virtual bool possibly(const List<Statement>& o) const;
  virtual bool possibly(const Return& o) const;
  virtual bool possibly(const Raw& o) const;
  virtual bool possibly(const While& o) const;
};
}
