/**
 * @file
 */
#pragma once

#include "bi/common/Located.hpp"

namespace bi {
class Cloner;
class Modifier;
class Visitor;

class Alias;
class AssignmentOperator;
class Basic;
class BinaryOperator;
class Class;
class ConversionOperator;
class Coroutine;
class EmptyStatement;
class ExpressionStatement;
class Function;
class For;
class GlobalVariable;
class If;
class Import;
class LocalVariable;
class MemberCoroutine;
class MemberFunction;
class MemberVariable;
class Program;
template<class T> class List;
class Raw;
class Return;
class UnaryOperator;
class While;

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
  virtual bool definitely(const Alias& o) const;
  virtual bool definitely(const AssignmentOperator& o) const;
  virtual bool definitely(const Basic& o) const;
  virtual bool definitely(const BinaryOperator& o) const;
  virtual bool definitely(const Class& o) const;
  virtual bool definitely(const ConversionOperator& o) const;
  virtual bool definitely(const Coroutine& o) const;
  virtual bool definitely(const EmptyStatement& o) const;
  virtual bool definitely(const ExpressionStatement& o) const;
  virtual bool definitely(const Function& o) const;
  virtual bool definitely(const For& o) const;
  virtual bool definitely(const GlobalVariable& o) const;
  virtual bool definitely(const If& o) const;
  virtual bool definitely(const Import& o) const;
  virtual bool definitely(const List<Statement>& o) const;
  virtual bool definitely(const LocalVariable& o) const;
  virtual bool definitely(const MemberCoroutine& o) const;
  virtual bool definitely(const MemberFunction& o) const;
  virtual bool definitely(const MemberVariable& o) const;
  virtual bool definitely(const Program& o) const;
  virtual bool definitely(const Return& o) const;
  virtual bool definitely(const Raw& o) const;
  virtual bool definitely(const UnaryOperator& o) const;
  virtual bool definitely(const While& o) const;

  virtual bool possibly(const Statement& o) const;
  virtual bool dispatchPossibly(const Statement& o) const = 0;
  virtual bool possibly(const Alias& o) const;
  virtual bool possibly(const AssignmentOperator& o) const;
  virtual bool possibly(const Basic& o) const;
  virtual bool possibly(const BinaryOperator& o) const;
  virtual bool possibly(const Class& o) const;
  virtual bool possibly(const ConversionOperator& o) const;
  virtual bool possibly(const Coroutine& o) const;
  virtual bool possibly(const EmptyStatement& o) const;
  virtual bool possibly(const ExpressionStatement& o) const;
  virtual bool possibly(const For& o) const;
  virtual bool possibly(const Function& o) const;
  virtual bool possibly(const GlobalVariable& o) const;
  virtual bool possibly(const If& o) const;
  virtual bool possibly(const Import& o) const;
  virtual bool possibly(const List<Statement>& o) const;
  virtual bool possibly(const LocalVariable& o) const;
  virtual bool possibly(const MemberCoroutine& o) const;
  virtual bool possibly(const MemberFunction& o) const;
  virtual bool possibly(const MemberVariable& o) const;
  virtual bool possibly(const Program& o) const;
  virtual bool possibly(const Return& o) const;
  virtual bool possibly(const Raw& o) const;
  virtual bool possibly(const UnaryOperator& o) const;
  virtual bool possibly(const While& o) const;
};
}
