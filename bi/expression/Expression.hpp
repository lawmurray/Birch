/**
 * @file
 */
#pragma once

#include "bi/common/Typed.hpp"
#include "bi/common/Located.hpp"
#include "bi/common/Unknown.hpp"
#include "bi/common/Lookup.hpp"
#include "bi/expression/ExpressionIterator.hpp"
#include "bi/expression/ExpressionConstIterator.hpp"

namespace bi {
class Cloner;
class Modifier;
class Visitor;
template<class T> class Call;

/**
 * Expression.
 *
 * @ingroup expression
 */
class Expression: public Located, public Typed {
public:
  /**
   * Constructor.
   *
   * @param type Type.
   * @param loc Location.
   */
  Expression(Type* type, Location* loc = nullptr);

  /**
   * Constructor.
   *
   * @param loc Location.
   */
  Expression(Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Expression() = 0;

  /**
   * Accept cloning visitor.
   *
   * @param visitor The visitor.
   *
   * @return Cloned (and potentially modified) expression.
   */
  virtual Expression* accept(Cloner* visitor) const = 0;

  /**
   * Accept modifying visitor.
   *
   * @param visitor The visitor.
   *
   * @return Modified expression.
   */
  virtual Expression* accept(Modifier* visitor) = 0;

  /**
   * Accept read-only visitor.
   *
   * @param visitor The visitor.
   */
  virtual void accept(Visitor* visitor) const = 0;

  /**
   * Is this a value expression? Such an expression contains no usage of
   * class types.
   */
  bool isValue() const;

  /**
   * Is expression empty?
   */
  virtual bool isEmpty() const;

  /**
   * Is result of expression assignable?
   */
  virtual bool isAssignable() const;

  /**
   * Strip parentheses, if any.
   */
  virtual Expression* strip();

  /**
   * Get the left operand of a binary, otherwise undefined.
   */
  virtual Expression* getLeft() const;

  /**
   * Get the right operand of a binary, otherwise undefined.
   */
  virtual Expression* getRight() const;

  /**
   * Number of expresions in an expression list.
   */
  int width() const;

  /**
   * Number of range expressions in an expression list.
   */
  virtual int depth() const;

  /**
   * Iterator to first element if this is a list, to one-past-the-last if
   * this is empty, otherwise to this.
   */
  ExpressionIterator begin();

  /**
   * Iterator to one-past-the-last.
   */
  ExpressionIterator end();

  /**
   * Iterator to first element if this is a list, to one-past-the-last if
   * this is empty, otherwise to this.
   */
  ExpressionConstIterator begin() const;

  /**
   * Iterator to one-past-the-last.
   */
  ExpressionConstIterator end() const;

  /**
   * Look up the type of a call.
   *
   * @param args Arguments.
   *
   * @return The type of the call.
   */
  virtual Lookup lookup(Expression* args);

  /**
   * Resolve a call.
   *
   * @param o The unresolved call.
   *
   * @return The resolved call.
   */
  virtual Parameter* resolve(Call<Parameter>* o);
  virtual FiberParameter* resolve(Call<FiberParameter>* o);
  virtual LocalVariable* resolve(Call<LocalVariable>* o);
  virtual FiberVariable* resolve(Call<FiberVariable>* o);
  virtual MemberVariable* resolve(Call<MemberVariable>* o);
  virtual GlobalVariable* resolve(Call<GlobalVariable>* o);
  virtual Function* resolve(Call<Function>* o);
  virtual MemberFunction* resolve(Call<MemberFunction>* o);
  virtual Fiber* resolve(Call<Fiber>* o);
  virtual MemberFiber* resolve(Call<MemberFiber>* o);
  virtual UnaryOperator* resolve(Call<UnaryOperator>* o);
  virtual BinaryOperator* resolve(Call<BinaryOperator>* o);
};
}
