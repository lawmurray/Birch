/**
 * @file
 */
#pragma once

namespace birch {
class Expression;

/**
 * Iterator over expression list.
 *
 * @ingroup expression
 */
class ExpressionIterator {
public:
  /**
   * Constructor.
   *
   * @param o The list, `nullptr` gives a one-past-end iterator.
   */
  ExpressionIterator(Expression* o = nullptr);

  ExpressionIterator& operator++();
  ExpressionIterator operator++(int);
  Expression* operator*();
  bool operator==(const ExpressionIterator& o) const;
  bool operator!=(const ExpressionIterator& o) const;

private:
  /**
   * The list.
   */
  Expression* o;
};
}
