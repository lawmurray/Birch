/**
 * @file
 */
#pragma once

namespace bi {
class Expression;

/**
 * Iterator over constant expression list.
 *
 * @ingroup expression
 */
class ExpressionConstIterator {
public:
  /**
   * Constructor.
   *
   * @param o The list, `nullptr` gives a one-past-end iterator.
   */
  ExpressionConstIterator(const Expression* o = nullptr);

  ExpressionConstIterator& operator++();
  ExpressionConstIterator operator++(int);
  const Expression* operator*();
  bool operator==(const ExpressionConstIterator& o) const;
  bool operator!=(const ExpressionConstIterator& o) const;

private:
  /**
   * The list.
   */
  const Expression* o;
};
}
