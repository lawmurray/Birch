/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"

namespace bi {
/**
 * Iterator over expression list.
 *
 * @ingroup compiler_common
 */
class ExpressionIterator {
public:
  /**
   * Constructor.
   *
   * @param o The list, `nullptr` gives a one-past-end iterator.
   */
  ExpressionIterator(const Expression* o = nullptr);

  ExpressionIterator& operator++();
  ExpressionIterator operator++(int);
  const Expression* operator*();
  bool operator==(const ExpressionIterator& o) const;
  bool operator!=(const ExpressionIterator& o) const;

private:
  /**
   * The list.
   */
  const Expression* o;
};
}
