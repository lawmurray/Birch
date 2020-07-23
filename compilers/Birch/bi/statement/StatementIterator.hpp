/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"

namespace bi {
/**
 * Iterator over statement list.
 *
 * @ingroup common
 */
class StatementIterator {
public:
  /**
   * Constructor.
   *
   * @param o The list, `nullptr` gives a one-past-end iterator.
   */
  StatementIterator(const Statement* o = nullptr);

  StatementIterator& operator++();
  StatementIterator operator++(int);
  const Statement* operator*();
  bool operator==(const StatementIterator& o) const;
  bool operator!=(const StatementIterator& o) const;

private:
  /**
   * The list.
   */
  const Statement* o;
};
}
