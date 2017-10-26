/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"

namespace bi {
/**
 * Iterator over type list.
 *
 * @ingroup compiler_common
 */
class TypeIterator {
public:
  /**
   * Constructor.
   *
   * @param o The list, `nullptr` gives a one-past-end iterator.
   */
  TypeIterator(const Type* o = nullptr);

  TypeIterator& operator++();
  TypeIterator operator++(int);
  const Type* operator*();
  bool operator==(const TypeIterator& o) const;
  bool operator!=(const TypeIterator& o) const;

private:
  /**
   * The list.
   */
  const Type* o;
};
}
