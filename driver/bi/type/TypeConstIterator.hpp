/**
 * @file
 */
#pragma once

namespace bi {
class Type;
/**
 * Iterator over constant type list.
 *
 * @ingroup type
 */
class TypeConstIterator {
public:
  /**
   * Constructor.
   *
   * @param o The list, `nullptr` gives a one-past-end iterator.
   */
  TypeConstIterator(const Type* o = nullptr);

  TypeConstIterator& operator++();
  TypeConstIterator operator++(int);
  const Type* operator*();
  bool operator==(const TypeConstIterator& o) const;
  bool operator!=(const TypeConstIterator& o) const;

private:
  /**
   * The list.
   */
  const Type* o;
};
}
