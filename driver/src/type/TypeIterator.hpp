/**
 * @file
 */
#pragma once

namespace birch {
class Type;
/**
 * Iterator over type list.
 *
 * @ingroup type
 */
class TypeIterator {
public:
  /**
   * Constructor.
   *
   * @param o The list, `nullptr` gives a one-past-end iterator.
   */
  TypeIterator(Type* o = nullptr);

  TypeIterator& operator++();
  TypeIterator operator++(int);
  Type* operator*();
  bool operator==(const TypeIterator& o) const;
  bool operator!=(const TypeIterator& o) const;

private:
  /**
   * The list.
   */
  Type* o;
};
}
