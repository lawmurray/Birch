/**
 * @file
 */
#pragma once

namespace birch {
/**
 * Functor that provides a partial ordering of classes and structs based on
 * inheritance relationships.
 */
struct inherits {
  bool operator()(const Class* a, const Class* b) {
    auto base = dynamic_cast<const NamedType*>(a->base);
    if (base) {
      return *b->name == *base->name;
    } else {
      return *b->name == "Object";
    }
  }

  bool operator()(const Struct* a, const Struct* b) {
    auto base = dynamic_cast<const NamedType*>(a->base);
    return base && *b->name == *base->name;
  }
};
}
