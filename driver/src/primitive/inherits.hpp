/**
 * @file
 */
#pragma once

namespace birch {
/**
 * Functor that provides a partial ordering of classes based on inheritance
 * relationships.
 */
struct inherits {
  bool operator()(const Class* a, const Class* b) {
    auto base = dynamic_cast<const NamedType*>(a->base);
    return base && *base->name == *b->name;
  }
};
}
