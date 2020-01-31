/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Functor that provides a partial ordering of classes based on inheritance
 * relationships.
 */
struct inherits {
  static bool operator()(const Class* a, const Class* b) {
    return o->base->isClass() && *o->base->name == *b->name;
  }
};
}
