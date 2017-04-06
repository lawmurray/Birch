/**
 * @file
 */
#pragma once

namespace bi {
class MemoryGroup;

/**
 * Value of a primitive type.
 *
 * @ingroup library
 *
 * @tparam Value Primitive type.
 * @tparam Group Allocator type.
 *
 * PrimitiveValue is specialised by Group.
 *
 * PrimitiveValue objects objects masquerade as objects of type Value, in
 * order that the struct-of-arrays idiom is as non-intrusive as possible for
 * client code. This involves overloading the assignment, type conversion and
 * address-of operators.
 */
template<class Value, class Group = MemoryGroup>
class PrimitiveValue {
  //
};
}
