/**
 * @file
 */
#pragma once

namespace bi {
class Parameter;
class LocalVariable;
class MemberVariable;
class GlobalVariable;
class Function;
class MemberFunction;
class Fiber;
class MemberFiber;
class BinaryOperator;
class UnaryOperator;
class Unknown;

/**
 * Categories of objects for identifier lookups.
 */
enum Lookup {
  PARAMETER,
  LOCAL_VARIABLE,
  MEMBER_VARIABLE,
  GLOBAL_VARIABLE,
  FUNCTION,
  MEMBER_FUNCTION,
  FIBER,
  MEMBER_FIBER,
  UNARY_OPERATOR,
  BINARY_OPERATOR,
  BASIC,
  CLASS,
  ALIAS,
  GENERIC,
  UNRESOLVED
};

/**
 * Convert a target type into a Lookup.
 */
template<class ObjectType>
struct lookup_result {
  static const Lookup value = UNRESOLVED;
};
template<>
struct lookup_result<Parameter> {
  static const Lookup value = PARAMETER;
};
template<>
struct lookup_result<LocalVariable> {
  static const Lookup value = LOCAL_VARIABLE;
};
template<>
struct lookup_result<MemberVariable> {
  static const Lookup value = MEMBER_VARIABLE;
};
template<>
struct lookup_result<GlobalVariable> {
  static const Lookup value = GLOBAL_VARIABLE;
};
template<>
struct lookup_result<Function> {
  static const Lookup value = FUNCTION;
};
template<>
struct lookup_result<MemberFunction> {
  static const Lookup value = MEMBER_FUNCTION;
};
template<>
struct lookup_result<Fiber> {
  static const Lookup value = FIBER;
};
template<>
struct lookup_result<MemberFiber> {
  static const Lookup value = MEMBER_FIBER;
};
template<>
struct lookup_result<UnaryOperator> {
  static const Lookup value = UNARY_OPERATOR;
};
template<>
struct lookup_result<BinaryOperator> {
  static const Lookup value = BINARY_OPERATOR;
};

}
