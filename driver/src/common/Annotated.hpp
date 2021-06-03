/**
 * @file
 */
#pragma once

namespace birch {
/**
 * Annotations for declarations.
 */
enum Annotation {
  /**
   * No annotation.
   */
  NONE = 0,

  /**
   * `dynamic` annotation on a parallel loop.
   */
  DYNAMIC = 1,

  /**
   * `let` annotation on a variable declaration.
   */
  LET = 2,

  /**
   * `abstract` annotation on a class or member function.
   */
  ABSTRACT = 4,

  /**
   * `override` annotation on a class or member function.
   */
  OVERRIDE = 8,

  /**
   * `final` annotation on a class or member function.
   */
  FINAL = 16,

  /**
   * `acyclic` annotation on a class.
   */
  ACYCLIC = 32,

  /**
   * Struct, rather than class (implies FINAL).
   */
  STRUCT = 64|16
};

/**
 * Annotated object.
 *
 * @ingroup common
 */
class Annotated {
public:
  /**
   * Constructor.
   *
   * @param annotation Annotations.
   */
  Annotated(const Annotation annotation);

  /**
   * Does this object have a particular annotation?
   */
  bool has(const Annotation a) const;

  /**
   * Set a particular annotation.
   */
  void set(const Annotation a);

  /**
   * Annotation.
   */
  Annotation annotation;
};
}
