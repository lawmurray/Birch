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
   * Is this a resume function, or local variable to be restored in a resume
   * function?
   */
  RESUME = 32,

  /**
   * Is this a start function?
   */
  START = 64,
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
   * Destructor.
   */
  virtual ~Annotated() = 0;

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
