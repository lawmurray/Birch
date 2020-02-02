/**
 * @file
 */
#pragma once

namespace bi {
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
   * `auto` annotation on a variable declaration.
   */
  AUTO = 2,

  /**
   * `final` annotation on a class or member function.
   */
  FINAL = 4,

  /**
   * `abstract` annotation on a class or member function.
   */
  ABSTRACT = 8,
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
