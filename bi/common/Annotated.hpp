/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Annotations for declarations.
 */
enum Annotation {
  NONE = 0,
  PARALLEL = 1,
  DYNAMIC = 2,
  AUTO = 4,
  FINAL = 8,
  ABSTRACT = 16,
  INSTANTIATED = 32
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
