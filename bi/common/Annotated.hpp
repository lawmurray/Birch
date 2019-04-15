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
  AUTO = 2,
  FINAL = 3,
  PRIOR_INSTANTIATION = 4
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
