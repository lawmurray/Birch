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
  AUTO = 2
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
   * @param args Arguments.
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
   * Annotation.
   */
  Annotation annotation;
};
}
