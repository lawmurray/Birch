/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Annotations for declarations.
 */
enum Annotation {
  NONE = 0, IS_CLOSED = 1
};

/**
 * Annotated object.
 *
 * @ingroup compiler_common
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
   * Is this closed?
   */
  bool isClosed() const;

  /**
   * Annotation.
   */
  Annotation annotation;
};
}
