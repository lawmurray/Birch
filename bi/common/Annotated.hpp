/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Annotations for declarations.
 */
enum Annotation {
  NONE = 0, IS_CLOSED = 1, IS_READ_ONLY = 2
};

/**
 * Annotated object.
 *
 * @ingroup birch_common
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
   * Is this read only?
   */
  bool isReadOnly() const;

  /**
   * Annotation.
   */
  Annotation annotation;
};
}
