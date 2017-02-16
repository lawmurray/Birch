/**
 * @file
 */
#pragma once

#include "bi/primitive/random_canonical.hpp"

namespace bi {
/**
 * Abstract method.
 */
class Method {
public:
  /**
   * Destructor.
   */
  virtual ~Method();

  /**
   * Add a random variable.
   */
  virtual int add(random_canonical* rv, const int state) = 0;

  /**
   * Get a random variable.
   */
  virtual random_canonical* get(const int state) = 0;

  /**
   * Simulate a random variable.
   */
  virtual void simulate(const int state) = 0;
};
}

/**
 * The chosen method.
 */
extern bi::Method* method;
