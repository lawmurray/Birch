/**
 * @file
 */
#pragma once

#include "bi/method/RandomInterface.hpp"

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
   *
   * @param rv Random variable.
   *
   * @return Id assigned to the random variable.
   */
  virtual int add(RandomInterface* rv) = 0;

  /**
   * Get a random variable.
   *
   * @param id Id of the random variable.
   */
  virtual RandomInterface* get(const int id) = 0;

  /**
   * Simulate a random variable.
   *
   * @param id Id of the random variable.
   */
  virtual void simulate(const int id) = 0;
};
}

/**
 * The chosen method.
 */
extern bi::Method* method;
