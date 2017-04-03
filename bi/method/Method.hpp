/**
 * @file
 */
#pragma once

#include "bi/method/DelayInterface.hpp"

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
   * Add a delay variate.
   *
   * @param o Delay variate.
   *
   * @return Id assigned to the delay variate.
   */
  virtual int add(DelayInterface* o) = 0;

  /**
   * Get a delay variate.
   *
   * @param id Id of the delay variate.
   */
  virtual DelayInterface* get(const int id) = 0;

  /**
   * Simulate a delay variate.
   *
   * @param id Id of the delay variate.
   */
  virtual void sample(const int id) = 0;

  /**
   * Observe a delay variate.
   *
   * @param id Id of the delay variate.
   */
  virtual void observe(const int id) = 0;

};
}

/**
 * The chosen method.
 */
extern bi::Method* method;
