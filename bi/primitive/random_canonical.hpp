/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Canonical model associated with random variable.
 */
class random_canonical {
public:
  /**
   * Destructor.
   */
  virtual ~random_canonical() {
    //
  }

  /**
   * Apply the simulate function.
   */
  virtual void simulate() = 0;

  /**
   * Apply the backward function.
   */
  virtual double backward() = 0;
};
}
