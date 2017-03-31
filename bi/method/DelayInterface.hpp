/**
 * @file
 */
#pragma once

#include "bi/method/DelayState.hpp"

namespace bi {
/**
 * Interface for delay variates.
 */
class DelayInterface {
public:
  /**
   * Destructor.
   */
  virtual ~DelayInterface() {
    //
  }

  /**
   * Apply the simulate function.
   */
  virtual void simulate() = 0;

  /**
   * Apply the observe function.
   */
  virtual double observe() = 0;

  /**
   * Get the id of the delay variate.
   */
  virtual int getId() const = 0;

  /**
   * Set the id of the delay variate.
   */
  virtual void setId(const int id) = 0;

  /**
   * Get the state of the delay variate.
   */
  virtual DelayState getState() const = 0;

  /**
   * Set the state of the delay variate.
   */
  virtual void setState(const DelayState state) = 0;
};
}
