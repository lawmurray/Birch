/**
 * @file
 */
#pragma once

#include "bi/method/RandomState.hpp"

namespace bi {
/**
 * Canonical model associated with random variable.
 */
class RandomInterface {
public:
  /**
   * Destructor.
   */
  virtual ~RandomInterface() {
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
   * Get the id of the random variable.
   */
  virtual int getId() const = 0;

  /**
   * Set the id of the random variable.
   */
  virtual void setId(const int id) = 0;

  /**
   * Get the state of the random variable.
   */
  virtual RandomState getState() const = 0;

  /**
   * Set the state of the random variable.
   */
  virtual void setState(const RandomState state) = 0;
};
}
