/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Possible states of a random variable.
 */
enum RandomState {
  /**
   * Value is unknown.
   */
  MISSING = 0,

  /**
   * A known value that has been simulated internally.
   */
  SIMULATED,

  /**
   * A known value that has been assigned externally.
   */
  ASSIGNED
};
}
