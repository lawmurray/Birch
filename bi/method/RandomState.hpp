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
   * A known value has been assigned externally.
   */
  ASSIGNED,

  /**
   * A known value has been simulated internally.
   */
  SIMULATED,
};
}
