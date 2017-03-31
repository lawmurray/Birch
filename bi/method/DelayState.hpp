/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Possible states of a delay variate.
 */
enum DelayState {
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
