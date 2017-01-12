/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Expirable object.
 */
class Expirable {
public:
  /**
   * Expire this object.
   */
  virtual void expire() = 0;
};
}
