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
   * Destructor.
   */
  virtual ~Expirable();

  /**
   * Expire this object.
   */
  virtual void expire() = 0;
};
}
