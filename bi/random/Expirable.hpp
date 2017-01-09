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
   * Constructor.
   */
  Expirable();

  /**
   * Expire this object.
   */
  virtual void expire();

  /**
   * Is this object expired?
   */
  bool isExpired() const;

private:
  /**
   * Is this object expired?
   */
  bool expired;
};
}
