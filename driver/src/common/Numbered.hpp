/**
 * @file
 */
#pragma once

namespace birch {
/**
 * A uniquely numbered object.
 */
class Numbered {
public:
  /**
   * Constructor.
   */
  Numbered();

  /**
   * Unique number.
   */
  int number;

private:
  /**
   * Counter.
   */
  static int COUNTER;
};
}
