/**
 * @file
 */
#pragma once

namespace birch {
/**
 * Exception.
 *
 * @ingroup exception
 */
struct Exception {
  /**
   * Default constructor.
   */
  Exception();

  /**
   * Constructor.
   */
  Exception(const std::string& msg);

  /**
   * Message.
   */
  std::string msg;
};
}
