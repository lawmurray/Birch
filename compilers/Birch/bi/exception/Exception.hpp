/**
 * @file
 */
#pragma once

namespace bi {
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
