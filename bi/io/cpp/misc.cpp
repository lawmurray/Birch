/**
 * @file
 */
#include "bi/io/cpp/misc.hpp"

#include <unordered_set>
#include <cassert>

bool bi::isTranslatable(const std::string& op) {
  static std::unordered_set<std::string> ops;
  static bool init = false;
  if (!init) {
    ops.insert("+");
    ops.insert("-");
    ops.insert("*");
    ops.insert("/");
    ops.insert("<");
    ops.insert(">");
    ops.insert("<=");
    ops.insert(">=");
    ops.insert("==");
    ops.insert("!=");
    ops.insert("!");
    ops.insert("||");
    ops.insert("&&");
    ops.insert("<-");

    init = true;
  }
  return ops.find(op) != ops.end();
}

std::string bi::translate(const std::string& op) {
  /* pre-condition */
  assert(isTranslatable(op));

  if (op == "<-") {
    return "=";
  } else {
    return op;
  }
}
