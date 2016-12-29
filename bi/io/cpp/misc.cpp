/**
 * @file
 */
#include "bi/io/cpp/misc.hpp"

#include <unordered_set>
#include <cassert>

bool bi::isTranslatable(const std::string& op) {
  static std::unordered_set<std::string> cppOps;
  static bool init = false;
  if (!init) {
    cppOps.insert("+");
    cppOps.insert("-");
    cppOps.insert("*");
    cppOps.insert("/");
    cppOps.insert("<");
    cppOps.insert(">");
    cppOps.insert("<=");
    cppOps.insert(">=");
    cppOps.insert("==");
    cppOps.insert("!=");
    cppOps.insert("!");
    cppOps.insert("||");
    cppOps.insert("&&");
    cppOps.insert("<-");

    init = true;
  }
  return cppOps.find(op) != cppOps.end();
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
