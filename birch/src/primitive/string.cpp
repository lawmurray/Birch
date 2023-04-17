/**
 * @file
 */
#include "src/primitive/string.hpp"

bool birch::isTranslatable(const std::string& op) {
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

    init = true;
  }
  return ops.find(op) != ops.end();
}

std::string birch::nice(const std::string& name) {
  /* translations */
  static std::regex reg;
  static std::unordered_map<std::string,std::string> ops;
  static bool init = false;
  if (!init) {
    ops["<-"] = "assign_";
    ops["<~"] = "left_tilde_";
    ops["~>"] = "right_tilde_";
    ops["~"] = "tilde_";
    ops[".."] = "range_";
    ops["+"] = "add_";
    ops["+ (unary)"] = "plus_";
    ops["+ (binary)"] = "add_";
    ops["-"] = "subtract_";
    ops["- (unary)"] = "negate_";
    ops["- (binary)"] = "subtract_";
    ops["*"] = "multiply_";
    ops["/"] = "divide_";
    ops["<"] = "lt_";
    ops[">"] = "gt_";
    ops["<="] = "le_";
    ops[">="] = "ge_";
    ops["=="] = "eq_";
    ops["!="] = "ne_";
    ops["!"] = "not_";
    ops["||"] = "or_";
    ops["&&"] = "and_";

    init = true;
  }

  /* translate operators */
  std::string str = name;
  if (ops.find(name) != ops.end()) {
    str = ops[name];
  }

  /* translate prime (apostrophe at end of name) */
  str = std::regex_replace(str, std::regex("'"), "_prime_");



  return str;
}

std::string birch::sanitize(const std::string& name) {
  std::string str = nice(name);

  /* stdin, stdout, and stderr are defined as macros in C, but also global
   * variables in Birch; this can create conflicts in some implementations,
   * e.g. musl, so explicitly rename the Birch variables here */
  if (str == "stdin" || str == "stdout" || str == "stderr") {
    str += "_";
  }

  return str;
}

std::string birch::escape_unicode(const std::string& str) {
  /* ideally we would just write the UTF-8 encoded character, but this only
   * works with more recent compilers; next best is to encode as \u0000
   * universal character name */
  std::wstring_convert<std::codecvt_utf8_utf16<char16_t>,char16_t> converter;
  std::u16string wstr = converter.from_bytes(str);
  std::stringstream buf;
  for (auto c : wstr) {
    if (c <= 127) {
      buf << (char)c;
    } else {
      buf << "\\u" << std::setfill('0') << std::setw(4) << std::hex << c;
    }
  }
  return buf.str();
}

std::string birch::lower(const std::string& str) {
  auto res = str;
  transform(res.begin(), res.end(), res.begin(), ::tolower);
  return res;
}

std::string birch::upper(const std::string& str) {
  auto res = str;
  transform(res.begin(), res.end(), res.begin(), ::toupper);
  return res;
}

std::string birch::tar(const std::string& name) {
  return std::regex_replace(lower(name), std::regex("\\."), "-");
}

std::string birch::canonical(const std::string& name) {
  return std::regex_replace(tar(name), std::regex("-"), "_");
}
