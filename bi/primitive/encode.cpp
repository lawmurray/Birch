/**
 * @file
 */
#include "bi/primitive/encode.hpp"

#include "boost/algorithm/string.hpp"

#include <iomanip>
#include <locale>
#include <codecvt>

std::string bi::encode32(const std::string& in) {
  std::string out;
  out.resize((in.length() + 4) / 5 * 7);

  unsigned long num, i1, i2, j;
  for (i1 = 0, i2 = 0; i1 < in.length(); i1 += 5, i2 += 7) {
    num = 0;
    for (j = 0; j < 5; ++j) {
      num <<= 7;
      if (i1 + j < in.length()) {
        num |= in[i1 + j] & 0x7F;
      }
    }
    for (j = 0; j < 7; ++j) {
      out[i2 + 6 - j] = encode32(num & 0x1F);
      num >>= 5;
    }
  }
  return out;
}

std::string bi::decode32(const std::string& in) {
  std::string out;
  out.resize((in.length() + 6) / 7 * 5);

  unsigned long num, i1, i2, j;
  for (i1 = 0, i2 = 0; i1 < in.length(); i1 += 7, i2 += 5) {
    num = 0;
    for (j = 0; j < 7; ++j) {
      num <<= 5;
      if (i1 + j < in.length()) {
        num |= decode32(in[i1 + j]) & 0x1F;
      }
    }
    for (j = 0; j < 5; ++j) {
      out[i2 + 4 - j] = num & 0x7F;
      num >>= 7;
    }
  }
  return out;
}

unsigned char bi::encode32(const unsigned char c) {
  /* pre-condition */
  assert(c < 32u);

  unsigned char d = c + ((c < 26u) ? 'a' : '0' - 26u);

  /* post-condition */
  assert((d >= 'a' && d <= 'z') || (d >= '0' && d <= '5'));

  return d;
}

unsigned char bi::decode32(const unsigned char c) {
  /* pre-condition */
  assert((c >= 'a' && c <= 'z') || (c >= '0' && c <= '5'));

  unsigned char d = c - ((c >= 'a') ? 'a' : '0' - 26u);

  /* post-condition */
  assert(d < 32u);

  return d;
}

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

    init = true;
  }
  return ops.find(op) != ops.end();
}

std::string bi::nice(const std::string& name) {
  /* translations */
  static std::regex reg;
  static std::unordered_map<std::string,std::string> ops;
  static bool init = false;
  if (!init) {
    ops["<-"] = "assign";
    ops["<~"] = "left_tilde";
    ops["~>"] = "right_tilde";
    ops["~"] = "tilde";
    ops[".."] = "range";
    ops["+"] = "add";
    ops["-"] = "subtract";
    ops["*"] = "multiply";
    ops["/"] = "divide";
    ops["<"] = "lt";
    ops[">"] = "gt";
    ops["<="] = "le";
    ops[">="] = "ge";
    ops["=="] = "eq";
    ops["!="] = "ne";
    ops["!"] = "not";
    ops["||"] = "or";
    ops["&&"] = "and";

    init = true;
  }

  /* translate operators */
  std::string str = name;
  if (ops.find(name) != ops.end()) {
    str = ops[name];
  }
  return str;
}

std::string bi::internalise(const std::string& name) {
  /* underscore on end to avoid conflicts with internal names */
  return escape_unicode(nice(name)) + "_";
}

std::string bi::escape_unicode(const std::string& str) {
  std::wstring_convert<std::codecvt_utf8_utf16<char16_t>,char16_t> converter;
  std::u16string wstr = converter.from_bytes(str);
  std::stringstream buf;
  for (wchar_t c : wstr) {
    if (c < 127) {
      buf << (char)c;
    } else {
      buf << "\\u" << std::setfill('0') << std::setw(4) << c;
    }
  }
  return buf.str();
}

std::string bi::detailed(const std::string& str) {
  std::regex reg(" *\n *\\* ?");
  std::stringstream buf;
  std::smatch match;
  std::string str1 = str;
  while (std::regex_search(str1, match, reg)) {
    buf << match.prefix() << '\n';
    str1 = match.suffix();
  }
  buf << str1;
  str1 = buf.str();
  boost::trim(str1);

  return str1;
}

std::string bi::brief(const std::string& str) {
  std::regex reg(".*?[\\.\\?\\!]");
  std::stringstream buf;
  std::smatch match;
  std::string str1 = one_line(str);
  if (std::regex_search(str1, match, reg)) {
    return one_line(match.str());
  } else {
    return "";
  }
}

std::string bi::one_line(const std::string& str) {
  std::string str1 = detailed(str);
  boost::replace_all(str1, "\n", " ");
  return str1;
}

std::string bi::anchor(const std::string& name) {
  std::string str = name;
  boost::to_lower(str);
  boost::replace_all(str, " ", "-");
  boost::replace_all(str, "_", "-");
  return str;
}

std::string bi::quote(const std::string& str, const std::string& indent) {
  std::string result = indent + str;
  boost::replace_all(result, "\n", std::string("\n") + indent);
  return result;
}
