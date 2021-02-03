/**
 * @file
 */
#include "src/primitive/string.hpp"

std::string birch::encode32(const std::string& in) {
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

std::string birch::decode32(const std::string& in) {
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

unsigned char birch::encode32(const unsigned char c) {
  /* pre-condition */
  assert(c < 32u);

  unsigned char d = c + ((c < 26u) ? 'a' : '0' - 26u);

  /* post-condition */
  assert((d >= 'a' && d <= 'z') || (d >= '0' && d <= '5'));

  return d;
}

unsigned char birch::decode32(const unsigned char c) {
  /* pre-condition */
  assert((c >= 'a' && c <= 'z') || (c >= '0' && c <= '5'));

  unsigned char d = c - ((c >= 'a') ? 'a' : '0' - 26u);

  /* post-condition */
  assert(d < 32u);

  return d;
}

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
    ops["-"] = "subtract_";
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

std::string birch::internalise(const std::string& name) {
  return nice(name);
}

std::string birch::escape_unicode(const std::string& str) {
  /* ideally we would write the UTF-8 encoded character (works with clang at
   * least), next best would be a \u0000 universal character name (works with
   * gcc at least), but failing that (Intel compiler) we just write _u0000 as
   * the name */
  std::wstring_convert<std::codecvt_utf8_utf16<char16_t>,char16_t> converter;
  std::u16string wstr = converter.from_bytes(str);
  std::stringstream buf;
  for (wchar_t c : wstr) {
    if (c < 127) {
      buf << (char)c;
    } else {
      buf << "_u" << std::setfill('0') << std::setw(4) << c;
    }
  }
  return buf.str();
}

std::string birch::detailed(const std::string& str) {
  std::regex reg(" *\n *\\* ?");
  std::stringstream buf;
  std::smatch match;
  std::string str1 = str;
  while (std::regex_search(str1, match, reg)) {
    buf << match.prefix() << '\n';
    str1 = match.suffix();
  }
  buf << str1;
  return buf.str();
}

std::string birch::brief(const std::string& str) {
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

std::string birch::one_line(const std::string& str) {
  return std::regex_replace(detailed(str), std::regex("\\n"), " ");
}

std::string birch::anchor(const std::string& name) {
  return std::regex_replace(lower(name), std::regex(" |_"), "-");
}

std::string birch::quote(const std::string& str, const std::string& indent) {
  return std::regex_replace(indent + str, std::regex("\\n"), std::string("\n") + indent);
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
  return "birch-" + lower(name);
}

std::string birch::canonical(const std::string& name) {
  return std::regex_replace(tar(name), std::regex("-"), "_");
}
