/**
 * @file
 */
#include "bi/primitive/encode.hpp"

#include "bi/io/mangler_ostream.hpp"
#include "bi/expression/all.hpp"

#include <regex>
#include <sstream>
#include <cassert>

void bi::encode32(const std::string& in, std::string& out) {
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
}

void bi::decode32(const std::string& in, std::string& out) {
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

std::string bi::mangle(const Signature* o) {
  std::stringstream buf;
  std::string decoded, encoded;

  /* encode */
  bi::mangler_ostream stream(buf);
  stream << o->parens->strip();
  decoded = buf.str();
  encode32(decoded, encoded);

  /* construct unique name */
  buf.str("");
  buf << internalise(o);
  if (encoded.length() > 0) {
    buf << '_' << encoded << '_';
  }

  return buf.str();
}

std::string bi::internalise(const Signature* o) {
  /* translations */
  static std::regex reg;
  static std::unordered_map<std::string,std::string> ops, greeks;
  static bool init = false;

  if (!init) {
    ops["->"] = "right";
    ops["<-"] = "left";
    ops["~>"] = "condition";
    ops["<~"] = "simulate";
    ops["~"] = "sim";
    ops[".."] = "range";
    ops["!"] = "not";
    ops["&&"] = "and";
    ops["||"] = "or";
    ops["<"] = "lt";
    ops[">"] = "gt";
    ops["<="] = "le";
    ops[">="] = "ge";
    ops["=="] = "eq";
    ops["!="] = "ne";
    ops["+"] = "add";
    ops["-"] = "sub";
    ops["*"] = "mul";
    ops["/"] = "div";

    greeks["α"] = "alpha";
    greeks["β"] = "beta";
    greeks["γ"] = "gamma";
    greeks["δ"] = "delta";
    greeks["ε"] = "epsilon";
    greeks["ζ"] = "zeta";
    greeks["η"] = "eta";
    greeks["θ"] = "theta";
    greeks["ι"] = "iota";
    greeks["κ"] = "kappa";
    greeks["λ"] = "lambda";
    greeks["μ"] = "mu";
    greeks["ν"] = "nu";
    greeks["ο"] = "omicron";
    greeks["π"] = "pi";
    greeks["ρ"] = "rho";
    greeks["σ"] = "sigma";
    greeks["τ"] = "tau";
    greeks["υ"] = "upsilon";
    greeks["φ"] = "phi";
    greeks["χ"] = "chi";
    greeks["ψ"] = "psi";
    greeks["ω"] = "omega";
    greeks["Γ"] = "Gamma";
    greeks["Δ"] = "Delta";
    greeks["Θ"] = "Theta";
    greeks["Λ"] = "Lambda";
    greeks["Π"] = "Pi";
    greeks["Σ"] = "Sigma";
    greeks["Υ"] = "Upsilon";
    greeks["Φ"] = "Phi";
    greeks["Ψ"] = "Psi";
    greeks["Ω"] = "Omega";

    std::stringstream buf;
    buf << '(';
    for (auto iter = greeks.begin(); iter != greeks.end(); ++iter) {
      if (iter != greeks.begin()) {
        buf << '|';
      }
      buf << iter->first;
    }
    buf << ')';
    reg = std::regex(buf.str());

    init = true;
  }

  std::string str = o->name->str();

  /* translate operators */
  if (ops.find(str) != ops.end()) {
    str = ops[str] + '_';
  }

  /* translate Greek letters */
  std::stringstream buf;
  std::smatch match;
  while (std::regex_search(str, match, reg)) {
    assert(greeks.find(match.str()) != greeks.end());
    buf << match.prefix();
    buf << greeks[match.str()];
    str = match.suffix();
  }
  buf << str;

  return buf.str();
}

std::string bi::escape(const std::string& str) {
  std::regex reg("[^0-9a-zA-Z_]");
  std::stringstream buf;
  std::smatch match;
  std::string str1 = str;
  while (std::regex_search(str1, match, reg)) {
    buf << match.prefix() << '\\' << match.str();
    str1 = match.suffix();
  }
  buf << str1;

  return buf.str();
}

bool bi::isSimple(const char c) {
  return (('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z')
      || ('0' <= c && c <= '9') || c == '_');
}
