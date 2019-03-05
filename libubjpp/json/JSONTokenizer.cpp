/**
 * @file
 */
#include "libubjpp/json/JSONTokenizer.hpp"

libubjpp::JSONTokenizer::JSONTokenizer(const std::string& data) :
    data(data),
    begin(data.begin()),
    iter(begin),
    end(data.end()) {
  auto options = std::regex_constants::ECMAScript
      | std::regex_constants::optimize;

  regexInt = std::regex("-?(?:0|[1-9][0-9]*)", options);
  regexFrac = std::regex("\\.[0-9]+", options);
  regexExp = std::regex("[Ee][+-]?[0-9]+", options);
  regexTrue = std::regex("true", options);
  regexFalse = std::regex("false", options);
  regexNull = std::regex("null", options);
}

int libubjpp::JSONTokenizer::next(ParserState* state) {
  auto options = std::regex_constants::match_continuous;
  std::string::const_iterator first, last;
  int token;

  token = 0;
  while (token == 0 && iter != end) {
    first = iter;
    last = iter;

    if (std::regex_search(iter, end, match, regexInt, options)) {
      /* definitely matched an integer, possibly a floating point */
      first = match[0].first;
      last = match[0].second;
      token = INT64;

      /* see if the integer is followed by a fractional or exponential
       * component, in which case it is a floating point */
      if (std::regex_search(last, end, match, regexFrac, options)) {
        last = match[0].second;
        token = DOUBLE;
      }
      if (std::regex_search(last, end, match, regexExp, options)) {
        last = match[0].second;
        token = DOUBLE;
      }

      char* endptr;
      if (token == INT64) {
        int64_type intValue = std::strtol(
            data.c_str() + std::distance(begin, iter), &endptr, 10);
        state->value = int64_type(intValue);
      } else {
        assert(token == DOUBLE);
        double_type doubleValue = std::strtod(
            data.c_str() + std::distance(begin, iter), &endptr);
        state->value = double_type(doubleValue);
      }
    } else if (std::regex_search(iter, end, match, regexTrue, options)) {
      last = match[0].second;
      token = BOOL;
      state->value = bool_type(true);
    } else if (std::regex_search(iter, end, match, regexFalse, options)) {
      last = match[0].second;
      token = BOOL;
      state->value = bool_type(false);
    } else if (std::regex_search(iter, end, match, regexNull, options)) {
      last = match[0].second;
      token = NIL;
      state->value = nil_type();
    } else if (*iter == '"') {
      /* matched a string */
      token = STRING;
      buf.str("");
      last = iter + 1;
      while (last != end && *last != '"') {
        if (*last == '\\') {
          /* escape sequence */
          ++last;
          switch (*last) {
          case '"':
            buf << '"';
            break;
          case '\\':
            buf << '\\';
            break;
          case '/':
            buf << '/';
            break;
          case 'b':
            buf << '\n';
            break;
          case 'f':
            buf << '\f';
            break;
          case 'n':
            buf << '\n';
            break;
          case 'r':
            buf << '\r';
            break;
          case 't':
            buf << '\t';
            break;
          case 'u':
            buf << "\\u";  // keep unicode characters encoded for now
            break;
          default:
            assert(false);
            // unknown escape sequence
          }
        } else {
          buf << *last;
        }
        ++last;
      }
      assert(last != end);  // syntax error, unclosed string
      ++last;  // remove final quote
      state->value = string_type(buf.str());
    } else {
      /* check the next character */
      last = iter + 1;
      switch (*iter) {
      case '{':
        token = LEFT_BRACE;
        break;
      case '}':
        token = RIGHT_BRACE;
        break;
      case '[':
        token = LEFT_BRACKET;
        break;
      case ']':
        token = RIGHT_BRACKET;
        break;
      case ',':
        token = COMMA;
        break;
      case ':':
        token = COLON;
        break;
      case ' ':
        token = 0;
        break;
      case '\t':
        token = 0;
        break;
      case '\r':
        token = 0;
        break;
      case '\n':
        ++state->line;
        token = 0;
        break;
      default:
        error(state);
        token = 0;
        break;
      }
    }
    iter = last;
  }
  return token;
}
