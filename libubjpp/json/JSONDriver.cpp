/**
 * @file
 */
#include "libubjpp/json/JSONDriver.hpp"

#include "libubjpp/json/JSONTokenizer.hpp"
#include "libubjpp/common/ParserState.hpp"

#include <iostream>
#include <fstream>

extern "C" void *ParseAlloc(void *(*mallocProc)(size_t));
extern "C" void ParseFree(void* p, void (*freeProc)(void*));
extern "C" void Parse(void *parser, int token, void* value, ParserState* state);

boost::optional<libubjpp::value> libubjpp::JSONDriver::parse(
    std::istream& stream) {
  /* slurp in the whole stream */
  std::string data;
  getline(stream, data,
      std::string::traits_type::to_char_type(
          std::string::traits_type::eof()));
  return parse(data);
}

boost::optional<libubjpp::value> libubjpp::JSONDriver::parse(
    const std::string& data) {
  void* value = 0;
  int token = 0;
  ParserState state;
  JSONTokenizer tokenizer(data);
  void* parser = ParseAlloc(malloc);
  do {
    token = tokenizer.next(&state);
    Parse(parser, token, value, &state);
  } while (token > 0 && !state.failed);
  ParseFree(parser, free);
  if (!state.failed) {
    return state.root();
  } else {
    return boost::none;
  }
}
