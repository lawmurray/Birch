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
extern "C" void Parse(void *parser, int token, int value, ParserState* state);

libubjpp::value libubjpp::JSONDriver::parse(const std::string &f) {
  /* slurp in the whole file */
  std::ifstream in(f);
  std::string data;
  getline(in, data,
      std::string::traits_type::to_char_type(
          std::string::traits_type::eof()));

  /* parse */
  int value, token;
  ParserState state;
  JSONTokenizer tokenizer(data);
  void* parser = ParseAlloc(malloc);
  do {
    token = tokenizer.next(&state);
    Parse(parser, token, value, &state);
  } while (token > 0);
  ParseFree(parser, free);

  return state.root();
}
