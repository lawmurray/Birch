/**
 * @file
 */
#include "libubjpp/common/ParserState.hpp"

void ParserState::push(const libubjpp::value& value) {
  values.push(value);
}

libubjpp::value ParserState::root() {
  libubjpp::value root = std::move(values.top());
  values.pop();
  return root;
}

void ParserState::member() {
  auto value = std::move(values.top());
  values.pop();
  auto key = std::move(values.top());
  values.pop();
  auto string = key.get<libubjpp::string_type>();
  assert(string);
  auto object = values.top().get<libubjpp::object_type>();
  assert(object);
  object.get().insert(std::make_pair(string.get(), value));
}

void ParserState::element() {
  auto value = std::move(values.top());
  values.pop();
  auto array = values.top().get<libubjpp::array_type>();
  assert(array);
  array.get().push_back(value);
}

void member(ParserState* state) {
  state->member();
}

void element(ParserState* state) {
  state->element();
}
