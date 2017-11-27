/**
 * @file
 */
#include "libubjpp/common/ParserState.hpp"

ParserState::ParserState() : failed(false) {
  //
}

libubjpp::value ParserState::root() {
  assert(!values.empty());
  libubjpp::value root = std::move(values.top());
  values.pop();
  return root;
}

void ParserState::push(const libubjpp::value& value) {
  if (!failed) {
    values.push(value);
  }
}

void ParserState::member() {
  if (!failed) {
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
}

void ParserState::element() {
  if (!failed) {
    auto value = std::move(values.top());
    values.pop();
    auto array = values.top().get<libubjpp::array_type>();
    assert(array);
    array.get().push_back(value);
  }
}

void ParserState::error() {
  failed = true;
}

void member(ParserState* state) {
  state->member();
}

void element(ParserState* state) {
  state->element();
}

void error(ParserState* state) {
  state->error();
}
