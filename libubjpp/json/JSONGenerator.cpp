/**
 * @file
 */
#include "libubjpp/json/JSONGenerator.hpp"

libubjpp::JSONGenerator::JSONGenerator(std::ostream& stream) :
    stream(stream),
    level(0) {
  //
}

void libubjpp::JSONGenerator::write(const value& value) {
  boost::apply_visitor(*this, value.get());
}

void libubjpp::JSONGenerator::operator()(const object_type& value) {
  stream << "{\n";
  for (auto iter = value.begin(); iter != value.end(); ++iter) {
    if (iter != value.begin()) {
      stream << ",\n";
    }
    write(iter->first);
    stream << ": ";
    write(iter->second);
  }
  stream << "\n}";
}

void libubjpp::JSONGenerator::operator()(const array_type& value) {
  stream << "[\n";
  for (auto iter = value.begin(); iter != value.end(); ++iter) {
    if (iter != value.begin()) {
      stream << ",\n";
    }
    write(*iter);
  }
  stream << "\n]";
}

void libubjpp::JSONGenerator::operator()(const string_type& value) {
  stream << '"' << value << '"';
}

void libubjpp::JSONGenerator::operator()(const float_type& value) {
  stream << value;
}

void libubjpp::JSONGenerator::operator()(const double_type& value) {
  stream << value;
}

void libubjpp::JSONGenerator::operator()(const int8_type& value) {
  stream << value;
}

void libubjpp::JSONGenerator::operator()(const uint8_type& value) {
  stream << value;
}

void libubjpp::JSONGenerator::operator()(const int16_type& value) {
  stream << value;
}

void libubjpp::JSONGenerator::operator()(const int32_type& value) {
  stream << value;
}

void libubjpp::JSONGenerator::operator()(const int64_type& value) {
  stream << value;
}

void libubjpp::JSONGenerator::operator()(const bool_type& value) {
  stream << (value ? "true" : "false");
}

void libubjpp::JSONGenerator::operator()(const nil_type& value) {
  stream << "null";
}

void libubjpp::JSONGenerator::operator()(const noop_type& value) {
  //
}
