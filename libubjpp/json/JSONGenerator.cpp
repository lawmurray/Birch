/**
 * @file
 */
#include "libubjpp/json/JSONGenerator.hpp"

libubjpp::JSONGenerator::JSONGenerator(std::ostream& stream) :
    stream(stream),
    level(0) {
  //
}

void libubjpp::JSONGenerator::apply(const value& value) {
  boost::apply_visitor(*this, value.unwrap());
}

void libubjpp::JSONGenerator::operator()(const object_type& value) {
  write("{\n");
  in();
  for (auto iter = value.begin(); iter != value.end(); ++iter) {
    if (iter != value.begin()) {
      write(",\n");
    }
    indent();
    apply(iter->first);
    write(": ");
    apply(iter->second);
  }
  write("\n");
  out();
  indent();
  write("}");
}

void libubjpp::JSONGenerator::operator()(const array_type& value) {
  /* arrays elements are output on one line each, unless the array contains
   * no further arrays or elements, in which case they are output all on one
   * line; check for this case first */
  bool isLeaf = true;
  for (auto iter = value.begin(); isLeaf && iter != value.end(); ++iter) {
    isLeaf = !iter->get<array_type>() && !iter->get<object_type>();
  }

  write("[");
  if (!isLeaf) {
    write("\n");
    in();
    indent();
  }
  for (auto iter = value.begin(); iter != value.end(); ++iter) {
    if (iter != value.begin()) {
      write(",");
      if (!isLeaf) {
        write("\n");
        indent();
      } else {
        write(" ");
      }
    }
    apply(*iter);
  }
  if (!isLeaf) {
    write("\n");
    out();
    indent();
  }
  write("]");
}

void libubjpp::JSONGenerator::operator()(const string_type& value) {
  write("\"");
  escape(value);
  write("\"");
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
  write(value ? "true" : "false");
}

void libubjpp::JSONGenerator::operator()(const nil_type& value) {
  write("null");
}

void libubjpp::JSONGenerator::operator()(const noop_type& value) {
  //
}

void libubjpp::JSONGenerator::write(const std::string& value) {
  stream << value;
}

void libubjpp::JSONGenerator::escape(const std::string& value) {
  std::stringstream buf;
  for (char c : value) {
    switch (c) {
    case '"':
      buf << "\\\"";
      break;
    case '\\':
      buf << "\\\\";
      break;
    case '/':
      buf << "\\/";
      break;
    case '\b':
      buf << "\\b";
      break;
    case '\f':
      buf << "\\f";
      break;
    case '\n':
      buf << "\\n";
      break;
    case '\r':
      buf << "\\r";
      break;
    case '\t':
      buf << "\\t";
      break;
    default:
      buf << c;
    }
  }
  stream << buf.str();
}

void libubjpp::JSONGenerator::in() {
  ++level;
}

void libubjpp::JSONGenerator::out() {
  --level;
}

void libubjpp::JSONGenerator::indent() {
  for (int i = 0; i < level; ++i) {
    stream << "  ";
  }
}
