/**
 * @file
 */
#pragma once

#include "bi/visitor/Visitor.hpp"

#include <iostream>

#define line(x) *this << indent << x << '\n'
#define start(x) *this << indent << x
#define middle(x) *this << x
#define finish(x) *this << x << '\n'

namespace bi {
/**
 * Output stream with indenting and basic output for names and literals.
 *
 * @ingroup compiler_io
 */
class indentable_ostream: public Visitor {
public:
  /**
   * Constructor.
   *
   * @param stream Base stream.
   * @param indent Initial indentation level.
   * @param header Output header only?
   */
  indentable_ostream(std::ostream& base, const int level = 0,
      const bool header = false);

  /*
   * Output operator for standard types.
   */
  bi::indentable_ostream& operator<<(const char o) {
    base << o;
    return *this;
  }

  bi::indentable_ostream& operator<<(const bool o) {
    base << o;
    return *this;
  }

  bi::indentable_ostream& operator<<(const int32_t o) {
    base << o;
    return *this;
  }

  bi::indentable_ostream& operator<<(const int64_t o) {
    base << o;
    return *this;
  }

  bi::indentable_ostream& operator<<(const size_t o) {
    base << o;
    return *this;
  }

  bi::indentable_ostream& operator<<(const float o) {
    base << o;
    return *this;
  }

  bi::indentable_ostream& operator<<(const double o) {
    base << o;
    return *this;
  }

  bi::indentable_ostream& operator<<(const char* o) {
    base << o;
    return *this;
  }

  bi::indentable_ostream& operator<<(const std::string& o) {
    base << o;
    return *this;
  }

  /*
   * Output operator for parse tree objects.
   */
  bi::indentable_ostream& operator<<(const Name* o);
  bi::indentable_ostream& operator<<(const File* o);
  bi::indentable_ostream& operator<<(const Package* o);
  bi::indentable_ostream& operator<<(const Expression* o);
  bi::indentable_ostream& operator<<(const Statement* o);
  bi::indentable_ostream& operator<<(const Type* o);
  bi::indentable_ostream& operator<<(const Location* o);

  /*
   * Append the contents of an input stream.
   */
  template<class InputStream>
  void append(const InputStream& stream) {
    base << stream.rdbuf();
  }

  /**
   * Increase indent level.
   */
  void in();

  /**
   * Decrease indent level.
   */
  void out();

protected:
  /**
   * Underlying stream.
   */
  std::ostream& base;

  /**
   * Current indent level.
   */
  int level;

  /**
   * Output header only?
   */
  bool header;

  /**
   * Current indent string.
   */
  std::string indent;
};
}
