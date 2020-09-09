/**
 * @file
 */
#pragma once

#include "src/visitor/Visitor.hpp"

#define line(x) *this << indent << x << '\n'
#define start(x) *this << indent << x
#define middle(x) *this << x
#define finish(x) *this << x << '\n'

namespace birch {
/**
 * Output stream with indenting and basic output for names and literals.
 *
 * @ingroup io
 */
class IndentableGenerator: public Visitor {
public:
  /**
   * Constructor.
   *
   * @param stream Base stream.
   * @param indent Initial indentation level.
   * @param header Output header only?
   */
  IndentableGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  /*
   * Output operator for standard types.
   */
  birch::IndentableGenerator& operator<<(const char o) {
    base << o;
    return *this;
  }

  birch::IndentableGenerator& operator<<(const bool o) {
    base << o;
    return *this;
  }

  birch::IndentableGenerator& operator<<(const int32_t o) {
    base << o;
    return *this;
  }

  birch::IndentableGenerator& operator<<(const int64_t o) {
    base << o;
    return *this;
  }

  birch::IndentableGenerator& operator<<(const size_t o) {
    base << o;
    return *this;
  }

  birch::IndentableGenerator& operator<<(const float o) {
    base << o;
    return *this;
  }

  birch::IndentableGenerator& operator<<(const double o) {
    base << o;
    return *this;
  }

  birch::IndentableGenerator& operator<<(const char* o) {
    base << o;
    return *this;
  }

  birch::IndentableGenerator& operator<<(const std::string& o) {
    base << o;
    return *this;
  }

  /*
   * Output operator for parse tree objects.
   */
  birch::IndentableGenerator& operator<<(const Name* o);
  birch::IndentableGenerator& operator<<(const File* o);
  birch::IndentableGenerator& operator<<(const Package* o);
  birch::IndentableGenerator& operator<<(const Expression* o);
  birch::IndentableGenerator& operator<<(const Statement* o);
  birch::IndentableGenerator& operator<<(const Type* o);
  birch::IndentableGenerator& operator<<(const Location* o);

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
