/**
 * @file
 */
#pragma once

#include "bi/visitor/Visitor.hpp"
#include "bi/primitive/owned_ptr.hpp"
#include "bi/primitive/unique_ptr.hpp"
#include "bi/primitive/shared_ptr.hpp"

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
class indentable_ostream : public Visitor {
public:
  /**
   * Constructor.
   *
   * @param stream Base stream.
   * @param indent Initial indentation level.
   * @param header Output header only?
   */
  indentable_ostream(std::ostream& base, const int level = 0, const bool header = false);

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
   * Output operator for pointers.
   */
  template<class T>
  bi::indentable_ostream& operator<<(const T* arg) {
    arg->accept(this);
    return *this;
  }

  template<class T>
  bi::indentable_ostream& operator<<(const owned_ptr<T>& arg) {
    *this << arg.get();
    return *this;
  }

  template<class T>
  bi::indentable_ostream& operator<<(const unique_ptr<T>& arg) {
    *this << arg.get();
    return *this;
  }

  template<class T>
  bi::indentable_ostream& operator<<(const shared_ptr<T>& arg) {
    *this << arg.get();
    return *this;
  }

  /*
   * Output operator for locations.
   */
  bi::indentable_ostream& operator<<(const bi::Location* o);

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

inline void bi::indentable_ostream::in() {
  ++level;
  indent.append(2, ' ');
}

inline void bi::indentable_ostream::out() {
  /* pre-condition */
  assert(level > 0);

  --level;
  indent.resize(2*level);
}
