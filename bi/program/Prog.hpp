/**
 * @file
 */
#pragma once

#include "bi/common/Located.hpp"

namespace bi {
class Cloner;
class Modifier;
class Visitor;

/**
 * Program.
 *
 * @ingroup compiler_program
 */
class Prog: public Located {
public:
  /**
   * Constructor.
   */
  Prog(shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Prog() = 0;

  /**
   * Accept cloning visitor.
   *
   * @param v The visitor.
   *
   * @return Cloned (and potentially modified) statement.
   */
  virtual Prog* acceptClone(Cloner* visitor) const = 0;

  /**
   * Accept modifying visitor.
   *
   * @param v The visitor.
   *
   * @return Modified statement.
   */
  virtual void acceptModify(Modifier* visitor) = 0;

  /**
   * Accept read-only visitor.
   *
   * @param v The visitor.
   */
  virtual void accept(Visitor* visitor) const = 0;

  /*
   * Bool cast to check for non-empty statement.
   */
  virtual operator bool() const;

  /*
   * Partial order comparison operators for comparing statements in terms of
   * specialisation.
   *
   * The first two are the most commonly used, and so overridden by derived
   * classes. The remainder are expressed in terms of these.
   */
  virtual bool operator<=(Prog& o) = 0;
  virtual bool operator==(const Prog& o) const = 0;
  bool operator<(Prog& o);
  bool operator>(Prog& o);
  bool operator>=(Prog& o);
  bool operator!=(Prog& o);
};
}

inline bi::Prog::Prog(shared_ptr<Location> loc) :
    Located(loc) {
  //
}

inline bi::Prog::~Prog() {
  //
}

inline bi::Prog::operator bool() const {
  return true;
}

inline bool bi::Prog::operator<(Prog& o) {
  return *this <= o && *this != o;
}

inline bool bi::Prog::operator>(Prog& o) {
  return o <= *this && o != *this;
}

inline bool bi::Prog::operator>=(Prog& o) {
  return o <= *this;
}

inline bool bi::Prog::operator!=(Prog& o) {
  return !(*this == o);
}
