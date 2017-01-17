/**
 * @file
 */
#pragma once

namespace bi {
enum possibly_state {
  UNTRUE = 0,
  POSSIBLE = 1,
  DEFINITE = 2
};

/**
 * A possibly behaves similarly to a boolean, but takes on one of three
 * states: definitely true, possibly true, and untrue. The usual
 * logical operators are overloaded for these.
 */
class possibly {
public:
  /**
   * Constructor.
   */
  explicit possibly(const possibly_state state) : state(state) {
    //
  }

  /**
   * Constructor from boolean.
   */
  explicit possibly(const bool state) : state(state ? DEFINITE : UNTRUE) {
    //
  }

  /**
   * Cast to boolean.
   */
  operator bool() const {
    return state == DEFINITE;
  }

  /**
   * Is this definitely true?
   */
  bool definite() const {
    return state == DEFINITE;
  }

  /**
   * Is this possibly, but not definitely, true?
   */
  bool possible() const {
    return state == POSSIBLE;
  }

  /**
   * Is this definitely untrue?
   */
  bool untrue() const {
    return state == UNTRUE;
  }

  /**
   * Logical and for possiblies.
   */
  possibly operator&&(const possibly& o) const;

  /**
   * Logical or for possiblies.
   */
  possibly operator||(const possibly& o) const;

  /**
   * Logical not for possiblies.
   */
  possibly operator!() const;

private:
  /**
   * State.
   */
  possibly_state state;
};

/*
 * Static possibilities.
 */
static possibly definite(DEFINITE);
static possibly possible(POSSIBLE);
static possibly untrue(UNTRUE);
}
