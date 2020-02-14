/**
 * @file
 */
#pragma once

namespace libbirch {
/**
 * Cloning visitor.
 */
class Cloner {
public:
  /**
   * Constructor.
   *
   * @param label The label for new objects.
   */
  Cloner(Label* label);

  template<class Arg, class... Args>
  void visit(Arg&& arg, Args&&... args) {
    visit(arg);
    visit(args...);
  }

private:
  /**
   * The label for new objects.
   */
  Label* label;
};
}
