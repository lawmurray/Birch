/**
 * @file
 */
#pragma once

namespace libbirch {
class Label;

/**
 * Relabel an object.
 *
 * @param oldLabel The old label.
 * @param newLabel The new label.
 * @param arg The object.
 *
 * This is specialized for objects of the array, tuple, optional, fiber, and
 * smart pointer types provided by LibBirch. It is used to label a newly
 * created object that is the result of copying another, immediately after
 * that copy, or to repurpose an object for a new label when the original
 * is to be discarded.
 */
template<class Arg>
void relabel(Label* oldLabel, Label* newLabel, Arg& arg) {
  //
}

/**
 * Relabel a list of objects.
 *
 * @param oldLabel The old label.
 * @param newLabel The new label.
 * @param arg First object.
 * @param args... Remaining objects.
 */
template<class Arg, class... Args>
void relabel(Label* oldLabel, Label* newLabel, Arg& arg, Args&... args) {
  relabel(oldLabel, newLabel, arg);
  relabel(oldLabel, newLabel, args...);
}

/**
 * Relabel an empty list of objects.
 *
 * @param oldLabel The old label.
 * @param newLabel The new label.
 */
inline void relabel(Label* oldLabel, Label* newLabel) {
  //
}

}
