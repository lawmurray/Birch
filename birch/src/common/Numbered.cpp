/**
 * @file
 */
#include "src/common/Numbered.hpp"

int birch::Numbered::COUNTER = 0;

birch::Numbered::Numbered() : number(++COUNTER) {
  //
}
