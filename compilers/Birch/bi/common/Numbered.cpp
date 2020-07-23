/**
 * @file
 */
#include "bi/common/Numbered.hpp"

int bi::Numbered::COUNTER = 0;

bi::Numbered::Numbered() : number(++COUNTER) {
  //
}

bi::Numbered::~Numbered() {
  //
}
