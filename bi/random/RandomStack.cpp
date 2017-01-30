/**
 * @file
 */
#include "bi/random/RandomStack.hpp"

int bi::RandomStack::push(const lambda_type& pull, const lambda_type& push) {
  pulls.push(pull);
  pushes.push(push);
  return pulls.size() - 1;
}

void bi::RandomStack::pop(const int pos) {
  while (pulls.size() > pos) {
    auto pull = pulls.top();
    auto push = pushes.top();

    pulls.pop();
    pushes.pop();

    pull();
    push();
  }
}

bi::RandomStack randomStack;
