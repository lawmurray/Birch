#!/bin/bash
set -eov pipefail

N1=10000  # for cdf tests
N2=1000   # for gradient tests
N3=10000  # for pdf tests
N4=10000  # for conjugacy tests

eval "`grep -r "program test_basic_" src/test     | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1/"                         | sort`"
eval "`grep -r "program test_cdf_" src/test       | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N1/"                  | sort`"
eval "`grep -r "program test_grad_" src/test      | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N2 --backward false/" | sort`"
eval "`grep -r "program test_grad_" src/test      | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N2 --backward true/"  | sort`"
eval "`grep -r "program test_pdf_" src/test       | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N3 --lazy false/"     | sort`"
eval "`grep -r "program test_pdf_" src/test       | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N3 --lazy true/"      | sort`"
eval "`grep -r "program test_conjugacy_" src/test | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N4 --lazy false/"     | sort`"
eval "`grep -r "program test_conjugacy_" src/test | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N4 --lazy true/"      | sort`"
