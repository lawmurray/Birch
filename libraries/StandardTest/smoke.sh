#!/bin/bash
set -eov pipefail

N1=100  # for cdf tests
N2=10   # for gradient tests
N3=100  # for pdf tests
N4=100  # for conjugacy tests
B=100
S=0

eval "`grep -r "program test_basic_" src     | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1/"                                 | sort`"
eval "`grep -r "program test_cdf_" src       | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N1/"                          | sort`"
eval "`grep -r "program test_grad_" src      | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N2 --backward false/"         | sort`"
eval "`grep -r "program test_grad_" src      | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N2 --backward true/"          | sort`"
eval "`grep -r "program test_pdf_" src       | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N3 -B $B -S $S --lazy false/" | sort`"
eval "`grep -r "program test_pdf_" src       | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N3 -B $B -S $S --lazy true/"  | sort`"
eval "`grep -r "program test_conjugacy_" src | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N4 --lazy false/"             | sort`"
eval "`grep -r "program test_conjugacy_" src | sed -E "s/^.*program ([A-Za-z0-9_]+).*$/birch \1 -N $N4 --lazy true/"              | sort`"
