# Python wrapper of C++ Hungarian algorithm implementation

This code was borrowed from https://github.com/jfrelinger/cython-munkres-wrapper.

There exists several implementations of the `Hungarian` algorithm accessible from `Python`:

 1. https://github.com/bmc/munkres
 2. http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
 3. https://github.com/jfrelinger/cython-munkres-wrapper
 
This one (3.) is, by some non-negligible difference, the fastest based on the following somehow naive experiment (experiment run on the `hudlrd-devbox`):

| Implementation |   5 x 5 matrix   |  25 x 25 matrix  | 125 x 125 matrix | 625 x 625 matrix |
|:--------------:|:----------------:|:----------------:|:----------------:|:----------------:|
| 1.              | 110 µs           | 14.9 ms          |  2.3 s           | -                |
| 2.              | 313 µs           |  2.7 ms          | 72.8 ms          | 16.4 s           |
| 3.              | 6.1 µs           |  132 µs          | 19.9 ms          |  5.2 s           |
