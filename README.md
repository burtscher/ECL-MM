# ECL-MM v1.0

ECL-MM is a fast CUDA implementation for computing a maximum matching (MM) in a bipartite graph. It operates on graphs stored in binary CSR format. A converter from matrix market to this format is included (mm2eclbp.cpp). Input files in matrix market format can be found at https://sparse.tamu.edu.

The converter can be compiled as follows:

    g++ -std=c++11 -O3 -march=native mm2eclbp.cpp -o mm2eclbp

To convert the file graph.mm into the file graph.egr, enter:

    ./mm2eclbp graph.mm graph.egr


The ECL-MM CUDA code consists of the source files ECL-MM_10.cu, ECLgraph.h, and ECLatomic.h. See the paper listed below for a description of ECL-MM. Note that ECL-MM is protected by the 3-Clause BSD license.

The MM code can be compiled as follows:

    nvcc -O3 -arch=sm_70 -Xcompiler -fopenmp ECL-MM_10.cu -o ecl-mm

To compute the MST of the file graph.egr, enter:

    ./ecl-mm graph.egr


### Publication

Anju Mongandampulath Akathoott and Martin Burtscher. "A Bidirectional GPU Algorithm for Computing Maximum Matchings in Bipartite Graphs." Proceedings of the 39th IEEE International Parallel and Distributed Processing Symposium. June 2025.


**Summary**: ECL-MST is ...


*This work has been supported in part by the National Science Foundation under Award Number 1955367.*
