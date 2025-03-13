# ECL-MM v1.0

ECL-MM is a fast CUDA implementation for computing a maximum matching (MM) in a bipartite graph. It operates on graphs stored in binary CSR format. A converter from matrix market to this format is included (mm2eclbp.cpp). Input files in matrix market format can be found at https://sparse.tamu.edu.

The inputs used in the paper are:

    ['amazon0312', 'r4-2e23', 'uk-2002', 'as-Skitter', 'hugebubbles-00000', 'rgg_n_2_24_s0', 'uk-2005', 'cit-Patents', 'hugetrace-00020', 'rmat22', 'wb-edu', 'coPapersDBLP', 'in-2004', 'roadNet-CA', 'web-Google', 'delaunay_n24', 'kkt_power', 'road_usa', 'wikipedia-20070206', 'europe_osm', 'kron_g500-logn21', 'soc-LiveJournal1']

To download an input, for example, amazon0312, search the name of the input in the [website](https://sparse.tamu.edu), righ-click on the 'Matrix Market' download button, copy the link, and then from the terminal, enter *wget* followed by the link:
    
    wget https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0312.tar.gz

The converter can be compiled as follows:

    g++ -std=c++11 -O3 mm2eclbp.cpp -o mm2eclbp

To convert the file graph.mm into the file graph.egr, enter:

    ./mm2eclbp graph.mm graph.egr


The ECL-MM CUDA code consists of the source files ECL-MM_10.cu, ECLgraph.h, and ECLatomic.h. See the paper listed below for a description of ECL-MM. Note that ECL-MM is protected by the 3-Clause BSD license.

The MM code can be compiled as follows:

    nvcc -O3 -arch=sm_70 -Xcompiler -fopenmp ECL-MM_10.cu -o ecl-mm

To compute the MM of the file graph.egr, enter:

    ./ecl-mm graph.egr


### Publication

Anju Mongandampulath Akathoott and Martin Burtscher. "A Bidirectional GPU Algorithm for Computing Maximum Matchings in Bipartite Graphs." Proceedings of the 39th IEEE International Parallel and Distributed Processing Symposium. June 2025. [pdf](https://userweb.cs.txstate.edu/~burtscher/papers/ipdps25b.pdf)


**Summary**: ECL-MM is an augmenting-path-based parallel algorithm for computing maximum matchings in bipartite graphs on GPUs. Its first phase computes a large initial matching very quickly. This phase does not involve path searches and is guaranteed to produce a matching that is both maximal and deterministic. The second phase computes augmenting paths. It employs parallel level-synchronous bidirectional breadth-first searches starting from unmatched vertices of both partitions of the bipartite graph. Doing so exposes significantly higher amounts of parallelism compared to single-directional searches. Moreover, it only needs to grow the search trees half-way, which reduces path overlaps as well as synchronization requirements and halves the number of steps required to form a complete augmenting path. ECL-MM improves load balancing by processing vertices at thread or warp granularity depending on their degrees. Together, these features make ECL-MM the currently fastest code for computing maximum matchings in bipartite graphs.


*This work has been supported in part by the National Science Foundation under Award Number 1955367.*
