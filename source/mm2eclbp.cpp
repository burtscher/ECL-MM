/*
This file is part of ECL-MM.

Copyright (c) 2025, Anju Mongandampulath Akathoott and Martin Burtscher

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://cs.txstate.edu/~burtscher/research/ECL-MM/ and at https://github.com/burtscher/ECL-MM.

Publication: This work is described in detail in the following paper.
Anju Mongandampulath Akathoott and Martin Burtscher. "A Bidirectional GPU Algorithm for Computing Maximum Matchings in Bipartite Graphs." Proceedings of the 39th IEEE International Parallel and Distributed Processing Symposium. June 2025.
*/


#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <utility>
#include <tuple>
#include <algorithm>
#include "ECLgraph.h"


int main(int argc, char* argv[])
{
  printf("MatrixMarket to ECL Bipartite Graph Converter\n\n");

  if (argc != 3) {fprintf(stderr, "USAGE: %s input_file_name output_file_name\n\n", argv[0]);  exit(-1);}

  FILE* fin = fopen(argv[1], "rt");  if (fin == NULL) {fprintf(stderr, "ERROR: could not open input file %s\n\n", argv[1]);  exit(-1);}

  char line[256];
  char* ptr = line;
  size_t linesize = 256;
  int cnt = getline(&ptr, &linesize, fin);

  if (cnt < 30) {fprintf(stderr, "ERROR: could not read first line\n\n");  exit(-1);}
  if (strstr(line, "%%MatrixMarket") == 0) {fprintf(stderr, "ERROR: first line does not contain \"%%%%MatrixMarket\"\n\n");  exit(-1);}
  if (strstr(line, "matrix") == 0) {fprintf(stderr, "ERROR: first line does not contain \"matrix\"\n\n");  exit(-1);}
  if (strstr(line, "coordinate") == 0) {fprintf(stderr, "ERROR: first line does not contain \"coordinate\"\n\n");  exit(-1);}
  if ((strstr(line, "general") == 0) && (strstr(line, "symmetric") == 0)) {fprintf(stderr, "ERROR: first line does not contain \"general\" or \"symmetric\"\n\n");  exit(-1);}
  if ((strstr(line, "real") == 0) && (strstr(line, "integer") == 0) && (strstr(line, "pattern") == 0)) {fprintf(stderr, "ERROR: first line does not contain \"integer\" or \"pattern\"\n\n");  exit(-1);}
  bool hasweights = false;
  if (strstr(line, "wt") != 0) hasweights = true;
  bool symmetric = false;
  if (strstr(line, "symmetric") != 0) symmetric = true;

  while (((cnt = getline(&ptr, &linesize, fin)) > 0) && (strstr(line, "%") != 0)) {}
  if (cnt < 3) {fprintf(stderr, "ERROR: could not find non-comment line\n\n");  exit(-1);}

  int nodes, nA, nB, edges;
  cnt = sscanf(line, "%d %d %d", &nA, &nB, &edges);
  if ((cnt != 3) || (nA < 1) || (edges < 0) || (nA != nB)) {fprintf(stderr, "ERROR: failed to parse first data line\n\n");  exit(-1);}

  nodes = nA + nB;
  printf("%s\t#name\n", argv[1]);
  printf("%d\t#nodes (A + B)\n", nodes);
  printf("%d\t#edges in file\n", edges);

  ECLgraph g;

  if (!hasweights) {
    printf("no\t#weights\n");

    int cnt = 0, src, dst, extraEdges = 0;
    std::vector<std::pair<int, int>> v;
    while (fscanf(fin, "%d %d", &src, &dst) == 2) {
      cnt++;
      if ((src < 1) || (src > nodes)) {fprintf(stderr, "ERROR: source out of range\n\n");  exit(-1);}
      if ((dst < 1) || (dst > nodes)) {fprintf(stderr, "ERROR: source out of range\n\n");  exit(-1);}
      v.push_back(std::make_pair(src - 1, nA + dst - 1));
      v.push_back(std::make_pair(nA + dst - 1, src - 1));
      if (symmetric && (src != dst)) {
        v.push_back(std::make_pair(dst - 1, nA + src - 1));
        v.push_back(std::make_pair(nA + src - 1, dst - 1));
        extraEdges++;
      }
    }
    fclose(fin);
    if (cnt != edges) {fprintf(stderr, "ERROR: failed to read correct number of edges\n\n");  exit(-1);}

    g.nodes = nodes;
    g.edges = 2 * (edges + extraEdges);
    g.nindex = (int*)calloc(nodes + 1, sizeof(int));
    g.nlist = (int*)malloc(g.edges * sizeof(int));
    g.eweight = NULL;
    if ((g.nindex == NULL) || (g.nlist == NULL)) {fprintf(stderr, "ERROR: memory allocation failed\n\n");  exit(-1);}

    std::sort(v.begin(), v.end());

    g.nindex[0] = 0;
    for (int i = 0; i < g.edges; i++) {
      int src = v[i].first;
      int dst = v[i].second;
      g.nindex[src + 1] = i + 1;
      g.nlist[i] = dst;
    }
  } else {
    printf("yes\t#weights, but ignoring for bipartite graph\n");
    int cnt = 0, src, dst, wei, extraEdges = 0; // Read and ignore wei
    std::vector<std::pair<int, int>> v;
    while (fscanf(fin, "%d %d %f", &src, &dst, &wei) == 3) {
      cnt++;
      if ((src < 1) || (src > nodes)) {fprintf(stderr, "ERROR: source out of range\n\n");  exit(-1);}
      if ((dst < 1) || (dst > nodes)) {fprintf(stderr, "ERROR: source out of range\n\n");  exit(-1);}
      v.push_back(std::make_pair(src - 1, nA + dst - 1));
      v.push_back(std::make_pair(nA + dst - 1, src - 1));
      if (symmetric && (src != dst)) {
        v.push_back(std::make_pair(dst - 1, nA + src - 1));
        v.push_back(std::make_pair(nA + src - 1, dst - 1));
        extraEdges++;
      }
    }
    fclose(fin);
    if (cnt != edges) {fprintf(stderr, "ERROR: failed to read correct number of edges: edges expected = %d, seen = %d\n\n", edges, cnt);  exit(-1);}

    g.nodes = nodes;
    g.edges = 2 * (edges + extraEdges);
    g.nindex = (int*)calloc(nodes + 1, sizeof(int));
    g.nlist = (int*)malloc(g.edges * sizeof(int));
    g.eweight = NULL;
    if ((g.nindex == NULL) || (g.nlist == NULL)) {fprintf(stderr, "ERROR: memory allocation failed\n\n");  exit(-1);}

    std::sort(v.begin(), v.end());

    g.nindex[0] = 0;
    for (int i = 0; i < g.edges; i++) {
      int src = v[i].first;
      int dst = v[i].second;
      g.nindex[src + 1] = i + 1;
      g.nlist[i] = dst;
    }
  }

  for (int i = 1; i < (nodes + 1); i++) {
    g.nindex[i] = std::max(g.nindex[i - 1], g.nindex[i]);
  }

  writeECLgraph(g, argv[2]);
  freeECLgraph(g);

  return 0;
}
