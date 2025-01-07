/*
ECL-MM: This code computes a maximum matching in a bipartite graph.

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
#include <string>
#include <time.h>
#include <algorithm>
#include <climits>
#include <cassert>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <sstream>
#include "ECLatomic.h"
#include "ECLgraph.h"

static const int Device = 0;
static const int TPB = 256; // Threads per block
static const int NONE = -1; 

static const int threshold = 16;

static bool goAgain, apFound;
static int sizeOfA, sizeOfB, numEdges, apSearch_itrCount, totalNumPaths, totalPathLengths, wlSize, nextWlSize, startIndexOfB, numPathsAugmentedInCurItr, n; // n = sizeOfA + sizeOfB
static int *mate;

static int *d_nbrIndx, *d_edge, *d_mate, *d_nextOption, *d_itrFlag, *d_endPoint, *d_parent, *d_source, *d_workList, *d_nextWorkList;
static int *d_wlSize, *d_nextWlSize, *d_totalPathLengths, *d_totalNumPaths;
static bool *d_apFound, *d_goAgain;

struct CPUTimer
{
  timeval beg, end;
  CPUTimer() {}
  ~CPUTimer() {}
  void start() {gettimeofday(&beg, NULL);}
  double elapsed() {gettimeofday(&end, NULL); return end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;}
};


static void printGraphSpec(char* fileName)
{
  std::string shortName(fileName);
  std::istringstream iss(shortName);
  std::string s;
  getline(iss, s, '.');
  std::istringstream iss2(s);
  std::string token;
  while (getline(iss2, token, '/'));
  printf("input=%s\n", token.c_str());
  printf("|A| = %d\n|B| = %d\n|E| = %d\n", sizeOfA, sizeOfB, numEdges);
}


static __global__ void populateWL_kernel(const int* const __restrict__ d_mate, const int* const __restrict__ d_nbrIndx, int* __restrict__ d_parent, int* __restrict__ d_source, int* __restrict__ d_workList, int n, int* __restrict__ d_wlSize)
{
  int v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v < n) {
    if (d_mate[v] == NONE && (d_nbrIndx[v + 1] != d_nbrIndx[v])) {
      int index = atomicAdd(d_wlSize, 1);
      d_workList[index] = v;
      d_parent[v] = v;
      d_source[v] = v;
    }
  }
}


static __global__ void recreateFrontier(int* __restrict__ d_workList, int* __restrict__ d_nextWorkList, int* __restrict__ d_parent, int* __restrict__ d_source, const int* const __restrict__ d_endPoint, int* __restrict__ d_wlSize, int* __restrict__ d_nextWlSize, const int n, const int sizeOfA, const int startIndexOfB)
{
  int v = blockDim.x * blockIdx.x + threadIdx.x; 
  if ((v < n) && (d_source[v] != NONE)) {
    if (d_endPoint[d_source[v]] == NONE) { // v is an active vertex
      if ((v < sizeOfA && d_source[v] < sizeOfA) || (v >= startIndexOfB && d_source[v] >= startIndexOfB)) {
        int index = atomicAdd(d_wlSize, 1);
        d_workList[index] = v;
      } else {
        int index = atomicAdd(d_nextWlSize, 1);
        d_nextWorkList[index] = v;
      }
    } else {
      // v is a vertex in a dead tree
      d_parent[v] = NONE;
      d_source[v] = NONE;
    }
  } 
}


static void reuse()
{
  wlSize = 0;
  nextWlSize = 0;
  cudaMemcpyAsync(d_wlSize, &wlSize, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_nextWlSize, &nextWlSize, sizeof(int), cudaMemcpyHostToDevice);
  recreateFrontier<<<(n + TPB - 1) / TPB, TPB>>>(d_workList, d_nextWorkList, d_parent, d_source, d_endPoint, d_wlSize, d_nextWlSize, n, sizeOfA, startIndexOfB);
  cudaMemcpyAsync(&wlSize, d_wlSize, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&nextWlSize, d_nextWlSize, sizeof(int), cudaMemcpyDeviceToHost);
}


static void setUp()
{
  if (numPathsAugmentedInCurItr > 0) {
    reuse();
  }
  else {
    cudaMemset(d_endPoint, NONE, n * sizeof(int));
    cudaMemset(d_parent, NONE, n * sizeof(int));
    cudaMemset(d_source, NONE, n * sizeof(int));
    wlSize = 0;
    nextWlSize = 0;
    cudaMemcpyAsync(d_wlSize, &wlSize, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_nextWlSize, &nextWlSize, sizeof(int), cudaMemcpyHostToDevice);
    populateWL_kernel<<<(n + TPB - 1) / TPB, TPB>>>(d_mate, d_nbrIndx, d_parent, d_source, d_workList, n, d_wlSize);
    cudaMemcpy(&wlSize, d_wlSize, sizeof(int), cudaMemcpyDeviceToHost);
  }
}


static __global__ void levelOne_kernel(const int* const __restrict__ d_wlSize, const int* const __restrict__ d_workList, int* __restrict__ d_source, int* __restrict__ d_endPoint, const int* const __restrict__ d_nbrIndx, const int* const __restrict__ d_edge, const int* const __restrict__ d_mate, int* __restrict__ d_parent, int* __restrict__ d_nextWlSize, int* __restrict__ d_nextWorkList, const int sizeOfA, const int startIndexOfB, bool* __restrict__ d_apFound)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (__any_sync(~0, i < (*d_wlSize))) {
    int beg, end, v, src_v, mateOf_v;
    int deg = -1;
    if (i < (*d_wlSize)) { 
      v = d_workList[i];
      src_v = atomicRead(&d_source[v]);
      mateOf_v = d_mate[v];
      if (atomicRead(&d_endPoint[src_v]) == NONE) {
        beg = d_nbrIndx[v];
        end = d_nbrIndx[v + 1];
        deg = end - beg;
        if (deg < threshold) { 
          // one thread does work:
          for (int j = beg; j < end; j++) {
            int nbr = d_edge[j];
            if (mateOf_v != nbr) { // (v, nbr) must be an unmatched edge
              int src_nbr = atomicRead(&d_source[nbr]);
              if (src_nbr != NONE) { // potential middle edge of an AP
                if ((src_v < sizeOfA && src_nbr >= startIndexOfB) || (src_v >= startIndexOfB && src_nbr < sizeOfA)) {
                  int v1, v2, src_v1, src_v2;
                  if (src_v < sizeOfA) {
                    v1 = v;
                    v2 = nbr;
                    src_v1 = src_v;
                    src_v2 = src_nbr;
                  } else {
                    v1 = nbr;
                    v2 = v;
                    src_v1 = src_nbr;
                    src_v2 = src_v;
                  }
                  int valToSet1;
                  if (d_mate[v1] != NONE)
                    valToSet1 = d_mate[v1];
                  else
                    valToSet1 = v1;
                  if ((atomicCAS(&(d_endPoint[src_v1]), NONE, valToSet1)) == NONE) { // indicates firstUpdateSuccess
                    int valToSet2;
                    if (d_mate[v2] != NONE)
                      valToSet2 = d_mate[v2];
                    else
                      valToSet2 = v2;

                    if ((atomicCAS(&(d_endPoint[src_v2]), NONE, valToSet2)) == NONE) { // indicates secUpdatesuccess
                      if (d_parent[v1] != v1) {
                        d_parent[v1] = v2;
                        atomicWrite(&d_endPoint[src_v2], v1);
                      } else {
                        d_parent[v2] = v1;
                        atomicWrite(&d_endPoint[src_v1], v2);
                      }
                      atomicWrite(d_apFound, true);
                      break;
                    } else {
                      atomicWrite(&d_endPoint[src_v1], NONE); // Resetting the first update, as this thread could not write to the second location successfully
                    }
                  }
                }
              } else {
                if ((atomicCAS(&(d_parent[nbr]), NONE, v)) == NONE) { // success
                  atomicWrite(&d_source[nbr], src_v);
                  int index = atomicAdd(d_nextWlSize, 1);
                  d_nextWorkList[index] = nbr;
                }
              }
            }
          }
        } 
      } 
    }
    // Work by threads with deg >= threshold:
    const int WS = 32; //warp size
    const int lane = threadIdx.x % WS;
    int bal = __ballot_sync(~0, deg >= threshold);
    while (bal != 0) {
      const int who = __ffs(bal) - 1;
      bal &= bal - 1;
      const int wbeg = __shfl_sync(~0, beg, who);
      const int wend = __shfl_sync(~0, end, who);
      const int wv = __shfl_sync(~0, v, who);
      const int wmateOf_v = __shfl_sync(~0, mateOf_v, who);
      const int wsrc_v = __shfl_sync(~0, src_v, who);
      for (int j = wbeg + lane; j < wend; j += WS) {
        int nbr = d_edge[j];
        if (wmateOf_v != nbr) { // (wv, nbr) must be an unmatched edge
          int src_nbr = atomicRead(&d_source[nbr]);
          if (src_nbr != NONE) { // potential middle edge of an AP
            if ((wsrc_v < sizeOfA && src_nbr >= startIndexOfB) || (wsrc_v >= startIndexOfB && src_nbr < sizeOfA)) {
              int v1, v2, src_v1, src_v2;
              if (wsrc_v < sizeOfA) {
                v1 = wv;
                v2 = nbr;
                src_v1 = wsrc_v;
                src_v2 = src_nbr;
              } else {
                v1 = nbr;
                v2 = wv;
                src_v1 = src_nbr;
                src_v2 = wsrc_v;
              }
              int valToSet1;
              if (d_mate[v1] != NONE)
                valToSet1 = d_mate[v1];
              else
                valToSet1 = v1;
              if ((atomicCAS(&(d_endPoint[src_v1]), NONE, valToSet1)) == NONE) { // indicates firstUpdateSuccess
                int valToSet2;
                if (d_mate[v2] != NONE)
                  valToSet2 = d_mate[v2];
                else
                  valToSet2 = v2;

                if ((atomicCAS(&(d_endPoint[src_v2]), NONE, valToSet2)) == NONE) { // indicates secUpdatesuccess
                  if (d_parent[v1] != v1) {
                    d_parent[v1] = v2;
                    atomicWrite(&d_endPoint[src_v2], v1);
                  } else {
                    d_parent[v2] = v1;
                    atomicWrite(&d_endPoint[src_v1], v2);
                  }
                  atomicWrite(d_apFound, true);
                  break;
                } else {
                  atomicWrite(&d_endPoint[src_v1], NONE); // Resetting the first update, as this thread could not write to the second location successfully
                }
              }
            }
          } else {
            if ((atomicCAS(&(d_parent[nbr]), NONE, wv)) == NONE) { // success
              atomicWrite(&d_source[nbr], wsrc_v);
              int index = atomicAdd(d_nextWlSize, 1);
              d_nextWorkList[index] = nbr;
            }
          }
        }
      }
    }
  }
}


static __global__ void levelTwo_kernel(const int* const __restrict__ d_nextWlSize, const int* const __restrict__ d_nextWorkList, const int* const __restrict__ d_mate, int* __restrict__ d_source, int* __restrict__ d_parent, int* __restrict__ d_endPoint, int* __restrict__ d_wlSize, int* __restrict__ d_workList, bool* __restrict__ d_goAgain, bool* __restrict__ d_apFound, int* __restrict__ d_totalPathLengths, const int sizeOfA, const int startIndexOfB)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < (*d_nextWlSize)) {
    int v = d_nextWorkList[i];
    int nbr = d_mate[v];
    if (d_source[nbr] != NONE) {
      if ((d_source[v] < sizeOfA && d_source[nbr] >= startIndexOfB) || (d_source[v] >= startIndexOfB && d_source[nbr] < sizeOfA)) {
        // Middle edge. Process only in one direction.
        int v1, v2;
        if (d_source[v] < sizeOfA) {
          v1 = v;
          v2 = nbr;
        } else {
          v1 = nbr;
          v2 = v;
        }
        if (atomicCAS(&(d_endPoint[d_source[v1]]), NONE, v1) == NONE) { // indicates firstUpdateSuccess
          if (atomicCAS(&(d_endPoint[d_source[v2]]), NONE, v2) == NONE) {
            atomicWrite(d_apFound, true);
            atomicAdd(d_totalPathLengths, 1);
          } else {
            d_endPoint[d_source[v1]] = NONE; // Resetting the first update, as this thread could not write to the second location successfully
          }
        }
      }
    } else {
      d_parent[nbr] = v;
      d_source[nbr] = d_source[v];
      atomicWrite(d_goAgain, true);
      int index = atomicAdd(d_wlSize, 1);
      d_workList[index] = nbr;
    }
  }
}


static void searchForAP()
{
  setUp();
  goAgain = true;
  while (goAgain) {
    goAgain = false;
    cudaMemcpyAsync(d_goAgain, &goAgain, sizeof(bool), cudaMemcpyHostToDevice);
    if (wlSize > 0) {
      levelOne_kernel<<<(wlSize + TPB - 1) / TPB, TPB>>>(d_wlSize, d_workList, d_source, d_endPoint, d_nbrIndx, d_edge, d_mate, d_parent, d_nextWlSize, d_nextWorkList, sizeOfA, startIndexOfB, d_apFound);
    }
    wlSize = 0;
    cudaMemcpyAsync(d_wlSize, &wlSize, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&nextWlSize, d_nextWlSize, sizeof(int), cudaMemcpyDeviceToHost);
    if (nextWlSize > 0) {
      levelTwo_kernel<<<(nextWlSize + TPB - 1) / TPB, TPB>>>(d_nextWlSize, d_nextWorkList, d_mate, d_source, d_parent, d_endPoint, d_wlSize, d_workList, d_goAgain, d_apFound, d_totalPathLengths, sizeOfA, startIndexOfB);
    }

    cudaMemcpyAsync(&goAgain, d_goAgain, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(&wlSize, d_wlSize, sizeof(int), cudaMemcpyDeviceToHost); 
    if (goAgain) {
      nextWlSize = 0;
      cudaMemcpyAsync(d_nextWlSize, &nextWlSize, sizeof(int), cudaMemcpyHostToDevice);
    }
  }
  cudaMemcpy(&apFound, d_apFound, sizeof(bool), cudaMemcpyDeviceToHost);
}


static __global__ void augFromMidToFreevertex(const int* const __restrict__ d_source, const int* const __restrict__ d_parent, const int* const __restrict__ d_endPoint, int* __restrict__ d_mate, const int sizeOfA, int* __restrict__ d_totalNumPaths, int* __restrict__ d_totalPathLengths, const int n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    if (d_source[i] == i) {
      int startVertex = d_endPoint[i];
      if (startVertex != -1) {
        if (d_parent[startVertex] == startVertex) {
          // Nothing to do towards free end. Trivial, as it is the end point.
          if (startVertex < sizeOfA) { // Inc only for one of the 2 halfpaths in an AP
            atomicAdd(d_totalNumPaths, 1);
          }
        } else {
          if (startVertex < sizeOfA) {
            atomicAdd(d_totalPathLengths, 1); // To count the middle edge
            atomicAdd(d_totalNumPaths, 1);
          }
          int v1 = startVertex;
          int v2 = d_parent[v1];
          while (true) {
            d_mate[v1] = v2;
            d_mate[v2] = v1;
            atomicAdd(d_totalPathLengths, 1);
            if (d_parent[v2] == v2) {
              break;
            } else {
              v1 = d_parent[v2];
              v2 = d_parent[v1];
              atomicAdd(d_totalPathLengths, 1);
            }
          }
        }
      }
    }
  }
}


static void augment()
{
  int oldNumPaths = totalNumPaths;
  augFromMidToFreevertex<<<(n + TPB - 1) / TPB, TPB>>>(d_source, d_parent, d_endPoint, d_mate, sizeOfA, d_totalNumPaths, d_totalPathLengths, n);
  cudaMemcpyAsync(&totalNumPaths, d_totalNumPaths, sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  numPathsAugmentedInCurItr = totalNumPaths - oldNumPaths;
}


static int findSizeOfMatching()
{
  int numMatchedEdges = 0;
  for (int a = 0; a < sizeOfA; a++) {
    if (mate[a] != -1)
      numMatchedEdges++;
  }
  return numMatchedEdges;
}


static void freeMemory()
{
  delete[] mate;
  cudaFree(d_nextOption);
  cudaFree(d_itrFlag);
  cudaFree(d_nbrIndx);
  cudaFree(d_edge);
  cudaFree(d_mate);
  cudaFree(d_endPoint);
  cudaFree(d_parent);
  cudaFree(d_source);
  cudaFree(d_workList);
  cudaFree(d_nextWorkList);
  cudaFree(d_wlSize);
  cudaFree(d_nextWlSize);
  cudaFree(d_goAgain);
  cudaFree(d_apFound);
  cudaFree(d_totalPathLengths);
  cudaFree(d_totalNumPaths);
}


static __device__ bool hasHigherPrio(const int a, const int a1, const int deg_a, int* __restrict__ d_nextOption, const int* const __restrict__ d_nbrIndx)
{
  if (a1 == NONE)
    return true;
  else {
    int nextOption_a1 = atomicRead(&d_nextOption[a1]); // atomic read since a1 may write to this location in parallel
    int nextOp_a = d_nextOption[a]; // Only a can write to this location
    int deg_a1 = d_nbrIndx[a1 + 1] - d_nbrIndx[a1];
    return ((deg_a - nextOp_a < deg_a1 - nextOption_a1) || ((deg_a - nextOp_a == deg_a1 - nextOption_a1) && a < a1)); // checking if a has fewer options left compared to a1
  }
}


static __global__ void init_processANodes(const int itr, const int size, bool* __restrict__ d_repeat, int* __restrict__ d_mate, const int* const __restrict__ d_nbrIndx, const int* const __restrict__ d_edge, int* __restrict__ d_nextOption, int* __restrict__ d_itrFlag)
{
  int a = blockDim.x * blockIdx.x + threadIdx.x;
  if (a < size) {
    if (atomicRead(&d_mate[a]) == NONE) {
      for (int i = d_nbrIndx[a] + d_nextOption[a]; i < d_nbrIndx[a + 1]; i++) {
        atomicAdd(&d_nextOption[a], 1);
        int myDeg = d_nbrIndx[a + 1] - d_nbrIndx[a];
        int b = d_edge[i];
        int curMateOfb;
        curMateOfb = atomicRead(&d_mate[b]);
        bool success = false;
        while (hasHigherPrio(a, curMateOfb, myDeg, d_nextOption, d_nbrIndx)) {
          int prevVal = atomicCAS(&(d_mate[b]), curMateOfb, a);
          if (prevVal == curMateOfb) {
            success = true;
            break;
          } else {
            curMateOfb = prevVal;
          }
        }
        if (success) {
          atomicWrite(&d_itrFlag[b], itr);
          if (curMateOfb != NONE) {
            atomicWrite(&d_mate[curMateOfb], NONE);
            atomicWrite(d_repeat, true);
          }
          break;
        }
      }
    }
  }
}


static __global__ void init_processBNodes(const int itr, const int sizeOfB, const int startIndexOfB, const int* const __restrict__ d_itrFlag, int* __restrict__ d_mate)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < sizeOfB) {
    int b = startIndexOfB + id;
    if (d_itrFlag[b] == itr) {
      int a = d_mate[b];
      d_mate[a] = b;
    }
  }
}


static void d_init()
{
  cudaMemset(d_nextOption, 0, sizeOfA * sizeof(int));
  cudaMemset(d_itrFlag, 0, n * sizeof(int));
  int itr = 0;
  bool repeat;
  bool *d_repeat;
  cudaMalloc((void **)&d_repeat, sizeof(bool));
  do {
    repeat = false;
    cudaMemcpy(d_repeat, &repeat, sizeof(bool), cudaMemcpyHostToDevice);
    itr++;
    init_processANodes<<<(sizeOfA + TPB - 1) / TPB, TPB>>>(itr, sizeOfA, d_repeat, d_mate, d_nbrIndx, d_edge, d_nextOption, d_itrFlag);
    init_processBNodes<<<(sizeOfB + TPB - 1) / TPB, TPB>>>(itr, sizeOfB, startIndexOfB, d_itrFlag, d_mate);
    cudaMemcpy(&repeat, d_repeat, sizeof(bool), cudaMemcpyDeviceToHost);
  } while (repeat);
  printf("initIterations = %d\n", itr);
  cudaFree(d_repeat);
}


static void allocateAndInitDS(ECLgraph& g)
{
  mate = new int [n];
  // Arrays:
  cudaMalloc((void **)&d_nextOption, sizeOfA * sizeof(int));
  cudaMalloc((void **)&d_itrFlag, n * sizeof(int));
  cudaMalloc((void **)&d_nbrIndx, (n + 1) * sizeof(int));
  cudaMalloc((void **)&d_edge, numEdges * sizeof(int));
  cudaMalloc((void **)&d_mate, n * sizeof(int));
  cudaMalloc((void **)&d_endPoint, n * sizeof(int));
  cudaMalloc((void **)&d_parent, n * sizeof(int));
  cudaMalloc((void **)&d_source, n * sizeof(int));
  cudaMalloc((void **)&d_workList, n * sizeof(int));
  cudaMalloc((void **)&d_nextWorkList, n * sizeof(int));

  // Variables:
  cudaMalloc((void **)&d_wlSize, sizeof(int));
  cudaMalloc((void **)&d_nextWlSize, sizeof(int));
  cudaMalloc((void **)&d_goAgain, sizeof(bool));
  cudaMalloc((void **)&d_apFound, sizeof(bool));
  cudaMalloc((void **)&d_totalPathLengths, sizeof(int));
  cudaMalloc((void **)&d_totalNumPaths, sizeof(int));
  
  cudaMemcpyAsync(d_totalPathLengths, &totalPathLengths, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_totalNumPaths, &totalNumPaths, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_nbrIndx, g.nindex, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_edge, g.nlist, numEdges * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(d_mate, NONE, n * sizeof(int));
}


static void printDegreeDetails(ECLgraph& g)
{
  int* degree = new int [g.nodes];
  int sum = 0;
  for (int v = 0; v < g.nodes; v++) {
    degree[v] = g.nindex[v + 1] - g.nindex[v];
    sum += degree[v];
  }
  int minDeg = *(std::min_element(degree, (degree + g.nodes)));
  int maxDeg = *(std::max_element(degree, (degree + g.nodes)));
  assert(sum >= 0 && sum < INT_MAX);
  float avgDeg = (float)sum / g.nodes;
  printf("minDeg = %d\nmaxDeg = %d\navgDeg = %.2f\n", minDeg, maxDeg, avgDeg);
  delete[] degree;
}


int main(int argc, char* argv[])
{
  printf("ECL-MM v1.0\n\n");  fflush(stdout);

  if (argc < 2) {
    fprintf(stderr, "USAGE: %s <inputFileName(s)>\nExiting...\n", argv[0]);
    exit(-1);
  }
  printf("threadsPerBlock = %d\n", TPB);
  // Processing one input
  CPUTimer readTimer;
  readTimer.start();
  ECLgraph g = readECLgraph(argv[1]);
  double rt = readTimer.elapsed();
  printf("GraphReadTime = %.2f s\n", rt);

  printf("threshold for enabling warp-centric processing= %d\n", threshold);
  
  n = g.nodes;
  sizeOfA = n / 2;
  sizeOfB = sizeOfA;
  startIndexOfB = sizeOfA;
  numEdges = g.edges;
  printGraphSpec(argv[1]);
  printDegreeDetails(g);
  totalNumPaths = 0;
  totalPathLengths = 0;
  numPathsAugmentedInCurItr = 0;

  // check GPU
  cudaSetDevice(Device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, Device);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: no CUDA capable device detected\n\n"); exit(-1);}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  printf("GPU: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n", deviceProp.name, SMs, mTpSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);
  const float bw = 2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) * 0.000001;
  printf("     %.1f GB/s (%.1f+%.1f) peak bandwidth (%d-bit bus)\n\n", bw, bw / 2, bw / 2, deviceProp.memoryBusWidth);
  CheckCuda(__LINE__);

  allocateAndInitDS(g);

  // Initialization of matching:
  CPUTimer t1;
  t1.start();
  d_init();
  double initTime = t1.elapsed();
  cudaMemcpy(mate, d_mate, n* sizeof(int), cudaMemcpyDeviceToHost);
  printf("initRuntime = %f\n", initTime);
  int m = findSizeOfMatching();
  printf("initialM = %d\n", m);
  printf("initialM_AsPercent = %.2f%%\n", 100.0 * m / sizeOfA);

  // AP Search Phase:
  apSearch_itrCount = 0;
  double apAndAugTime = 0.0;
  CPUTimer ts;
  ts.start();
  do {
    apFound = false;
    cudaMemcpyAsync(d_apFound, &apFound, sizeof(bool), cudaMemcpyHostToDevice);
    apSearch_itrCount++;
    searchForAP();
    if (apFound) {
      augment();
    } else {
      apAndAugTime = ts.elapsed();
      printf("apSearchTime = %f\n", apAndAugTime);
      printf("totalRunTime = %f\n", initTime + apAndAugTime );
      printf("apSearchItr = %d\n", apSearch_itrCount);
    }
  } while (apFound);

  cudaMemcpy(mate, d_mate, n * sizeof(int), cudaMemcpyDeviceToHost);
  int finalM = findSizeOfMatching();
  cudaMemcpy(&totalNumPaths, d_totalNumPaths, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&totalPathLengths, d_totalPathLengths, sizeof(int), cudaMemcpyDeviceToHost);

  printf("finalM = %d\n", finalM);
  printf("finalM_AsPercent = %.2f%%\n", 100.0 * finalM / sizeOfA);
  printf("avgAPLen = %f\n", (float)totalPathLengths / totalNumPaths);
  printf("numNodesDivByTotalRunTime = %lf\n", (double)n / (initTime + apAndAugTime));
  printf("numNodesDivByAPSearchPhaseTime = %lf\n", (double)n / (apAndAugTime));
  printf("numEdgesDivByTotalRunTime = %lf\n", (double)numEdges / (initTime + apAndAugTime));
  printf("numEdgesDivByAPSearchPhaseTime = %lf\n", (double)numEdges / (apAndAugTime));
  printf("-----------------------------\n");
  freeECLgraph(g);
  freeMemory();
}
