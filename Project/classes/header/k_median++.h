#ifndef K_MEDIAN_PP_H
#define K_MEDIAN_PP_H

#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include <algorithm>
#include "point.h"  // struct Point { int x, y; double w; ... }

using namespace std;

// ---------------------------
// Utilities: distance and cost
// ---------------------------

// Euclidean distance (L2)
double euclidean_distance(const Point& a, const Point& b);

// Weighted k-median cost: sum_{p} w(p) * dist(p, C)
double k_median_cost(const vector<Point>& pts,
                     const vector<Point>& centers);

// -------------------------------------------
// 1-median (geometric median) approximation: Weiszfeld
// -------------------------------------------
// Approximate the weighted geometric median of points in one cluster.
//   - input: clusterPoints = points assigned to this cluster
//   - initial position: (x0, y0), e.g., current center or weighted mean
//   - output: approximate geometric median as a Point (weight field unused)
Point geometric_median_weiszfeld(const vector<Point>& clusterPoints,
                                 double x0, double y0,
                                 int maxIter = 100,
                                 double tol = 1e-3);

// -------------------------------------------
// k-median++ initialization (on R_j)
// -------------------------------------------
// Input: pts = R_j (weighted representative points)
// Output: k initial centers (vector<Point>, weight can be set to 1.0)
vector<Point> k_median_pp_init(const vector<Point>& pts,
                               int k,
                               mt19937& rng);

// -------------------------------------------
// Full k-median++ algorithm (on R_j)
// -------------------------------------------
// Input:
//   - pts: R_j (weighted representative points)
//   - k:   number of centers
//   - maxIters: number of Lloyd-style iterations
// Output:
//   - final centers (vector<Point>, weight is dummy 1.0)
//   - cost can be computed with k_median_cost(pts, centers)
vector<Point> k_median_pp(const vector<Point>& pts,
                          int k,
                          int maxIters = 100,
                          int maxWeiszfeldIters = 50,
                          double tol = 1e-3,
                          uint32_t seed = 12345);

#endif // K_MEDIAN_PP_H
