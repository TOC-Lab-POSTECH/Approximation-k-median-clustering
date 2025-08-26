//Gonzalez '85

#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "point.h"

// Result structure for Gonzalez k-center algorithm
struct KCenterResult {
    std::vector<Point> centers;     // chosen k centers
    std::vector<int> assign;        // assignment of each point to a center
    std::vector<double> min_dists;  // nearest distance of each point to a center
    double L = 0.0;                 // final radius

    // additions for V = centers ∪ {farthest point from centers}
    Point farthest_point;           // the farthest point achieving L
    int   farthest_index;      // its index in the original P
};

// input : 2 points
// output : return squared distance
inline double dist2(const Point& a, const Point& b) {
    const double dx = double(a.x) - double(b.x);
    const double dy = double(a.y) - double(b.y);
    return dx*dx + dy*dy;
}

// input : 2 points
// output : return the euclidean distance
inline double dist(const Point& a, const Point& b) {
    return std::sqrt(dist2(a, b));
}

int select_farthest_index(const std::vector<double>& min_dists) {
    int argmax = -1;
    double best = -1.0;
    for (int i = 0; i < (int)min_dists.size(); ++i) {
        if (min_dists[i] > best) {
            best = min_dists[i];
            argmax = i;
        }
    }
    return argmax;
}

double compute_radius(const std::vector<double>& min_dists) {
    double L = 0.0;
    for (double d : min_dists) L = std::max(L, d);
    return L;
}

void init_assignments(const std::vector<Point>& P,
    const std::vector<Point>& centers,
    std::vector<int>& assign,
    std::vector<double>& min_dists) {
//n : number of point
const int n = (int)P.size();

assign.assign(n, -1);
min_dists.assign(n, std::numeric_limits<double>::infinity());
for (int i = 0; i < n; ++i) {
double d = dist(P[i], centers[0]); 
min_dists[i] = d;
assign[i]    = 0;
}
}

void update_assignments_with_new_center(const std::vector<Point>& P,
    const Point& new_center,
    int new_center_id,
    std::vector<int>& assign,
    std::vector<double>& min_dists) {
const int n = (int)P.size();

for (int i = 0; i < n; ++i) {
double d = dist(P[i], new_center);
if (d < min_dists[i]) {
    min_dists[i] = d;
    assign[i]    = new_center_id;
    }
}
}

// Gonzalez 2-approximation k-center clustering
KCenterResult gonzalez_k_center(const std::vector<Point>& P, int k, int start_idx = 0) {
    KCenterResult R;
    const int n = (int)P.size();
    if (n == 0 || k <= 0) {
        R.L = 0.0;
        return R;
    }
    k = std::min(k, n);
    if (start_idx < 0 || start_idx >= n) start_idx = 0;
    
    //optimization
    R.centers.reserve(k);
    
    R.assign.assign(n, -1);
    R.min_dists.assign(n, std::numeric_limits<double>::infinity());

    // 1) choose first center
    R.centers.push_back(P[start_idx]);
    init_assignments(P, R.centers, R.assign, R.min_dists);

    // 2) repeat: add farthest point as new center and update
    while ((int)R.centers.size() < k) {
        int argmax = select_farthest_index(R.min_dists);
        R.centers.push_back(P[argmax]);
        int new_cid = (int)R.centers.size() - 1;
        update_assignments_with_new_center(P, R.centers.back(), new_cid, R.assign, R.min_dists);
    }

    // 3) compute final radius
    R.L = compute_radius(R.min_dists);

    // 4) find the farthest point that achieves L
    //    (ties: pick the first)
    int far_idx = 0;
    double best = -1.0;
    for (int i = 0; i < n; ++i) {
        if (R.min_dists[i] > best) {
            best = R.min_dists[i];
            far_idx = i;
        }
    }
    R.farthest_index = far_idx;
    R.farthest_point = P[far_idx];

    return R;
}