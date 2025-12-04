#include "k_median++.h"

// ---------------------------
// Utilities: distance and cost
// ---------------------------

// Euclidean distance (L2)
double euclidean_distance(const Point& a, const Point& b) {
    double dx = (double)a.x - (double)b.x;
    double dy = (double)a.y - (double)b.y;
    return sqrt(dx * dx + dy * dy);
}

// Weighted k-median cost: sum_{p} w(p) * dist(p, C)
double k_median_cost(const vector<Point>& pts,
                     const vector<Point>& centers) {
    if (centers.empty()) return numeric_limits<double>::infinity();
    double cost = 0.0;
    for (const auto& p : pts) {
        double best = numeric_limits<double>::infinity();
        for (const auto& c : centers) {
            double d = euclidean_distance(p, c);
            if (d < best) best = d;
        }
        cost += p.w * best;
    }
    return cost;
}

// -------------------------------------------
// 1-median (geometric median) approximation: Weiszfeld
// -------------------------------------------
// Approximate the weighted geometric median of points in one cluster.
Point geometric_median_weiszfeld(const vector<Point>& clusterPoints,
                                 double x0, double y0,
                                 int maxIter,
                                 double tol) {
    double x = x0;
    double y = y0;

    for (int it = 0; it < maxIter; ++it) {
        double numX = 0.0, numY = 0.0;
        double denom = 0.0;
        bool allVeryClose = true;

        for (const auto& p : clusterPoints) {
            double dx = x - (double)p.x;
            double dy = y - (double)p.y;
            double dist = sqrt(dx * dx + dy * dy);

            // If a point is extremely close to the current center,
            // snap to that point and stop.
            if (dist < 1e-9) {
                x = (double)p.x;
                y = (double)p.y;
                return Point((int)round(x), (int)round(y), 1.0);
            }

            allVeryClose = false;
            double w = p.w;
            numX += w * ((double)p.x / dist);
            numY += w * ((double)p.y / dist);
            denom += w / dist;
        }

        if (allVeryClose || denom == 0.0) break;

        double newX = numX / denom;
        double newY = numY / denom;

        double move = sqrt((x - newX) * (x - newX) + (y - newY) * (y - newY));
        x = newX;
        y = newY;

        if (move < tol) break;
    }

    return Point((int)round(x), (int)round(y), 1.0);
}

// -------------------------------------------
// k-median++ initialization (on R_j)
// -------------------------------------------
// Input: pts = R_j (weighted representative points)
vector<Point> k_median_pp_init(const vector<Point>& pts,
                               int k,
                               mt19937& rng) {
    vector<Point> centers;
    int n = (int)pts.size();
    if (n == 0 || k <= 0) return centers;

    // Cap k by the number of points
    k = min(k, n);

    // 1. First center: sample randomly proportional to weight w(p)
    vector<double> prefix(n);
    double totalW = 0.0;
    for (int i = 0; i < n; ++i) {
        totalW += pts[i].w;
        prefix[i] = totalW;
    }
    uniform_real_distribution<double> urand(0.0, totalW);
    double r = urand(rng);
    int firstIdx = (int)(lower_bound(prefix.begin(), prefix.end(), r) - prefix.begin());
    centers.push_back(pts[firstIdx]);

    // 2. Remaining centers: sample proportional to D(x) (k-median style)
    while ((int)centers.size() < k) {
        vector<double> distWeights(n);
        double sum = 0.0;

        for (int i = 0; i < n; ++i) {
            const auto& p = pts[i];
            // Distance from p to its nearest existing center
            double best = numeric_limits<double>::infinity();
            for (const auto& c : centers) {
                double d = euclidean_distance(p, c);
                if (d < best) best = d;
            }
            // Sampling weight ~ w(p) * D(p)
            double val = p.w * best;
            distWeights[i] = val;
            sum += val;
        }

        if (sum <= 0.0) {
            // All points identical: fill remaining centers with random points (duplicates allowed)
            while ((int)centers.size() < k) {
                uniform_int_distribution<int> ui(0, n - 1);
                centers.push_back(pts[ui(rng)]);
            }
            break;
        }

        // Sample one index from cumulative distribution
        double r2 = uniform_real_distribution<double>(0.0, sum)(rng);
        double acc = 0.0;
        int idx = n - 1;
        for (int i = 0; i < n; ++i) {
            acc += distWeights[i];
            if (acc >= r2) {
                idx = i;
                break;
            }
        }
        centers.push_back(pts[idx]);
    }

    return centers;
}

// -------------------------------------------
// Full k-median++ algorithm (on R_j)
// -------------------------------------------
vector<Point> k_median_pp(const vector<Point>& pts,
                          int k,
                          int maxIters,
                          int maxWeiszfeldIters,
                          double tol,
                          uint32_t seed) {
    vector<Point> centers;
    int n = (int)pts.size();
    if (n == 0 || k <= 0) return centers;

    mt19937 rng(seed);

    // 1. k-median++ initialization
    centers = k_median_pp_init(pts, k, rng);
    int csz = (int)centers.size();
    if (csz == 0) return centers;

    // 2. Lloyd-style iterations: assignment → 1-median update
    vector<int> assignment(n, -1);

    for (int it = 0; it < maxIters; ++it) {
        // 2-1. Assign each point to its nearest center
        bool changed = false;
        for (int i = 0; i < n; ++i) {
            const auto& p = pts[i];
            int bestIdx = -1;
            double bestDist = numeric_limits<double>::infinity();

            for (int c = 0; c < csz; ++c) {
                double d = euclidean_distance(p, centers[c]);
                if (d < bestDist) {
                    bestDist = d;
                    bestIdx = c;
                }
            }

            if (assignment[i] != bestIdx) {
                assignment[i] = bestIdx;
                changed = true;
            }
        }

        // Early stop if assignments do not change
        if (!changed && it > 0) break;

        // 2-2. For each cluster, update center via weighted geometric median
        vector<vector<Point>> clusters(csz);
        for (int i = 0; i < n; ++i) {
            int cid = assignment[i];
            if (cid >= 0 && cid < csz) {
                clusters[cid].push_back(pts[i]);
            }
        }

        bool centersMovedLittle = true;

        for (int c = 0; c < csz; ++c) {
            auto& clusterPoints = clusters[c];

            if (clusterPoints.empty()) {
                // Empty cluster: reinitialize center from a random point to avoid collapse
                uniform_int_distribution<int> ui(0, n - 1);
                centers[c] = pts[ui(rng)];
                centersMovedLittle = false;
                continue;
            }

            // Initial position: weighted mean of the cluster
            double sumW = 0.0, sumX = 0.0, sumY = 0.0;
            for (const auto& p : clusterPoints) {
                sumW += p.w;
                sumX += p.w * (double)p.x;
                sumY += p.w * (double)p.y;
            }
            double initX = sumX / max(sumW, 1e-9);
            double initY = sumY / max(sumW, 1e-9);

            Point newCenter = geometric_median_weiszfeld(clusterPoints,
                                                         initX, initY,
                                                         maxWeiszfeldIters,
                                                         tol);

            double move = euclidean_distance(centers[c], newCenter);
            if (move > tol) centersMovedLittle = false;

            // Weight of center is not used; keep it as 1.0
            centers[c] = newCenter;
        }

        if (centersMovedLittle) break;
    }

    return centers;
}
