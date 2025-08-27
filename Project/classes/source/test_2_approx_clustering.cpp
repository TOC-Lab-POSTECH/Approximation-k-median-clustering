// test_gonzalez.cpp
// Verify Gonzalez 2-approx k-center.
// - small n: compare with optimal brute force
// - large n: benchmark & sanity checks (no brute force)

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <chrono>

#include "../header/2_approx_clustering.h"  // your header

// --- distance utils (squared distance for fast comparisons) ---
static inline double d2(const Point& a, const Point& b) {
    double dx = double(a.x) - double(b.x);
    double dy = double(a.y) - double(b.y);
    return dx*dx + dy*dy;
}

// Compute k-center radius for a fixed set of centers (by indices in P)
static double radius_with_centers(const std::vector<Point>& P,
                                  const std::vector<int>& center_idx) {
    double worst2 = 0.0;
    for (int i = 0; i < (int)P.size(); ++i) {
        double best2 = std::numeric_limits<double>::infinity();
        for (int ci : center_idx) best2 = std::min(best2, d2(P[i], P[ci]));
        worst2 = std::max(worst2, best2);
    }
    return std::sqrt(worst2);
}

// Brute-force optimal k-center among data points (small n only!)
static std::pair<double, std::vector<int>>
optimal_kcenter_bruteforce(const std::vector<Point>& P, int k) {
    const int n = (int)P.size();
    k = std::min(k, n);
    std::vector<int> comb(k);
    for (int i = 0; i < k; ++i) comb[i] = i;

    double bestR = std::numeric_limits<double>::infinity();
    std::vector<int> bestC = comb;

    auto eval = [&](const std::vector<int>& C) {
        double r = radius_with_centers(P, C);
        if (r < bestR) { bestR = r; bestC = C; }
    };

    eval(comb);
    while (true) {
        int i = k - 1;
        while (i >= 0 && comb[i] == n - k + i) --i;
        if (i < 0) break;
        ++comb[i];
        for (int j = i + 1; j < k; ++j) comb[j] = comb[j - 1] + 1;
        eval(comb);
    }
    return {bestR, bestC};
}

// --- pretty printing helpers ---
static void print_centers(const char* tag, const std::vector<Point>& C) {
    std::cout << tag << " centers (" << C.size() << "): ";
    for (size_t i = 0; i < C.size(); ++i) {
        std::cout << "(" << C[i].x << "," << C[i].y << ")";
        if (i + 1 != C.size()) std::cout << ", ";
    }
    std::cout << "\n";
}

// --- small toy datasets ---
static std::vector<Point> make_square_corners() {
    return { {0,0}, {10,0}, {0,10}, {10,10}, {5,5} };
}
static std::vector<Point> make_two_clusters() {
    std::vector<Point> P;
    for (int i = 0; i < 5; ++i) P.push_back({i, 0});
    for (int i = 0; i < 5; ++i) P.push_back({100 + i, 0});
    return P;
}
static std::vector<Point> make_small_random(int n = 10, int seed = 7) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> uni(-50, 50);
    std::vector<Point> P; P.reserve(n);
    for (int i = 0; i < n; ++i) P.push_back({uni(rng), uni(rng)});
    return P;
}

// --- large datasets (for benchmarking) ---
static std::vector<Point> make_large_random(int n, int seed = 123) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> uni(-100000, 100000);
    std::vector<Point> P; P.reserve(n);
    for (int i = 0; i < n; ++i) P.push_back({uni(rng), uni(rng)});
    return P;
}

// optional: clustered large dataset
static std::vector<Point> make_large_clusters(int n, int clusters = 20, int seed = 456) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> ctr(-200000, 200000);
    std::normal_distribution<double>   noise(0.0, 2000.0);
    std::vector<Point> centers;
    for (int c = 0; c < clusters; ++c) centers.push_back({ctr(rng), ctr(rng)});

    std::vector<Point> P; P.reserve(n);
    for (int i = 0; i < n; ++i) {
        const auto& c = centers[i % clusters];
        int x = int(std::lround(double(c.x) + noise(rng)));
        int y = int(std::lround(double(c.y) + noise(rng)));
        P.push_back({x, y});
    }
    return P;
}

// --- small-n case: exact optimal vs greedy ---
static void run_case_small(const std::vector<Point>& P, int k, const char* name) {
    std::cout << "=== CASE: " << name << " (n=" << P.size() << ", k=" << k << ") ===\n";

    KCenterResult G = gonzalez_k_center(P, k);
    print_centers("Greedy", G.centers);
    std::cout << "Greedy radius L_g = " << G.L
              << " ; farthest = (" << G.farthest_point.x << "," << G.farthest_point.y
              << "), idx=" << G.farthest_index << "\n";

    auto [Lopt, Copt_idx] = optimal_kcenter_bruteforce(P, k);
    std::vector<Point> Copt; Copt.reserve(Copt_idx.size());
    for (int idx : Copt_idx) Copt.push_back(P[idx]);
    print_centers("Optimal", Copt);
    std::cout << "Optimal radius L* = " << Lopt << "\n";

    double ratio = (Lopt > 0.0) ? (G.L / Lopt) : 1.0;
    std::cout << "Ratio L_g / L* = " << ratio
              << (ratio <= 2.0000001 ? "  (OK ≤ 2)" : "  (VIOLATION)") << "\n\n";
}

// --- large-n case: benchmark & sanity checks ---
// --- large-n case: benchmark & repeated sample checks ---
static void run_case_large(const std::vector<Point>& P,
                           const std::vector<int>& ks,
                           const char* name,
                           int sample_size = 20,
                           int repeat = 5,
                           int sample_seed = 999) {
    using namespace std::chrono;
    std::cout << "=== LARGE CASE: " << name << " (n=" << P.size() << ") ===\n";

    double prevL = std::numeric_limits<double>::infinity();
    int total_k = (int)ks.size();
    int step = 0;

    std::mt19937 rng(sample_seed);

    for (int k : ks) {
        step++;
        std::cout << "[INFO] Processing k=" << k
                  << " (" << step << "/" << total_k << ")\n";

        auto t0 = high_resolution_clock::now();
        KCenterResult G = gonzalez_k_center(P, k);
        auto t1 = high_resolution_clock::now();
        double ms = duration<double, std::milli>(t1 - t0).count();

        std::cout << "k=" << k
                  << " | centers=" << G.centers.size()
                  << " | L=" << G.L
                  << " | time=" << ms << " ms\n";

        // Sanity 1: radius must be non-increasing as k grows
        if (prevL < std::numeric_limits<double>::infinity()) {
            if (G.L > prevL + 1e-9) {
                std::cout << "  [WARN] L increased when k grew (prev=" << prevL << ").\n";
            }
        }
        prevL = G.L;

        // Sanity 2: multiple samples for lower-bound estimate
        if (sample_size > 0 && sample_size < (int)P.size()) {
            std::uniform_int_distribution<int> pick(0, (int)P.size() - 1);

            for (int rep = 1; rep <= repeat; ++rep) {
                std::vector<Point> S;
                S.reserve(sample_size);
                for (int i = 0; i < sample_size; ++i) {
                    S.push_back(P[pick(rng)]);
                }

                auto [LoptS, _] = optimal_kcenter_bruteforce(S, k);

                double worst2 = 0.0;
                for (int i = 0; i < (int)S.size(); ++i) {
                    double best2 = std::numeric_limits<double>::infinity();
                    for (const auto& c : G.centers) {
                        best2 = std::min(best2, d2(S[i], c));
                    }
                    worst2 = std::max(worst2, best2);
                }
                double Lg_on_S = std::sqrt(worst2);

                std::cout << "  [sample rep " << rep << "]"
                          << " L*_S=" << LoptS
                          << " | greedy-on-S=" << Lg_on_S
                          << " | ratio=" << (LoptS > 0 ? (Lg_on_S / LoptS) : std::numeric_limits<double>::infinity())
                          << "\n";
            }
        }

        std::cout << "[INFO] Finished k=" << k << "\n\n";
    }
    std::cout << "=== Completed LARGE CASE: " << name << " ===\n\n";
}



int main() {
    const int N = 50000;                  // 50k points
    std::vector<int> KSET = {4, 8, 16};   // k 값들 테스트

    auto P_big = make_large_random(N, 2025);

    // 샘플 20개씩, 5회 반복 (원하는 만큼 repeat 늘릴 수 있음)
    run_case_large(P_big, KSET, "uniform_50k_repeated", 23, 5, 123);

    return 0;
}