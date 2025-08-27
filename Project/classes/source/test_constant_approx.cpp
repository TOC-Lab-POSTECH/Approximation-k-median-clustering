// File: ../source/test_constant_approx.cpp
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <chrono>
#include <limits>
#include <algorithm>
#include <unordered_set>
#include <iomanip>  // for fixed / setprecision

#include "../header/point.h"
#include "../header/constant_approx.h"

// =======================================================
// Data generation helpers
// =======================================================

// Generate a blob with uniform weight
static void make_blob(std::vector<Point>& P,
                      int cx, int cy,
                      int count,
                      int half_span,
                      double w_each,
                      unsigned seed)
{
    // Uniform integer noise in [-half_span, half_span]
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> U(-half_span, half_span);

    P.reserve(P.size() + (size_t)count);
    for (int i = 0; i < count; ++i) {
        int x = cx + U(rng);
        int y = cy + U(rng);
        P.emplace_back(x, y, w_each);
    }
}

// Generate a blob with random weights in [w_min, w_max]
static void make_blob_weighted_random(std::vector<Point>& P,
                                      int cx, int cy,
                                      int count,
                                      int half_span,
                                      double w_min,
                                      double w_max,
                                      unsigned seed_pos,
                                      unsigned seed_w)
{
    std::mt19937 rng_pos(seed_pos);
    std::uniform_int_distribution<int> Upos(-half_span, half_span);

    std::mt19937 rng_w(seed_w);
    std::uniform_real_distribution<double> Uw(std::min(w_min, w_max),
                                              std::max(w_min, w_max));

    P.reserve(P.size() + (size_t)count);
    for (int i = 0; i < count; ++i) {
        int x = cx + Upos(rng_pos);
        int y = cy + Upos(rng_pos);
        double w = std::max(0.0, Uw(rng_w));
        P.emplace_back(x, y, w);
    }
}

// Compute total weight
static double sum_weight(const std::vector<Point>& P) {
    double W = 0.0;
    for (const auto& p : P) W += (p.w > 0.0 ? p.w : 0.0);
    return W;
}

// =======================================================
// Distance + cost
// =======================================================

// Euclidean distance
static inline double dist_euclid(const Point& a, const Point& b) {
    const double dx = double(a.x) - double(b.x);
    const double dy = double(a.y) - double(b.y);
    return std::sqrt(dx*dx + dy*dy);
}

// k-median objective (sum of weighted L2 distances to nearest center)
static double kmedian_cost_L2(const std::vector<Point>& P,
                              const std::vector<Point>& C)
{
    if (C.empty()) return 0.0;
    double cost = 0.0;
    for (const auto& p : P) {
        double best = std::numeric_limits<double>::infinity();
        for (const auto& c : C) {
            double d = dist_euclid(p, c);
            if (d < best) best = d;
        }
        cost += p.w * best;
    }
    return cost;
}

// =======================================================
// Greedy k-medoids baseline (BUILD-like)
// =======================================================

static std::vector<Point> select_k_medoids_greedy_from_candidates(
    const std::vector<Point>& P,
    const std::vector<Point>& Cand,
    int k)
{
    if (k <= 0 || Cand.empty()) return {};
    int kk = std::min<int>(k, (int)Cand.size());

    std::vector<double> best(P.size(), std::numeric_limits<double>::infinity());
    std::vector<Point> chosen;
    chosen.reserve(kk);

    for (int it = 0; it < kk; ++it) {
        int best_idx = -1;
        double best_score = std::numeric_limits<double>::infinity();

        for (int s = 0; s < (int)Cand.size(); ++s) {
            const Point& c = Cand[s];
            double sum_cost = 0.0;

            if (chosen.empty()) {
                for (size_t i = 0; i < P.size(); ++i) {
                    double d = dist_euclid(P[i], c);
                    sum_cost += P[i].w * d;
                }
            } else {
                for (size_t i = 0; i < P.size(); ++i) {
                    double d = dist_euclid(P[i], c);
                    double nb = std::min(best[i], d);
                    sum_cost += P[i].w * nb;
                }
            }

            if (sum_cost < best_score) {
                best_score = sum_cost;
                best_idx = s;
            }
        }

        const Point& winner = Cand[best_idx];
        chosen.push_back(winner);

        for (size_t i = 0; i < P.size(); ++i) {
            double d = dist_euclid(P[i], winner);
            if (d < best[i]) best[i] = d;
        }
    }

    return chosen;
}

// =======================================================
// Candidate subsampling
// =======================================================

static std::vector<int> sample_indices(int n, int m, unsigned seed) {
    m = std::min(m, n);
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;

    std::mt19937 rng(seed);
    for (int i = 0; i < m; ++i) {
        std::uniform_int_distribution<int> U(i, n - 1);
        int j = U(rng);
        std::swap(idx[i], idx[j]);
    }
    idx.resize(m);
    return idx;
}

static std::vector<Point> subsample_candidates_from_P(const std::vector<Point>& P,
                                                      int m, unsigned seed)
{
    std::vector<Point> S; S.reserve(std::min(m, (int)P.size()));
    auto idx = sample_indices((int)P.size(), m, seed);
    for (int i : idx) S.push_back(P[i]);
    return S;
}

// =======================================================
// Tests
// =======================================================

static void test_large_n_weighted() {
    std::cout << "=== Test: large-n (>= 10000), weighted data ===\n";

    const int k = 10;
    const int blobs = k;
    const int per_blob = 5000;
    const int half_span = 10;

    std::vector<Point> P; P.reserve(blobs * per_blob);

    int base_x = 0;
    for (int b = 0; b < blobs; ++b) {
        if (b < 3) {
            make_blob_weighted_random(P, base_x, 0, per_blob, half_span,
                                      0.5, 1.5, 100+b, 200+b);
        } else if (b < 7) {
            make_blob_weighted_random(P, base_x, 0, per_blob, half_span,
                                      1.0, 3.0, 100+b, 200+b);
        } else {
            make_blob_weighted_random(P, base_x, 0, per_blob, half_span,
                                      3.0, 10.0, 100+b, 200+b);
        }
        base_x += 200;
    }
    // Add heavy weighted outliers
    P.emplace_back(base_x+1000, 800,  500.0);
    P.emplace_back(base_x+1200, -600, 800.0);

    const double Wtot = sum_weight(P);
    const double logW = std::log2(std::max(2.0, Wtot));

    std::cout << "Input: |P|=" << P.size()
              << ", W_total=" << Wtot
              << ", log2(W)=" << logW << "\n";

    const double gamma = 2.0;
    const unsigned seed = 12345;

    auto t0 = std::chrono::high_resolution_clock::now();
    ConstantCentersResult CF = build_constant_factor_centers(P, k, gamma, seed);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cf = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "CF Rounds=" << CF.rounds
              << ", |X_all|=" << CF.X_all.size()
              << ", W_picked=" << CF.W_picked << " / " << CF.W_total
              << "  (" << ms_cf << " ms)\n";

    assert(CF.W_total > 0.0);
    double remain = CF.W_total - CF.W_picked;
    std::cout << "Remaining weight = " << remain << "\n";
    assert(remain <= 1e-8 * CF.W_total);

    int bound_rounds = (int)std::ceil(12.0 * logW);
    std::cout << "Round bound (12 * log2 W) = " << bound_rounds << "\n";
    assert(CF.rounds <= bound_rounds);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::vector<Point> C_from_X = select_k_medoids_greedy_from_candidates(P, CF.X_all, k);
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_selX = std::chrono::duration<double, std::milli>(t3 - t2).count();

    double cost_from_X = kmedian_cost_L2(P, C_from_X);
    std::cout << "Cost(X_all->k medoids) = " << cost_from_X
              << "  (" << ms_selX << " ms)\n";

    const int m_cand = 2000;
    auto Cand_base = subsample_candidates_from_P(P, m_cand, 777);

    auto t4 = std::chrono::high_resolution_clock::now();
    std::vector<Point> C_base = select_k_medoids_greedy_from_candidates(P, Cand_base, k);
    auto t5 = std::chrono::high_resolution_clock::now();
    double ms_selB = std::chrono::duration<double, std::milli>(t5 - t4).count();

    double cost_base = kmedian_cost_L2(P, C_base);
    double ratio = cost_from_X / std::max(1e-12, cost_base);

    std::cout << std::fixed << std::setprecision(6)
              << "Cost(baseline from P-subset m=" << m_cand << ") = " << cost_base
              << "  (" << ms_selB << " ms)\n"
              << "Empirical constant factor = " << ratio << "x\n";

    std::cout << "[OK] large-n weighted test completed.\n";
}

// Evaluate k-median cost on a subset Q (to keep runtime manageable)
static double kmedian_cost_L2_on_subset(const std::vector<Point>& Q,
                                        const std::vector<Point>& C)
{
    if (C.empty()) return 0.0;
    double cost = 0.0;
    for (const auto& p : Q) {
        double best = std::numeric_limits<double>::infinity();
        for (const auto& c : C) {
            double dx = double(p.x) - double(c.x);
            double dy = double(p.y) - double(c.y);
            double d  = std::sqrt(dx*dx + dy*dy);
            if (d < best) best = d;
        }
        cost += p.w * best;
    }
    return cost;
}

static void test_very_large_n_weighted() {
    std::cout << "=== Test: very-large-n (~100k), weighted data ===\n";

    // Build k well-separated blobs; n ~ 100000
    const int k = 20;
    const int blobs = k;
    const int per_blob = 5000;     // 5000 * 20 = 100000
    const int half_span = 12;

    std::vector<Point> P; P.reserve((size_t)blobs * per_blob);

    int base_x = 0;
    for (int b = 0; b < blobs; ++b) {
        // Vary weight ranges per blob
        if (b < 6) {
            make_blob_weighted_random(P, base_x, 0, per_blob, half_span,
                                      0.5, 2.0, 100+b, 200+b);
        } else if (b < 14) {
            make_blob_weighted_random(P, base_x, 0, per_blob, half_span,
                                      1.0, 4.0, 100+b, 200+b);
        } else {
            make_blob_weighted_random(P, base_x, 0, per_blob, half_span,
                                      3.0, 12.0, 100+b, 200+b);
        }
        base_x += 220;
    }
    // A couple of very heavy far outliers
    P.emplace_back(base_x + 1500, 900, 1000.0);
    P.emplace_back(base_x + 1800, -700, 1500.0);

    const double Wtot = sum_weight(P);
    const double logW = std::log2(std::max(2.0, Wtot));

    std::cout << "Input: |P|=" << P.size()
              << ", W_total=" << Wtot
              << ", log2(W)=" << logW << "\n";

    // Use a smaller gamma to promote multiple rounds (smaller Y each round)
    const double gamma = 0.5;     // try 0.25~1.0 as you like
    const unsigned seed = 20250827;

    auto t0 = std::chrono::high_resolution_clock::now();
    ConstantCentersResult CF = build_constant_factor_centers(P, k, gamma, seed);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cf = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "CF Rounds=" << CF.rounds
              << ", |X_all|=" << CF.X_all.size()
              << ", W_picked=" << CF.W_picked << " / " << CF.W_total
              << "  (" << ms_cf << " ms)\n";

    // Coverage check
    assert(CF.W_total > 0.0);
    double remain = CF.W_total - CF.W_picked;
    std::cout << "Remaining weight = " << remain << "\n";
    assert(remain <= 1e-8 * CF.W_total);

    // Round bound O(log W)
    int bound_rounds = (int)std::ceil(12.0 * logW);
    std::cout << "Round bound (12 * log2 W) = " << bound_rounds << "\n";
    assert(CF.rounds <= bound_rounds);

    // ---- Evaluation on a subset to keep runtime reasonable ----
    const int eval_m = 10000;  // evaluation subset size
    auto eval_idx = sample_indices((int)P.size(), eval_m, 424242);
    std::vector<Point> P_eval; P_eval.reserve(eval_m);
    for (int i : eval_idx) P_eval.push_back(P[i]);

    // Select k centers from X_all via greedy k-medoids
    auto t2 = std::chrono::high_resolution_clock::now();
    std::vector<Point> C_from_X = select_k_medoids_greedy_from_candidates(P_eval, CF.X_all, k);
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_selX = std::chrono::duration<double, std::milli>(t3 - t2).count();

    double cost_from_X = kmedian_cost_L2_on_subset(P_eval, C_from_X);
    std::cout << "Cost(X_all->k medoids) on eval(" << eval_m << ") = "
              << cost_from_X << "  (" << ms_selX << " ms)\n";

    // Baseline: random candidate subset from P_eval (same eval set for fairness)
    const int m_cand = 4000; // you can try 3k~6k
    auto Cand_base = subsample_candidates_from_P(P_eval, m_cand, 737373);

    auto t4 = std::chrono::high_resolution_clock::now();
    std::vector<Point> C_base = select_k_medoids_greedy_from_candidates(P_eval, Cand_base, k);
    auto t5 = std::chrono::high_resolution_clock::now();
    double ms_selB = std::chrono::duration<double, std::milli>(t5 - t4).count();

    double cost_base = kmedian_cost_L2_on_subset(P_eval, C_base);
    double ratio = cost_from_X / std::max(1e-12, cost_base);

    std::cout << std::fixed << std::setprecision(6)
              << "Cost(baseline from eval-subset m=" << m_cand << ") = " << cost_base
              << "  (" << ms_selB << " ms)\n"
              << "Empirical constant factor (on eval) = " << ratio << "x\n";

    std::cout << "[OK] very-large-n weighted test completed.\n";
}


int main() {
    try {
        test_very_large_n_weighted();   // NEW
        // test_large_n_weighted();     // 필요하면 기존 것도 함께
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception\n";
        return 1;
    }
    std::cout << "All tests passed.\n";
    return 0;
}
