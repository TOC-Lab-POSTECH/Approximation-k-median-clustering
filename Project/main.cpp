#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>  
#include <fstream>
#include <iomanip>
#include "point.h"
#include "RangeCountingOracle.h"
#include "Representative.h"
#include "k_median++.h"

using namespace std;
using namespace std::chrono;

static inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

static inline double ms_to_sec(double ms) { return ms / 1000.0; }

static inline long double complete_edges(long long n) {
    // m = n(n-1)/2
    return (long double)n * (long double)(n - 1) / 2.0L;
}

static inline long double edges_per_0p1sec(long long n, double timeMs) {
    // Compute edges per 0.1 second for complete graph
    // edges_per_0.1s = 0.1 * m / T, where m=n(n-1)/2, T is in seconds
    long double m = complete_edges(n);
    long double tSec = (long double)timeMs / 1000.0L;
    if (tSec <= 0) return 0.0L;
    return 0.1L * m / tSec;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    ofstream csv("results.csv");
csv << "N,eps,k,baseline_cost,baseline_time_ms,first_success_j,aborted_runs,Rj_size,"
       "rco_cost_on_P,rco_cost_on_Rj,rco_build_ms,rco_km_ms,rco_total_ms,eps_eff\n";

csv << fixed << setprecision(6);


    // -------------------------------
    // 0. Parameter settings
    // -------------------------------
    // List of N values to benchmark
     vector<int> N_list = {100000, 200000, 300000, 400000, 500000};
//     const int k = 200;           // number of clusters
    vector<int> K_list = {10, 20, 30, 40, 50};
    vector<double> EPS_list = {0.10, 0.20, 0.30};
//     const double eps_rep = 0.2; // ε used to construct R_j (dense/sparse threshold)
//     const double eps_guess = 0.2;// ε used for OPT guesses r_j = (1+ε)^j
    const int maxKMiter = 20;    // max Lloyd iterations for k-median++
    const int baselineRuns = 3;  // #runs on P, take the best cost
    const int runsPerJ = 1;      // #runs of k-median++ per R_j

    mt19937 rng(123);            // global RNG for this experiment

for (int N : N_list) {
    rng.seed(123 + N); // keep runs reproducible but different per N
    cout << "\n\n==============================\n";
    cout << "Benchmark for N = " << N << "\n";
    cout << "==============================\n";
    // -------------------------------
    // 1. Generate original point set P in [0, 2N) × [0, 2N)
    // -------------------------------
    vector<Point> P;
    P.reserve(N);
    uniform_int_distribution<int> coordDist(0, 2 * N - 1);

    for (int i = 0; i < N; ++i) {
        int x = coordDist(rng);
        int y = coordDist(rng);
        P.emplace_back(x, y, 1.0); // weight = 1 for each point
    }

    cout << "Generated " << P.size() << " points in [0, " << 2 * N - 1 << "]^2.\n";

    // -------------------------------
    // 2. Build RangeTree (Range Counting Oracle)
    // -------------------------------
    auto t0 = steady_clock::now();
    RangeTree tree(P);
    auto t1 = steady_clock::now();
    double buildTreeMs = duration<double, milli>(t1 - t0).count();

    cout << "RangeTree built in " << buildTreeMs << " ms.\n";
    cout << "Bounding box of P: x in [" << tree.getMinX() << ", " << tree.getMaxX()
         << "], y in [" << tree.getMinY() << ", " << tree.getMaxY() << "]\n\n";


for (int k : K_list) {
    cout << "\n==============================\n";
    cout << "Now running for k = " << k << "\n";
    cout << "==============================\n";

    // -------------------------------
    // Baseline: run k-median++ directly on P (once per k)
    // -------------------------------
    cout << "=== Baseline: k-median++ on P ===\n";

    double baseline_cost = numeric_limits<double>::infinity();
    vector<Point> baseline_centers;
    double baseline_best_run_time_ms = 0.0;
    double baseline_total_time_ms = 0.0;

    auto base_t0 = steady_clock::now();

    for (int run = 0; run < baselineRuns; ++run) {
        // Baseline is independent of eps, so do NOT include ei here.
        uint32_t seed = 1000u + (uint32_t)k * 1000u + (uint32_t)run;

        auto tb0 = steady_clock::now();
        vector<Point> centersP = k_median_pp(P, k, maxKMiter, 50, 1e-3, seed);
        auto tb1 = steady_clock::now();

        double runTimeMs = duration<double, milli>(tb1 - tb0).count();
        double costP = k_median_cost(P, centersP);

        cout << "[P run " << run << "] time = " << runTimeMs
             << " ms, cost(P) = " << costP << "\n";

        if (costP < baseline_cost) {
            baseline_cost = costP;
            baseline_centers = centersP;
            baseline_best_run_time_ms = runTimeMs;
        }
    }

    auto base_t1 = steady_clock::now();
    baseline_total_time_ms = duration<double, milli>(base_t1 - base_t0).count();

    cout << "Best baseline on P: cost = " << baseline_cost
         << ", best-run time = " << baseline_best_run_time_ms
         << " ms, total baseline time = " << baseline_total_time_ms
         << " ms\n\n";

    // -------------------------------
    // EPS loop: reuse baseline_cost / baseline_total_time_ms
    // -------------------------------
    for (int ei = 0; ei < (int)EPS_list.size(); ++ei) {
        double eps = EPS_list[ei];

        cout << "\n################################\n";
        cout << "Now running for eps = " << eps << "\n";
        cout << "################################\n";
    // -------------------------------
    // 4. RCO pipeline: r_j = (1+eps_guess)^j, OPT in [1, 4N^2]
    //    Stop at FIRST successful j
    // -------------------------------
    double OPT_min = 1.0;
    double OPT_max = 4.0 * N * N;  // upper bound for OPT

    int t_max = (int)ceil( log(OPT_max / OPT_min) / log(1.0 + eps) );

    cout << "=== RCO pipeline: guesses r_j = (1+eps_guess)^j, j=0..t_max ===\n";
    cout << "eps   = " << eps  << " (for R_j construction)\n";
    cout << "t_max     = " << t_max     << " guesses\n\n";

    int firstSuccessJ = -1;
    int abortedRuns = 0;

    double firstCostOnP = numeric_limits<double>::infinity();
    double firstCostOnRj = numeric_limits<double>::infinity();

    double firstBuildMs = 0.0;
    double firstKmMs = 0.0;
    double firstTotalPipelineMs = 0.0;

    size_t firstRjSize = 0;

    bool foundFirstSuccess = false;

    for (int j = 0; j <= t_max; ++j) {
        double rj = OPT_min * pow(1.0 + eps, j);
        cout << "---- j = " << j << ", r_j = " << rj << " ----\n";

        // 4-1. Build R_j
        RjConfig cfg;
        cfg.n   = N;
        cfg.k   = k;
        cfg.eps = eps;
        cfg.rj  = rj;

        RepresentativeSetBuilder repBuilder(tree, cfg);

        auto tr0 = steady_clock::now();
        auto [ok, Rj] = repBuilder.build();
        auto tr1 = steady_clock::now();

        double buildRjMs = duration<double, milli>(tr1 - tr0).count();

        if (!ok) {
            cout << "  R_j build aborted (|K_j| too large). "
                 << "build time = " << buildRjMs << " ms\n";
            abortedRuns++;
            continue;
        }

        cout << "  R_j build success. |R_j| = " << Rj.size()
             << ", build time = " << buildRjMs << " ms\n";

        // 4-2. Run k-median++ on R_j (runsPerJ times, pick best on P)
        double bestCostThisJ_onP  = numeric_limits<double>::infinity();
        double bestCostThisJ_onRj = numeric_limits<double>::infinity();
        vector<Point> bestCentersThisJ;

        double kmTimeThisJ = 0.0;
        double totalPipelineMsThisJ = buildRjMs;

        for (int run = 0; run < runsPerJ; ++run) {
            uint32_t seed = 2000u + (uint32_t)ei * 100000u + (uint32_t)k * 1000u + (uint32_t)j * 10u + (uint32_t)run;

            auto tk0 = steady_clock::now();
            vector<Point> centersR = k_median_pp(Rj, k, maxKMiter, 50, 1e-3, seed);
            auto tk1 = steady_clock::now();

            double kmTimeMs = duration<double, milli>(tk1 - tk0).count();
            kmTimeThisJ += kmTimeMs;
            totalPipelineMsThisJ += kmTimeMs;

            double costR_onRj = k_median_cost(Rj, centersR); // cost on representative set
            double costR_onP  = k_median_cost(P,  centersR); // cost on original data

            cout << "    [R_j run " << run << "] "
                 << "time = " << kmTimeMs
                 << " ms, cost(R_j) = " << costR_onRj
                 << ", cost(P) = " << costR_onP << "\n";

            if (costR_onP < bestCostThisJ_onP) {
                bestCostThisJ_onP  = costR_onP;
                bestCostThisJ_onRj = costR_onRj;
                bestCentersThisJ   = centersR;
            }
        }

        cout << "  Best for this j: cost(P) = " << bestCostThisJ_onP
             << ", cost(R_j) = " << bestCostThisJ_onRj
             << ", k-median++ time (this j) = " << kmTimeThisJ << " ms"
             << ", total pipeline time (this j) = " << totalPipelineMsThisJ << " ms\n\n";

        // 4-3. First successful j: record and STOP
        firstSuccessJ = j;
        firstRjSize = Rj.size();
        firstBuildMs = buildRjMs;
        firstKmMs = kmTimeThisJ;
        firstTotalPipelineMs = totalPipelineMsThisJ;
        firstCostOnRj = bestCostThisJ_onRj;
        firstCostOnP = bestCostThisJ_onP;

        foundFirstSuccess = true;
        break;
    }

    // -------------------------------
    // 5. Summary and comparison
    // -------------------------------
    cout << "==============================\n";
    cout << "Summary:\n";
     cout << "  Baseline (P) best cost      = " << baseline_cost
          << " (k-median++ on P)\n";
     cout << "  Baseline (P) time (best run)= " << baseline_best_run_time_ms << " ms\n";
     cout << "  Baseline (P) total time     = " << baseline_total_time_ms << " ms\n\n";


    if (!foundFirstSuccess) {
        cout << "No successful R_j was built. Cannot compare RCO pipeline.\n";
            csv << N << "," << eps << "," << k << ","
        << baseline_cost << "," << baseline_best_run_time_ms << ","
        << -1 << "," << abortedRuns << "," << 0 << ","
        << numeric_limits<double>::infinity() << "," << numeric_limits<double>::infinity() << ","
        << 0.0 << "," << 0.0 << "," << 0.0 << ","
        << 0.0
        << "\n";

    continue; // move to next k
    }

    cout << "  RCO pipeline (FIRST success):\n";
    cout << "    First successful j            = " << firstSuccessJ << "\n";
    cout << "    Aborted runs (|K_j| too large)= " << abortedRuns << "\n";
    cout << "    |R_j|                         = " << firstRjSize << "\n";
    cout << "    Cost(P) using centers from Rj = " << firstCostOnP << "\n";
    cout << "    Cost(R_j)                     = " << firstCostOnRj << "\n";
    cout << "    R_j build time                = " << firstBuildMs << " ms\n";
    cout << "    k-median++ time on R_j        = " << firstKmMs << " ms\n";
    cout << "    Total pipeline time           = " << firstTotalPipelineMs << " ms\n\n";

    // -------------------------------
    // 6. Empirical approximation quality: eps_eff
    // -------------------------------
    double eps_eff = firstCostOnP / baseline_cost  - 1.0;
    if (eps_eff < 0) eps_eff = 0.0;

    cout << "==== Approximation quality (empirical) ====\n";
    cout << "  Baseline cost on P              = " << baseline_cost  << "\n";
    cout << "  First-success R_j cost on P     = " << firstCostOnP << "\n";
    cout << "  Effective epsilon (eps_eff)     = " << eps_eff
         << "  (i.e., cost_RCO <= (1 + eps_eff) * cost_P)\n";

    double eps_target = eps; // or eps_rep
    cout << "  Target epsilon (from construction/guess) = " << eps << "\n";
    if (eps_eff <= eps) {
        cout << "  => Empirically within (1 + " << eps_target
             << ")-approx of baseline.\n";
    } else {
        cout << "  => Empirically worse than (1 + " << eps_target
             << ")-approx relative to baseline.\n";
    }

    // -------------------------------
    // 7. Time comparison (end-to-end) + edges/0.1s (complete graph)
    // -------------------------------
    cout << "\n==== Time comparison (end-to-end) ====\n";
    cout << "  Baseline: one k-median++ on P      ~ " << baseline_best_run_time_ms << " ms (best run)\n";
    cout << "  RCO: first-success pipeline time  ~ " << firstTotalPipelineMs << " ms\n";

    cout << "\n==== edges / 0.1 sec (complete graph) ====\n";
    long double m = complete_edges(N);
    cout << "  Complete edges m = n(n-1)/2 = " << (double)m << "\n";
    cout << "  Baseline edges/0.1s = " << (double)edges_per_0p1sec(N, baseline_best_run_time_ms) << "\n";
    cout << "  RCO edges/0.1s      = " << (double)edges_per_0p1sec(N, firstTotalPipelineMs) << "\n";

    csv << N << "," << k << ","
    << baseline_cost  << "," << baseline_best_run_time_ms << ","
    << firstSuccessJ << "," << abortedRuns << "," << firstRjSize << ","
    << firstCostOnP << "," << firstCostOnRj << ","
    << firstBuildMs << "," << firstKmMs << "," << firstTotalPipelineMs << ","
    << eps_eff
    << "\n";

          }
     }
}
     csv.close();
     return 0;
}
