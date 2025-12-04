#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>  

#include "point.h"
#include "RangeCountingOracle.h"
#include "Representative.h"
#include "k_median++.h"

using namespace std;
using namespace std::chrono;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // -------------------------------
    // 0. Parameter settings
    // -------------------------------
    const int N = 10000;      // number of original points
    const int k = 100;           // number of clusters
    const double eps_rep = 0.2;  // ε used to construct R_j (dense/sparse threshold)
    const double eps_guess = 0.2;// ε used for OPT guesses r_j = (1+ε)^j
    const int maxKMiter = 20;    // max Lloyd iterations for k-median++
    const int baselineRuns = 3;  // #runs on P, take the best cost
    const int runsPerJ = 1;      // #runs of k-median++ per R_j

    mt19937 rng(123);            // global RNG for this experiment

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

    // -------------------------------
    // 3. Baseline: run k-median++ directly on P
    // -------------------------------
    double bestCostP = numeric_limits<double>::infinity();
    vector<Point> bestCentersP;
    double bestTimeP = 0.0;

    cout << "=== Baseline: k-median++ on P ===\n";
    for (int run = 0; run < baselineRuns; ++run) {
        uint32_t seed = 1000 + run;

        auto tb0 = steady_clock::now();
        vector<Point> centersP = k_median_pp(P, k, maxKMiter, 50, 1e-3, seed);
        auto tb1 = steady_clock::now();

        double timeMs = duration<double, milli>(tb1 - tb0).count();
        double costP = k_median_cost(P, centersP);

        cout << "[P run " << run << "] time = " << timeMs
             << " ms, cost(P) = " << costP << "\n";

        if (costP < bestCostP) {
            bestCostP = costP;
            bestCentersP = centersP;
            bestTimeP = timeMs;
        }
    }
    cout << "Best baseline on P: cost = " << bestCostP
         << ", time = " << bestTimeP << " ms\n\n";

    // -------------------------------
    // 4. RCO pipeline: r_j = (1+eps_guess)^j, OPT in [1, 4N^2]
    //    For each j:
    //      - Build R_j (measure time, but exclude from algorithm-time comparison)
    //      - Run k-median++ on R_j and sum these times
    // -------------------------------
    double OPT_min = 1.0;
    double OPT_max = 4.0 * N * N;  // upper bound for OPT

    int t_max = (int)ceil( log(OPT_max / OPT_min) / log(1.0 + eps_guess) );

    cout << "=== RCO pipeline: guesses r_j = (1+eps_guess)^j, j=0..t_max ===\n";
    cout << "eps_rep   = " << eps_rep   << " (for R_j construction)\n";
    cout << "eps_guess = " << eps_guess << " (for OPT guesses)\n";
    cout << "t_max     = " << t_max     << " guesses\n\n";

    // Aggregated statistics for the RCO pipeline
    double totalKMTimeRj = 0.0;  // total time spent in k-median++ on all R_j
    double totalBuildRjMs = 0.0; // total time to construct all R_j (reference only)
    int    numSuccessfulJ = 0;   // #j for which R_j was successfully constructed

    double bestCostR_onP  = numeric_limits<double>::infinity(); // min_j cost(P, C_j)
    double bestCostR_onRj = numeric_limits<double>::infinity(); // min_j cost(R_j, C_j)
    vector<Point> bestCentersR;
    int bestJ = -1;

    for (int j = 0; j <= t_max; ++j) {
        double rj = OPT_min * pow(1.0 + eps_guess, j);
        cout << "---- j = " << j << ", r_j = " << rj << " ----\n";

        // 4-1. Build R_j
        RjConfig cfg;
        cfg.n   = N;
        cfg.k   = k;
        cfg.eps = eps_rep;
        cfg.rj  = rj;

        RepresentativeSetBuilder repBuilder(tree, cfg);

        auto tr0 = steady_clock::now();
        auto [ok, Rj] = repBuilder.build();
        auto tr1 = steady_clock::now();

        double buildRjMs = duration<double, milli>(tr1 - tr0).count();
        totalBuildRjMs += buildRjMs;

        if (!ok) {
            cout << "  R_j build aborted (|K_j| too large). "
                 << "build time = " << buildRjMs << " ms\n";
            // Treat this j as a too-small guess and skip it
            continue;
        }

        numSuccessfulJ++;
        cout << "  R_j build success. |R_j| = " << Rj.size()
             << ", build time = " << buildRjMs << " ms\n";

        // 4-2. Run k-median++ on R_j
        double bestCostThisJ_onP  = numeric_limits<double>::infinity();
        double bestCostThisJ_onRj = numeric_limits<double>::infinity();
        vector<Point> bestCentersThisJ;
        double totalKMTimeThisJ = 0.0;

        for (int run = 0; run < runsPerJ; ++run) {
            uint32_t seed = 2000 + j * 100 + run;

            auto tk0 = steady_clock::now();
            vector<Point> centersR = k_median_pp(Rj, k, maxKMiter, 50, 1e-3, seed);
            auto tk1 = steady_clock::now();

            double kmTimeMs = duration<double, milli>(tk1 - tk0).count();
            totalKMTimeThisJ += kmTimeMs;

            double costR_onRj = k_median_cost(Rj, centersR); // cost on representative set
            double costR_onP  = k_median_cost(P,  centersR); // cost on original data

            cout << "    [R_j run " << run << "] "
                 << "time = " << kmTimeMs
                 << " ms, cost(R_j) = " << costR_onRj
                 << ", cost(P) = " << costR_onP << "\n";

            // For this j, keep the run that is best on P
            if (costR_onP < bestCostThisJ_onP) {
                bestCostThisJ_onP  = costR_onP;
                bestCostThisJ_onRj = costR_onRj;
                bestCentersThisJ   = centersR;
            }
        }

        cout << "  Best for this j: cost(P) = " << bestCostThisJ_onP
             << ", cost(R_j) = " << bestCostThisJ_onRj
             << ", total k-median++ time for this j = "
             << totalKMTimeThisJ << " ms\n\n";

        totalKMTimeRj += totalKMTimeThisJ;

        // 4-3. Update global best j* based on cost(P)
        if (bestCostThisJ_onP < bestCostR_onP) {
            bestCostR_onP  = bestCostThisJ_onP;
            bestCostR_onRj = bestCostThisJ_onRj;
            bestCentersR   = bestCentersThisJ;
            bestJ          = j;
        }
    }

    // -------------------------------
    // 5. Summary and comparison
    // -------------------------------
    cout << "==============================\n";
    cout << "Summary:\n";
    cout << "  Baseline (P) best cost      = " << bestCostP
         << " (k-median++ on P)\n";
    cout << "  Baseline (P) time per run   = " << bestTimeP << " ms (best run)\n\n";

    if (bestJ == -1) {
        cout << "No successful R_j was built. Cannot compare RCO pipeline.\n";
        return 0;
    }

    cout << "  RCO pipeline:\n";
    cout << "    #successful j             = " << numSuccessfulJ
         << " out of " << (t_max + 1) << "\n";
    cout << "    Best j*                   = " << bestJ << "\n";
    cout << "    Best R_j-based cost(P)    = " << bestCostR_onP << "\n";
    cout << "    Corresponding cost(R_j)   = " << bestCostR_onRj << "\n";
    cout << "    Total k-median++ time\n"
         << "      over all successful R_j = " << totalKMTimeRj << " ms\n";
    cout << "      (R_j construction time is excluded from this sum)\n";
    cout << "    Total R_j construction time (all j) = "
         << totalBuildRjMs << " ms (for reference only)\n\n";

    // -------------------------------
    // 6. Empirical (1+ε)-approx quality: eps_eff
    // -------------------------------
    double eps_eff = bestCostR_onP / bestCostP - 1.0;
    if (eps_eff < 0) eps_eff = 0.0;

    cout << "==== Approximation quality (empirical) ====\n";
    cout << "  Baseline cost on P        = " << bestCostP << "\n";
    cout << "  Best R_j-based cost on P  = " << bestCostR_onP << "\n";
    cout << "  Effective epsilon (eps_eff) = " << eps_eff
         << "  (i.e., cost_RCO <= (1 + eps_eff) * cost_P)\n";

    // Compare eps_eff with eps_guess or eps_rep to see if behavior matches the target ε
    double eps_target = eps_guess; // or eps_rep
    cout << "  Target epsilon (from construction/guess) = " << eps_target << "\n";
    if (eps_eff <= eps_target) {
        cout << "  => Empirically within (1 + " << eps_target
             << ")-approx of baseline.\n";
    } else {
        cout << "  => Empirically worse than (1 + " << eps_target
             << ")-approx relative to baseline.\n";
    }

    // -------------------------------
    // 7. Time comparison (ignoring R_j construction)
    // -------------------------------
    cout << "\n==== Time comparison (ignoring R_j construction) ====\n";
    cout << "  Baseline: one k-median++ on P     ~ " << bestTimeP << " ms (per run)\n";
    cout << "  RCO: sum over j of k-median++ on R_j ~ "
         << totalKMTimeRj << " ms\n";
    cout << "       (each R_j is smaller than P, but we run on many j's)\n";

    return 0;
}
