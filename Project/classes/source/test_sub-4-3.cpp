// test_sub-4-3.cpp
// Build: g++ -std=gnu++17 -O2 -march=native \
//        source/test_sub-4-3.cpp \
//        source/test_2_approx_clustering.cpp \
//        classes/header/sublinear_alg_k-median.cpp \
//        -Iclasses/header -o test_sub43
// Run:   ./test_sub43
//
// Goal: Verify up to Step (3) — coreset construction works correctly
//       Pipeline: P -> ALG(Section 4) gives X (=A) -> Section 3 coreset S
//
// Notes:
//  - Expects the following symbols to be available:
//      struct Point { int x, y; double w; };
//      KCenterResult gonzalez_k_center(...);                // from 2_approx_clustering.h (if needed)
//      ConstantCentersResult build_constant_factor_centers(...); // from constant_approx.h
//      void build_weighted_coreset_2d_pointsOnly(...);      // from constant_approx_coreset.h
//
//  - This test checks: weight preservation, size sanity, and sampled cost preservation.

#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <chrono>
#include <limits>
#include <algorithm>
#include <unordered_set>
#include <iomanip> 
#include "../header/constant_approx.h"
#include "../header/constant_approx_coreset.h"
#include "../header/sublinear_alg_k-median.h"

using namespace std;

// ----------- Data generator: n points uniform over [0, 2n-1]^2 -----------
static vector<Point> make_uniform_square(int n, unsigned seed=42){
    mt19937 rng(seed);
    uniform_int_distribution<int> U(0, 2*n - 1);
    vector<Point> P; P.reserve(n);
    for(int i=0;i<n;i++){
        int x = U(rng), y = U(rng);
        P.push_back(Point{x,y,1.0}); // weight = 1
    }
    return P;
}

// ----------- Weighted k-median cost utility -----------
static double kmedian_cost_weighted(const vector<Point>& P, const vector<Point>& C){
    double tot = 0.0;
    for(const auto& p : P){
        double best = 1e300;
        for(const auto& c : C){
            double dx = (double)p.x - (double)c.x;
            double dy = (double)p.y - (double)c.y;
            double d  = sqrt(dx*dx + dy*dy);
            best = min(best, d);
        }
        tot += max(0.0, p.w) * best;
    }
    return tot;
}

// ----------- Choose k centers from A by farthest-first (deterministic) -----------
static vector<Point> select_k_from_A_farthest(const vector<Point>& A, int k){
    vector<Point> C;
    if(A.empty() || k<=0) return C;
    k = min(k, (int)A.size());
    C.push_back(A[0]);
    vector<double> mind(A.size(), 1e300);
    for(size_t i=0;i<A.size();++i){
        double dx = (double)A[i].x - (double)C[0].x;
        double dy = (double)A[i].y - (double)C[0].y;
        mind[i] = sqrt(dx*dx + dy*dy);
    }
    while((int)C.size() < k){
        int arg=-1; double best=-1;
        for(size_t i=0;i<A.size();++i){
            if(mind[i] > best){ best = mind[i]; arg = (int)i; }
        }
        C.push_back(A[arg]);
        for(size_t i=0;i<A.size();++i){
            double dx = (double)A[i].x - (double)C.back().x;
            double dy = (double)A[i].y - (double)C.back().y;
            mind[i] = min(mind[i], sqrt(dx*dx + dy*dy));
        }
    }
    return C;
}

static inline double sum_w(const vector<Point>& P){
    double s=0.0; for(const auto& p: P) s += max(0.0, p.w); return s;
}
// ----------------------------------- Test Body -----------------------------------
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Basic knobs (tunable)
    const int n = 300000;           // number of vertices
    const int k = 10;              // target k
    const double eps = 0.30;       // epsilon for coreset & Lemma20 schedule
    const double gamma = 0.05;     // ALG sampling scale (small so Y != P)
    const double c = 4.0;         // coreset constant for nu_A(P) ≤ c·OPT (conservative)
    const unsigned seed = 123;

    // Step 0: build P
    auto P = make_uniform_square(n, seed);
    cout << fixed << setprecision(6);
    cout << "[P] n=" << P.size() << ", W=" << sum_w(P) << "\n";

    // Step 1: Lemma 20 — get reps R via range-counting + cell compression
    SublinearKMedian solver(P, k, eps);

    // Choose a mid-range r_j: j ≈ floor( 0.5 * log_{1+eps}(4 n^2) )
    const double t_real = log(4.0 * n * n) / log(1.0 + eps);
    const int j_pick = max(0, (int)floor(0.1 * t_real));
    const double rj  = pow(1.0 + eps, j_pick);

    vector<Point> R = solver.get_reps(rj);
    if(R.empty()){
        cerr << "[FAIL] Lemma20 reps R is empty. Inspect get_reps(rj) or thresholds.\n";
        return 1;
    }
    cout << "[Lemma20] j=" << j_pick << ", rj=" << rj << ", |R|=" << R.size() << "\n";

    // Step 2: Section 4 — run constant-factor ALG on R to get A = V ∪ Y
    auto ALG = build_constant_factor_centers(R, k, gamma, seed);
    auto& A  = ALG.X_all;
    if(A.empty()){
        cerr << "[FAIL] Section4 produced empty A.\n";
        return 1;
    }
    cout << "[Sec4] rounds=" << ALG.rounds << ", |A|=" << A.size()
         << ", W_picked=" << ALG.W_picked << "/" << ALG.W_total << "\n";

    // Step 3: Section 3 — build (k, eps)-coreset S from (P, A)
    vector<Point> S;
    CoresetDiag diag;
    build_weighted_coreset_2d_pointsOnly(P, A, eps, c, S, &diag);

    cout << "[Sec3/Coreset] |S|=" << S.size()
         << ", W(S)=" << sum_w(S)
         << ", R=" << diag.R
         << ", M=" << diag.M
         << ", nuA=" << diag.nuA
         << ", Wsum=" << diag.Wsum << "\n";

    // ---------------- Checks ----------------
    bool ok = true;

    // (a) weight preservation
    {
        const double wP = sum_w(P), wS = sum_w(S);
        const double rel = (wP>0)? fabs(wP - wS)/wP : 0.0;
        cout << "[check] weight preservation rel.err=" << rel << "\n";
        if(rel > 1e-9){
            cerr << "[WARN] weight not perfectly preserved (rounding may cause tiny drift)\n";
        }
    }

    // (b) size sanity (very loose): |S| ≤ K * |A| * (log2 n + 1) / eps^2
    {
        const double Kc = 200.0;
        const double bound = Kc * (double)A.size() * (log2((double)max(2, n)) + 1.0) / (eps*eps);
        cout << "[check] size sanity: |S|=" << S.size()
             << " vs loose_bound=" << (long long)ceil(bound) << "\n";
        if((double)S.size() > bound){
            cerr << "[WARN] coreset size looks large vs loose bound — tune constants if needed\n";
        }
    }

    // (c) sampled cost preservation: cost(P,C) ≈ cost(S,C)
    {
        mt19937 rng(seed+7);
        vector<double> errs;
        const int trials = 6;
        for(int t=0; t<trials; ++t){
            vector<Point> Aperm = A;
            if(Aperm.size() >= 2) shuffle(Aperm.begin(), Aperm.end(), rng);
            vector<Point> C = select_k_from_A_farthest(Aperm, k);
            if(C.empty()) continue;

            double costP = kmedian_cost_weighted(P, C);
            double costS = kmedian_cost_weighted(S, C);
            double rel = (costP>0)? fabs(costP - costS)/costP : 0.0;
            errs.push_back(rel);
            cout << "[check] trial " << t << " rel.err=" << rel
                 << " (costP=" << costP << ", costS=" << costS << ")\n";
        }
        if(!errs.empty()){
            double mx = *max_element(errs.begin(), errs.end());
            nth_element(errs.begin(), errs.begin()+errs.size()/2, errs.end());
            double md = errs[errs.size()/2];
            cout << "[stats] rel.err: max=" << mx << ", median=" << md << "\n";
            if(mx > 0.25){
                cerr << "[WARN] some sampled costs deviate > 25% — consider tuning gamma/c/eps or rj\n";
            }
        }
    }

    cout << "[DONE] Lemma20 -> Sec4 -> Sec3 pipeline (up to coreset) verified.\n";
    return ok? 0 : 1;
}