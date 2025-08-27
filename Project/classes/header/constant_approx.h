#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include "2_approx_clustering.h" 

// --- rho computation --------------------------------------------------------
// rho = ceil(gamma * k * (log2 W)^2).
static std::size_t compute_rho(std::size_t W, std::size_t n, int k, double gamma) {
    if (W == 0 || n == 0 || k <= 0 || gamma <= 0.0) return 0;

    // log term: at least 1
    double log_term = std::max(1.0, std::log2(static_cast<double>(std::max<std::size_t>(2, W))));
    double rho_real = gamma * static_cast<double>(k) * (log_term * log_term);

    std::size_t rho = static_cast<std::size_t>(std::ceil(rho_real));
    if (rho < 1) rho = 1;
    if (rho > n) rho = n;   // 샘플 수는 점 개수를 넘지 못함
    return rho;
}


// --- sampling Y (without/with replacement) ----------------------------------
// If rho >= n, returns P. 'seed' controls reproducibility.
static std::vector<Point> sample_Y(const std::vector<Point>& P,
                                   int k,
                                   double gamma,
                                   unsigned seed = 42) {
    const std::size_t n = P.size();
    std::vector<Point> Y;
    if (n == 0) return Y;

    // 총 weight
    double W_real = 0.0;
    for (const auto& p : P) W_real += std::max(0.0, p.w);

    // rho 계산
    const std::size_t rho = compute_rho(static_cast<std::size_t>(std::ceil(W_real)), n, k, gamma);
    if (rho == 0) return Y;
    if (rho >= n) { Y = P; return Y; }  

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    struct Key { double key; std::size_t idx; };
    std::vector<Key> keys;
    keys.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        double wi = std::max(0.0, P[i].w);
        if (wi <= 0.0) continue; 
        double u = std::max(1e-18, U(rng)); 
        double key = -std::log(u) / wi;
        keys.push_back({key, i});
    }

    if (keys.size() <= rho) {
        Y.reserve(keys.size());
        for (auto& kv : keys) Y.push_back(P[kv.idx]);
        return Y;
    }

    std::nth_element(keys.begin(), keys.begin() + rho, keys.end(),
                     [](const Key& a, const Key& b){ return a.key < b.key; });
    keys.resize(rho);

    Y.reserve(rho);
    for (auto& kv : keys) Y.push_back(P[kv.idx]);
    return Y;
}


// Hash key for (x,y) : For make_X
static inline std::uint64_t xy_key(int x, int y) {
    // Pack two 32-bit signed ints into one 64-bit key (stable, fast)
    return ( (std::uint64_t)(std::uint32_t)x << 32 ) ^ (std::uint32_t)y;
}

// Build X = V ∪ Y with hash-based O(|V| + |Y|) dedup
static std::vector<Point> make_X(const std::vector<Point>& V,
                                 const std::vector<Point>& Y) {
    std::vector<Point> X;
    X.reserve(V.size() + Y.size());

    // Use a hash set of coordinate keys to dedup by (x,y)
    std::unordered_set<std::uint64_t> seen;
    seen.reserve(V.size() + Y.size());
    seen.max_load_factor(0.7f); // reduce rehash probability

    // Insert V first (all unique by definition of Gonzalez, but dedup anyway)
    for (const auto& v : V) {
        const std::uint64_t key = xy_key(v.x, v.y);
        if (seen.insert(key).second) {
            X.push_back(v);
        }
    }

    // Insert Y only if not already present in V (or previous Y)
    for (const auto& y : Y) {
        const std::uint64_t key = xy_key(y.x, y.y);
        if (seen.insert(key).second) {
            X.push_back(y);
        }
    }

    return X;
}

// For each point p in P, compute r(p) = dist(p, X)
static std::vector<double> compute_r(const std::vector<Point>& P,
                                     const std::vector<Point>& X) {
    std::vector<double> r;
    r.reserve(P.size());
    for (const auto& p : P) {
        double best = std::numeric_limits<double>::infinity();
        for (const auto& c : X) {
            best = std::min(best, dist(p, c));
        }
        r.push_back(best);
    }
    return r;
}

static inline int ring_index_by_W(double r, double L, double W) {
    const double eps = 1e-12;
    if (r <= L/(4.0*W) + eps) return 0;          // [0, L/(4W)]
    if (r >= 2.0*L*W - eps)  return 1000000;     // [2LW, ∞)

    // For i >= 1: bucket range is [ 2^{i-3} * L/W , 2^{i-2} * L/W )
    // Equivalent: 2^{i-3} <= (rW)/L < 2^{i-2}
    // Hence: i = floor(log2((rW)/L)) + 3
    double t = std::max(eps, (r * W) / L);
    int i = (int)std::floor(std::log2(t)) + 3;
    if (i < 1) i = 1; // 안전장치 (이론상 도달 X)
    return i;         // 0, 1, 2, ... (0은 특례 버킷)
}

struct BucketPartition {
    std::vector<int> bucket_idx;                       // 각 점 p의 버킷 번호
    std::unordered_map<int, std::vector<int>> members; // 버킷 -> 점 인덱스 목록
    std::unordered_map<int, double> wsum;              // 버킷 -> 가중치 합
    double W = 0.0;                                    // 전체 가중치
    double beta = 0.0;                                 // β = W / (20 log W)
    int alpha = 0;                                     // w(P_i) > 2β 인 마지막 버킷
};

static BucketPartition partition_by_distance_weighted(const std::vector<Point>& P,
                                                      const std::vector<Point>& X,
                                                      double L) {
    BucketPartition out;
    const int n = (int)P.size();
    out.bucket_idx.assign(n, 0);

    // 총 가중치 W
    double W = 0.0;
    for (const auto& p : P) W += std::max(0.0, p.w);
    out.W = std::max(2.0, W); // log 안정화

    // β = W / (20 log W)
    double logW = std::max(1.0, std::log2(out.W));
    out.beta = out.W / (20.0 * logW);

    // r(p) 계산 (이미 있는 compute_r 사용)
    std::vector<double> r = compute_r(P, X);

    // 버킷 채우기
    out.members.reserve(n);
    out.wsum.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (P[i].w <= 0.0) continue; // 가중치 0은 무시
        int b = ring_index_by_W(r[i], L, out.W);
        out.bucket_idx[i] = b;
        out.members[b].push_back(i);
        out.wsum[b] += P[i].w;
    }

    // α = w(P_i) > 2β를 만족하는 가장 큰 버킷 인덱스
    std::vector<int> keys; keys.reserve(out.wsum.size());
    for (auto& kv : out.wsum) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    out.alpha = 0;
    for (int b : keys) {
        if (out.wsum[b] > 2.0 * out.beta) out.alpha = std::max(out.alpha, b);
    }

    return out;
}

// (B) V도 포함하는 P':  P' = V ∪ (⋃_{i≤α} P_i)
struct PPrimeWithV {
    std::vector<char> keep;     // P의 각 인덱스가 P'에 포함되면 1
    std::vector<int>  indices;  // 포함된 인덱스 목록
    double W_prime = 0.0;       // P'의 가중치 합
};

static PPrimeWithV build_P_prime_by_alpha_and_V(const std::vector<Point>& P,
                                                const BucketPartition& part,
                                                const std::vector<Point>& V)
{
    PPrimeWithV out;
    out.keep.assign((int)P.size(), 0);

    // 1) V 포함 (좌표 동일 비교)
    for (int i = 0; i < (int)P.size(); ++i) {
        for (const auto& v : V) {
            if (P[i].x == v.x && P[i].y == v.y) {
                if (!out.keep[i]) {
                    out.keep[i] = 1;
                    out.indices.push_back(i);
                    out.W_prime += std::max(0.0, P[i].w);
                }
                break;
            }
        }
    }

    // 2) alpha 이하 버킷 포함
    for (const auto& kv : part.members) {
        int b = kv.first;
        if (b <= part.alpha) {
            for (int idx : kv.second) {
                if (!out.keep[idx]) {
                    out.keep[idx] = 1;
                    out.indices.push_back(idx);
                    out.W_prime += std::max(0.0, P[idx].w);
                }
            }
        }
    }
    return out;
}

static inline double total_weight(const std::vector<Point>& P) {
    double W = 0.0;
    for (const auto& p : P) W += std::max(0.0, p.w);
    return W;
}


struct ConstantCentersResult {
    std::vector<Point> X_all;   // 누적된 상수배 근사 센터 집합
    std::vector<char>  picked;  // 원본 P에서 제거되었는지(=P'에 포함되었는지)
    int rounds = 0;
    double W_total = 0.0;
    double W_picked = 0.0;
};

static ConstantCentersResult build_constant_factor_centers(
    const std::vector<Point>& P,
    int k,
    double gamma,
    unsigned seed = 42)
{
    ConstantCentersResult out;
    const int n = (int)P.size();
    out.picked.assign(n, 0);
    out.W_total = total_weight(P);

    // Compute dynamic round cap and stopping tolerances
    const double Wtot = out.W_total;                       // total weight
    const double logW = std::log2(std::max(2.0, Wtot));    // guard against log(0)
    const int    max_rounds = std::max(64, (int)std::ceil(8 * logW)); 

    std::vector<Point> X_all;
    int it = 0;

    while (out.W_picked + 1e-12 < out.W_total) {
        // (1) 남은 점 모으기 + 원본 인덱스 매핑
        std::vector<Point> remain; remain.reserve(n);
        std::vector<int>   r_idx;  r_idx.reserve(n);
        for (int i = 0; i < n; ++i) if (!out.picked[i]) {
            remain.push_back(P[i]);
            r_idx.push_back(i);
        }
        if (remain.empty()) break;

        // (2) Gonzalez: V와 반경 L
        auto R = gonzalez_k_center(remain, k);
        std::vector<Point> V = R.centers;
        if (R.farthest_index >= 0 && R.farthest_index < (int)remain.size()) {
            V.push_back(remain[R.farthest_index]); // 논문대로 farthest 포함
        }
        double L = R.L; // 거리 단위

        // (3) 가중치 무복원 샘플 Y
        std::vector<Point> Y = sample_Y(remain, k, gamma, seed + it);

        // (4) X = V ∪ Y
        std::vector<Point> X = make_X(V, Y);

        // (5) 거리-버킷 분할 → α
        auto part = partition_by_distance_weighted(remain, X, L);

        // (6) P' = V ∪ (⋃_{i≤α} P_i)
        auto Pprime = build_P_prime_by_alpha_and_V(remain, part, V);

        // (7) 원본 인덱스로 반영 + 가중치 누적
        for (int j = 0; j < (int)remain.size(); ++j) {
            if (Pprime.keep[j]) {
                int i = r_idx[j];
                if (!out.picked[i]) {
                    out.picked[i] = 1;
                    out.W_picked += std::max(0.0, P[i].w);
                }
            }
        }

        // (8) X_all 누적 (좌표 중복 제거)
        X_all = make_X(X_all, X);

        ++it;
        if (it > max_rounds) { std::cerr << "[warn] too many rounds\n"; break; }

        // (옵션) 라운드 로그
        std::cerr << "[round " << it << "] "
                << "L=" << L
                << ", |V|=" << V.size()
                << ", |Y|=" << Y.size()
                << ", |X|=" << X.size()
                << ", |X_all|=" << X_all.size()
                << ", W_picked=" << out.W_picked
                << " / " << out.W_total
                << "\n";

    }

    out.rounds = it;
    out.X_all = std::move(X_all);
    return out;
}