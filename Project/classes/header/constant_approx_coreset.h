// ============================================================
// Weighted coreset (Section 3.1.1, k-median, 2D)
//   - 입력 P: 가중 점집합 (reps)
//   - 입력 A: 가중 "센터" 점집합 (centers를 Point로 표현)
//   - 출력 S_out: 가중 코어셋
//   - d=2, 거리 = Euclidean
// ============================================================
#include "point.h"
#include <vector>
#include <cmath>
#include <unordered_map>
#include <limits>

// point.h 에 다음과 같은 구조가 있다고 가정:
// struct Point { int x, y; double w; };

static inline double dist2d_double(double ax, double ay, double bx, double by) {
    double dx = ax - bx, dy = ay - by;
    return std::sqrt(dx*dx + dy*dy);
}

struct CoresetDiag {
    double R = 0.0;     // scale = ν_A(P) / (c · W)
    int    M = 0;       // #rings (2^M R ≥ maxDist)
    double nuA = 0.0;   // Σ w(p)·dist(p,A)
    double Wsum = 0.0;  // Σ w(p)
};

static void build_weighted_coreset_2d_pointsOnly(
    const std::vector<Point>& P,   // weighted input points (reps)
    const std::vector<Point>& A,   // weighted centers as Points (x,y 사용; w는 메타)
    double epsilon,                // ε in (0,1)
    double c,                      // constant s.t. ν_A(P) ≤ c · OPT
    std::vector<Point>& S_out,     // OUTPUT: weighted coreset
    CoresetDiag* diag_out = nullptr
){
    S_out.clear();
    CoresetDiag diag;

    const int n = (int)P.size();
    const int m = (int)A.size();
    if (n == 0 || m == 0) { if (diag_out) *diag_out = diag; return; }

    const int d = 2;
    const double eps = epsilon;

    // 1) 총 가중치 W, 최근접 센터 및 거리, ν_A(P)
    double W = 0.0;
    for (const auto& p : P) W += std::max(0.0, p.w);
    diag.Wsum = W;

    std::vector<int>    nearIdx(n, -1);
    std::vector<double> nearDist(n, 0.0);

    auto nearest_c = [&](double px, double py){
        int best = -1; double bd = std::numeric_limits<double>::infinity();
        for (int i=0;i<m;++i){
            double dmin = dist2d_double(px, py, (double)A[i].x, (double)A[i].y);
            if (dmin < bd){ bd = dmin; best = i; }
        }
        return std::pair<int,double>(best, bd);
    };

    double nuA = 0.0;
    for (int i=0;i<n;++i){
        auto [idx, dmin] = nearest_c((double)P[i].x, (double)P[i].y);
        nearIdx[i]  = idx;
        nearDist[i] = dmin;
        nuA += P[i].w * dmin;  // 점의 가중치만 반영
    }
    diag.nuA = nuA;

    // 2) 스케일 R = ν_A(P) / (c · W)
    if (W <= 0.0) { if (diag_out) *diag_out = diag; return; }
    double R = nuA / (c * W);
    diag.R = R;

    // 엣지 케이스: 모두 센터에 일치 (R==0) → 하나의 대표만 반환
    if (!(R > 0.0)) {
        double totW = 0.0; for (auto& p : P) totW += p.w;
        if (!P.empty()) S_out.push_back(Point{ P.front().x, P.front().y, totW });
        if (diag_out) { diag.M = 0; *diag_out = diag; }
        return;
    }

    // 3) 링 수 M: 2^M R ≥ maxDist
    double maxDist = 0.0;
    for (int i=0;i<n;++i) maxDist = std::max(maxDist, nearDist[i]);
    if (!(maxDist > 0.0)) {
        double totW = 0.0; for (auto& p : P) totW += p.w;
        if (!P.empty()) S_out.push_back(Point{ P.front().x, P.front().y, totW });
        if (diag_out) { diag.M = 0; *diag_out = diag; }
        return;
    }
    double ratio = std::max(1.0, maxDist / R);
    int M = (int)std::ceil(std::log2(ratio));
    diag.M = M;

    // 4) (center i, ring j, grid index gx, gy) 해시 버킷
    struct Key {
        int ci, ring;
        long long gx, gy;
        bool operator==(const Key& o) const {
            return ci==o.ci && ring==o.ring && gx==o.gx && gy==o.gy;
        }
    };
    struct KeyHash {
        size_t operator()(Key const& k) const noexcept {
            size_t h = 1469598103934665603ULL;
            auto mix = [&](long long v){ h ^= (size_t)v; h *= 1099511628211ULL; };
            mix(k.ci); mix(k.ring); mix(k.gx); mix(k.gy);
            return h;
        }
    };
    struct Agg { double repx=0, repy=0; double wsum=0; bool has=false; };

    std::unordered_map<Key, Agg, KeyHash> buckets;
    buckets.reserve((size_t)n);

    auto twoPowJ_R = [&](int j)->double { return std::ldexp(R, j); }; // R * 2^j

    // 5) 각 점을 (센터별 상대좌표) 격자 셀로 스냅 → 셀별 가중치 합산
    for (int i=0;i<n;++i){
        const auto &p = P[i];
        const int ci  = nearIdx[i];
        const double dp = nearDist[i];

        // ring index j
        int j = (dp < R) ? 0 : std::min(M, (int)std::floor(std::log2(dp / R)));

        // grid cell size r_j = (ε * (R·2^j)) / (10 c d)
        double base = twoPowJ_R(j);
        double cell = (eps * base) / (10.0 * c * (double)d);
        if (!(cell > 0.0)) cell = 1e-12; // 안전장치

        // grid index (센터 A[ci] 기준 상대좌표)
        long long gx = (long long)std::floor(((double)p.x - (double)A[ci].x) / cell);
        long long gy = (long long)std::floor(((double)p.y - (double)A[ci].y) / cell);

        Key key{ci, j, gx, gy};
        auto it = buckets.find(key);
        if (it == buckets.end()){
            Agg a; a.has = true; a.repx = (double)p.x; a.repy = (double)p.y; a.wsum = p.w;
            buckets.emplace(key, a);
        } else {
            it->second.wsum += p.w;
        }
    }

    // 6) 비어있지 않은 셀마다 대표 1개 + 집계 가중치 → 코어셋 방출
    S_out.reserve(buckets.size());
    for (auto &kv : buckets){
        const auto& a = kv.second;
        S_out.emplace_back( (int)std::llround(a.repx), (int)std::llround(a.repy), a.wsum );
    }

    if (diag_out) *diag_out = diag;
}
