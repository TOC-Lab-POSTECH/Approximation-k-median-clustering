#include "point.h"
#include "RangeCountingOracle.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>     // sqrt, log, pow, ceil, round
#include <iomanip>   // fixed, setprecision>
#include <limits>

using namespace std;

/* ======================
   거리/코스트 유틸
   ====================== */

static inline double dist2d(double ax, double ay, double bx, double by) {
    double dx = ax - bx, dy = ay - by;
    return sqrt(dx*dx + dy*dy);
}

static inline double weighted_cost_to_point(const vector<Point>& R,
                                            const vector<double>& w,
                                            int cx, int cy) {
    double cost = 0.0;
    int m = (int)R.size();
    for (int i=0;i<m;++i) cost += w[i] * dist2d(R[i].x, R[i].y, cx, cy);
    return cost;
}

/* ======================
   Alg_ε k-med : TEST STUB (정수 격자 센터)
   - Lloyd-style + Weiszfeld(연속) -> 정수 스냅 -> 주변 1-링 탐색
   - 실제 [21] 알고리즘이 아닌 테스트용 스텁입니다.
   ====================== */

// 클러스터에 대해 Weiszfeld로 연속 후보를 구하고 정수 격자로 스냅 + 1-링 탐색
static Point discrete_geometric_median_snap(const vector<Point>& R,
                                            const vector<int>& idxs,
                                            const vector<double>& w,
                                            Point init_int,
                                            int max_iter_weiszfeld = 50) {
    if (idxs.empty()) return init_int;

    // 1) Weiszfeld (연속)
    double x = init_int.x, y = init_int.y;
    const double EPS = 1e-7;
    for (int it=0; it<max_iter_weiszfeld; ++it) {
        double numx = 0.0, numy = 0.0, denom = 0.0;
        bool coincide = false;
        for (int id : idxs) {
            double px = R[id].x, py = R[id].y;
            double wi = w[id];
            double d = dist2d(x, y, px, py);
            if (d < 1e-12) { x = px; y = py; coincide = true; break; }
            double inv = wi / d;
            numx += inv * px; numy += inv * py; denom += inv;
        }
        if (!coincide) {
            if (denom < 1e-18) break;
            double nx = numx / denom, ny = numy / denom;
            if (dist2d(x,y,nx,ny) < EPS) { x = nx; y = ny; break; }
            x = nx; y = ny;
        }
    }

    // 2) 가장 가까운 정수 격자 스냅
    int bx = (int)llround(x);
    int by = (int)llround(y);

    // 3) 주변 1-링(±1)에서 최적 정수 격자점 찾기
    double bestCost = numeric_limits<double>::infinity();
    Point best(bx, by, 1.0);
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            int cx = bx + dx, cy = by + dy;
            double c = 0.0;
            for (int id : idxs) {
                c += w[id] * dist2d(R[id].x, R[id].y, cx, cy);
            }
            if (c < bestCost) {
                bestCost = c;
                best.x = cx; best.y = cy;
            }
        }
    }
    return best; // 정수 격자 센터
}

struct KMedResult {
    vector<Point> centers;  // 정수 격자 센터들
    double gamma_cost;      // 대표점(R) 기준 가중 비용
};

// 정수 격자 센터를 반환하는 k-median 스텁
static KMedResult alg_eps_kmed_stub_discrete(const vector<Point>& R, int k, double eps) {
    int m = (int)R.size();
    if (m == 0 || k <= 0) return { {}, 0.0 };

    vector<double> w(m, 1.0);
    for (int i=0;i<m;++i) w[i] = max(1.0, R[i].w);

    // k >= m 이면 대표점 그 자체를 센터로 사용
    if (k >= m) {
        vector<Point> centers;
        centers.reserve(m);
        for (int i=0;i<m;++i) centers.push_back(Point(R[i].x, R[i].y, 1.0));
        double cost = 0.0;
        for (int i=0;i<m;++i) {
            double best = 1e100;
            for (auto &c: centers) {
                best = min(best, w[i]*dist2d(R[i].x, R[i].y, c.x, c.y));
            }
            cost += best;
        }
        return { centers, cost };
    }

    vector<Point> centers; centers.reserve(k);

    // 초기화: 가중 평균(연속) → 정수 스냅
    double sx=0, sy=0, sw=0;
    for (int i=0;i<m;++i){ sx += w[i]*R[i].x; sy += w[i]*R[i].y; sw += w[i]; }
    int cx0 = (int)llround(sx / max(sw, 1e-18));
    int cy0 = (int)llround(sy / max(sw, 1e-18));
    centers.push_back(Point(cx0, cy0, 1.0));

    // farthest-first (가중): 정수 좌표 대표점 중 가장 멀리 떨어진 것부터 선택
    for (int cidx=1;cidx<k;++cidx) {
        int bestIdx = -1; double bestScore = -1.0;
        for (int i=0;i<m;++i) {
            double dmin = 1e100;
            for (auto &c: centers) dmin = min(dmin, dist2d(R[i].x, R[i].y, c.x, c.y));
            double score = w[i]*dmin;
            if (score > bestScore) { bestScore = score; bestIdx = i; }
        }
        centers.push_back(Point(R[bestIdx].x, R[bestIdx].y, 1.0));
    }

    // Lloyd-style 반복: 할당 → 정수 지오메트릭 미디안 업데이트
    const int ITER = 10;
    vector<int> assign(m, 0);
    for (int it=0; it<ITER; ++it) {
        // 할당
        for (int i=0;i<m;++i) {
            double best = 1e100; int bestj=0;
            for (int j=0;j<k;++j) {
                double d = dist2d(R[i].x, R[i].y, centers[j].x, centers[j].y);
                if (d < best) { best = d; bestj = j; }
            }
            assign[i] = bestj;
        }
        // 업데이트 (클러스터별 정수 지오메트릭 미디안)
        for (int j=0;j<k;++j) {
            vector<int> idxs;
            for (int i=0;i<m;++i) if (assign[i]==j) idxs.push_back(i);
            if (idxs.empty()) continue;
            Point init = centers[j];
            centers[j] = discrete_geometric_median_snap(R, idxs, w, init, 50);
        }
    }

    // 대표점 비용 Γ 계산
    double cost = 0.0;
    for (int i=0;i<m;++i) {
        double best = 1e100;
        for (int j=0;j<k;++j)
            best = min(best, w[i]*dist2d(R[i].x, R[i].y, centers[j].x, centers[j].y));
        cost += best;
    }
    return { centers, cost };
}

/* ======================
   쿼드트리 기반 Sublinear k-median (Appendix A)
   ====================== */

struct Cell {
    int x1, y1;   // inclusive (좌하단)
    int side;     // side length = 2^i
    int level;    // i
};

struct RunResult {
    bool stopped = false;
    vector<Point> reps;   // R_j (대표점; 정수 좌표 + 가중치)
    vector<Point> centers;// k개 센터(정수 좌표)
    double Gamma = 1e100; // 대표집합 비용
};

// 튜닝 가능한 상수(작게 잡으면 테스트가 쉬움)
static constexpr double DELTA_KMED_SCALE = 16.0; // 증명 상수 2^20 대신 테스트용 스케일
static constexpr int    SPARSE_LIMIT_SCALE = 64;

class SublinearKMedian {
public:
    SublinearKMedian(const vector<Point>& points, int _k, double _eps)
        : P(points), tree(points), n((int)points.size()), k(_k), eps(_eps)
    {
        // delta_kmed = (상수배) * (k * log n) / eps^3
        double ln_n = max(1.0, log(max(2, n)));
        delta_kmed = DELTA_KMED_SCALE * ((double)k * ln_n) / pow(eps, 3.0);

        // 도메인 정사각형 잡기 (입력 바운딩 박스 기준)
        int minX = tree.getMinX(), maxX = tree.getMaxX();
        int minY = tree.getMinY(), maxY = tree.getMaxY();
        int widthX = max(1, maxX - minX + 1);
        int widthY = max(1, maxY - minY + 1);
        int needSide = max({ 2*n, widthX, widthY });

        i_max = 0; S = 1;
        while (S < needSide) { S <<= 1; i_max++; }
        X0 = minX; Y0 = minY;

        // 조기중단 한계
        sparse_limit = (int)ceil(SPARSE_LIMIT_SCALE * (double)k * ln_n / pow(eps, 3.0));
        if (sparse_limit < 1) sparse_limit = 1;
    }

    pair<vector<Point>, double> solve() {
        // r_j = (1+eps)^j,  j = 0..t,   t = ceil(log_{1+eps}(4 n^2))
        double U = max(1, 4*n*n);
        int t = (int)ceil( log(U) / log(1.0 + eps) );

        vector<RunResult> results; results.reserve(t+1);
        for (int j=0; j<=t; ++j) {
            double rj = pow(1.0 + eps, j);
            results.push_back( run_one_guess(rj) );
        }

        // 선택 규칙: 가장 작은 j* s.t. r_j <= Gamma_j < (1+eps)*r_{j+1}
        vector<Point> best_centers;
        double best_gamma = 1e100;
        int best_j = -1;

        for (int j=0; j<=t; ++j) {
            if (results[j].stopped) continue;
            double rj = pow(1.0 + eps, j);
            double rj1 = pow(1.0 + eps, j+1);
            double G = results[j].Gamma;
            if (rj <= G && G < (1.0 + eps) * rj1) {
                best_centers = results[j].centers;
                best_gamma = G;
                best_j = j;
                break;
            }
        }
        if (best_j < 0) {
            // 백업: Gamma 최소 run
            for (int j=0; j<=t; ++j) {
                if (results[j].stopped) continue;
                if (results[j].Gamma < best_gamma) {
                    best_gamma = results[j].Gamma;
                    best_centers = results[j].centers;
                    best_j = j;
                }
            }
        }
        return { best_centers, best_gamma };
    }

    vector<Point> get_reps(double rj) {
        RunResult rr = run_one_guess(rj);
        return rr.reps;
    }


private:
    const vector<Point>& P;
    RangeTree tree;
    int n, k;
    double eps;

    double delta_kmed;
    int i_max, S;
    int X0, Y0;
    int sparse_limit;

    int rangeCountCell(const Cell& c) const {
        int x2 = c.x1 + c.side - 1;
        int y2 = c.y1 + c.side - 1;
        return tree.range_count(c.x1, x2, c.y1, y2);
    }

    RunResult run_one_guess(double rj) {
        RunResult rr;
        vector<Cell> stack;
        stack.push_back(Cell{X0, Y0, S, i_max});
        vector<Point> reps;
        int sparse_cnt = 0;

        while (!stack.empty()) {
            Cell cur = stack.back(); stack.pop_back();
            int nc = rangeCountCell(cur);
            if (nc == 0) continue;

            double thresh = delta_kmed * (rj / (double)(1<<cur.level)); // δ_kmed * rj / 2^i
            bool dense = (double)nc >= thresh;

            if (dense && cur.level > 0) {
                int half = cur.side / 2;
                stack.push_back(Cell{cur.x1,           cur.y1,           half, cur.level - 1});
                stack.push_back(Cell{cur.x1 + half,    cur.y1,           half, cur.level - 1});
                stack.push_back(Cell{cur.x1,           cur.y1 + half,    half, cur.level - 1});
                stack.push_back(Cell{cur.x1 + half,    cur.y1 + half,    half, cur.level - 1});
            } else {
                // sparse (또는 더 못 쪼갬) → 대표점
                int cx = cur.x1 + cur.side/2;
                int cy = cur.y1 + cur.side/2;
                reps.push_back(Point(cx, cy, (double)nc));
                sparse_cnt++;
                if (sparse_cnt > sparse_limit) {
                    rr.stopped = true; // 과소 추측: 조기 중단
                    return rr;
                }
            }
        }

        rr.reps = move(reps);
        // 블랙박스 대체 (정수 센터 k개)
        auto km = alg_eps_kmed_stub_discrete(rr.reps, k, eps);
        rr.centers = km.centers;
        rr.Gamma = km.gamma_cost;
        return rr;
    }
};

/* ======================
   데모 main
   ====================== */
// int main() {
//     ios::sync_with_stdio(false);
//     cin.tie(nullptr);

//     // 간단한 테스트 데이터 (3개 군집)
//     vector<Point> points;
//     for (int i=0;i<20;++i) points.push_back(Point(10 + (i%3), 10 + (i%3)));
//     for (int i=0;i<20;++i) points.push_back(Point(60 + (i%4), 15 + (i%2)));
//     for (int i=0;i<20;++i) points.push_back(Point(25 + (i%5), 70 + (i%3)));

//     int k = 3;
//     double eps = 0.2;

//     SublinearKMedian solver(points, k, eps);
//     auto [centers, gamma] = solver.solve();

//     cout << fixed << setprecision(4);
//     cout << "[Returned centers]\n";
//     for (size_t i=0;i<centers.size(); ++i) {
//         cout << "  C[" << i << "] = (" << centers[i].x << ", " << centers[i].y << ")\n";
//     }
//     cout << "Gamma (cost on representatives): " << gamma << "\n";

//     // 원본 P에서 비용 평가(단위 가중)
//     auto cost_on_P = [&](const vector<Point>& cs){
//         double cost = 0.0;
//         for (auto &p : points) {
//             double best = 1e100;
//             for (auto &c : cs) best = min(best, dist2d(p.x, p.y, c.x, c.y));
//             cost += best;
//         }
//         return cost;
//     };
//     cout << "Cost on original P (unit weights): " << cost_on_P(centers) << "\n";
//     return 0;
// }
