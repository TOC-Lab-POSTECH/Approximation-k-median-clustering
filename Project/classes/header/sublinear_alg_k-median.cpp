#include "point.h"
#include "RangeCountingOracle.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>     // sqrt, log, pow, ceil, round
#include <iomanip>   // fixed, setprecision

using namespace std;

struct Center { double x, y; };
static inline double dist2d(double ax, double ay, double bx, double by) {
    double dx = ax - bx, dy = ay - by;
    return sqrt(dx*dx + dy*dy);
}

/* ======================
   Alg_ε k-med : TEST STUB
   - 가중 k-median 대용: Lloyd-style + 가중 Weiszfeld (지오메트릭 미디안)
   - 실제 [21] 알고리즘이 아님 (테스트/데모용)
   ====================== */

static Center weighted_geometric_median(const vector<Point>& pts,
                                        const vector<int>& idxs,
                                        const vector<double>& w,
                                        Center init,
                                        int max_iter=50) {
    double EPS = 1e-7;
    double x = init.x, y = init.y;
    for (int it = 0; it < max_iter; ++it) {
        double numx = 0.0, numy = 0.0, denom = 0.0;
        bool coincide = false;
        for (int id : idxs) {
            double px = pts[id].x, py = pts[id].y;
            double wi = w[id];
            double d = dist2d(x, y, px, py);
            if (d < 1e-12) { x = px; y = py; coincide = true; break; }
            double inv = wi / d;
            numx += inv * px; numy += inv * py; denom += inv;
        }
        if (!coincide) {
            double nx = numx / max(denom, 1e-18);
            double ny = numy / max(denom, 1e-18);
            if (dist2d(x,y,nx,ny) < EPS) { x = nx; y = ny; break; }
            x = nx; y = ny;
        }
    }
    return {x,y};
}

struct KMedResult {
    vector<Center> centers;
    double gamma_cost;
};

static KMedResult alg_eps_kmed_stub(const vector<Point>& R, int k, double eps) {
    // 간단한 가중 k-median 대체 구현
    int m = (int)R.size();
    vector<double> w(m, 1.0);
    for (int i=0;i<m;++i) w[i] = max(1.0, R[i].w);

    if (m == 0 || k == 0) return { {}, 0.0 };
    vector<Center> centers;
    centers.reserve(k);

    if (k >= m) {
        for (int i=0;i<m;++i) centers.push_back({(double)R[i].x, (double)R[i].y});
        double cost = 0.0;
        for (int i=0;i<m;++i) {
            double best = 1e100;
            for (auto &c: centers) best = min(best, w[i]*dist2d(R[i].x, R[i].y, c.x, c.y));
            cost += best;
        }
        return { centers, cost };
    }

    // 초기화: 가중 중심 + farthest-first
    double sx=0, sy=0, sw=0;
    for (int i=0;i<m;++i){ sx += w[i]*R[i].x; sy += w[i]*R[i].y; sw += w[i]; }
    centers.push_back({sx/max(sw,1e-18), sy/max(sw,1e-18)});
    for (int cidx=1;cidx<k;++cidx) {
        int bestIdx = -1; double bestScore = -1.0;
        for (int i=0;i<m;++i) {
            double dmin = 1e100;
            for (auto &c: centers) dmin = min(dmin, dist2d(R[i].x, R[i].y, c.x, c.y));
            double score = w[i]*dmin;
            if (score > bestScore) { bestScore = score; bestIdx = i; }
        }
        centers.push_back({(double)R[bestIdx].x, (double)R[bestIdx].y});
    }

    // Lloyd-style + Weiszfeld
    int ITER = 10;
    vector<int> assign(m, 0);
    for (int it=0; it<ITER; ++it) {
        // assignment
        for (int i=0;i<m;++i) {
            double best = 1e100; int bestj=0;
            for (int j=0;j<k;++j) {
                double d = dist2d(R[i].x, R[i].y, centers[j].x, centers[j].y);
                if (d < best) { best = d; bestj = j; }
            }
            assign[i] = bestj;
        }
        // update
        for (int j=0;j<k;++j) {
            vector<int> idxs;
            for (int i=0;i<m;++i) if (assign[i]==j) idxs.push_back(i);
            if (idxs.empty()) continue;
            Center init = centers[j];
            centers[j] = weighted_geometric_median(R, idxs, w, init, 50);
        }
    }

    double cost = 0.0;
    for (int i=0;i<m;++i) {
        double best = 1e100;
        for (int j=0;j<k;++j) best = min(best, dist2d(R[i].x, R[i].y, centers[j].x, centers[j].y));
        cost += w[i]*best;
    }
    return { centers, cost };
}

/* ======================
   쿼드트리 기반 Sublinear k-median (Appendix A)
   ====================== */

struct Cell {
    int x1, y1;   // inclusive
    int side;     // side length = 2^i
    int level;    // i
};

struct RunResult {
    bool stopped = false;
    vector<Point> reps;       // 대표점 집합 R_j (가중)
    vector<Center> centers;   // Alg_ε k-med 결과 중심들
    double Gamma = 1e100;     // 대표점 비용
};

class SublinearKMedian {
public:
    SublinearKMedian(const vector<Point>& points, int _k, double _eps)
        : P(points), tree(points), n((int)points.size()), k(_k), eps(_eps)
    {
        // delta_kmed = 2^20 * (k log n) / eps^3
        double ln_n = max(1.0, log(max(2, n)));
        delta_kmed = (double)(1<<20) * ( (double)k * ln_n ) / pow(eps, 3.0);

        // 도메인: 입력 바운딩 박스에서 시작, 한 변을 2의 거듭제곱으로 키워 정사각형으로
        int minX = tree.getMinX(), maxX = tree.getMaxX();
        int minY = tree.getMinY(), maxY = tree.getMaxY();
        int widthX = max(1, maxX - minX + 1);
        int widthY = max(1, maxY - minY + 1);
        int needSide = max({ 2*n, widthX, widthY });

        i_max = 0; S = 1;
        while (S < needSide) { S <<= 1; i_max++; }
        X0 = minX; Y0 = minY;

        // 조기중단 임계치: O(k log n / eps^3) (상수 팩터는 다소 여유있게)
        int C = 16;
        sparse_limit = (int)ceil(C * (double)k * ln_n / pow(eps, 3.0));
        if (sparse_limit < 1) sparse_limit = 1;
    }

    pair<vector<Center>, double> solve() {
        // r_j = (1+eps)^j, j = 0..t,  t = ceil(log_{1+eps}(4 n^2))
        double U = max(1, 4*n*n);
        int t = (int)ceil( log(U) / log(1.0 + eps) );

        vector<RunResult> results; results.reserve(t+1);
        for (int j=0; j<=t; ++j) {
            double rj = pow(1.0 + eps, j);
            results.push_back( run_one_guess(rj) );
        }

        // 선택 규칙: 가장 작은 j* s.t. r_j <= Gamma_j < (1+eps) * r_{j+1}
        vector<Center> best_centers;
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
            // 이론상 드문 경우: 대안으로 최소 Gamma 선택
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

private:
    const vector<Point>& P;
    RangeTree tree;
    int n, k;
    double eps;

    // 파라미터
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

            double thresh = delta_kmed * (rj / (double)(1<<cur.level)); // delta_kmed * rj / 2^i
            bool dense = (double)nc >= thresh;

            if (dense && cur.level > 0) {
                int half = cur.side / 2;
                stack.push_back(Cell{cur.x1,           cur.y1,           half, cur.level - 1});
                stack.push_back(Cell{cur.x1 + half,    cur.y1,           half, cur.level - 1});
                stack.push_back(Cell{cur.x1,           cur.y1 + half,    half, cur.level - 1});
                stack.push_back(Cell{cur.x1 + half,    cur.y1 + half,    half, cur.level - 1});
            } else {
                // sparse (또는 더 쪼갤 수 없음)
                double cx = cur.x1 + cur.side / 2.0;
                double cy = cur.y1 + cur.side / 2.0;
                reps.push_back(Point((int)round(cx), (int)round(cy), (double)nc));
                sparse_cnt++;
                if (sparse_cnt > sparse_limit) {
                    rr.stopped = true; // 과소 추측 → 조기 중단
                    return rr;
                }
            }
        }

        rr.reps = move(reps);
        // 블랙박스 (테스트 스텁)
        auto km = alg_eps_kmed_stub(rr.reps, k, eps);
        rr.centers = km.centers;
        rr.Gamma = km.gamma_cost;
        return rr;
    }
};

/* ======================
   데모 main
   ====================== */
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // 간단한 테스트 데이터 (3개 군집)
    vector<Point> points;
    for (int i=0;i<20;++i) points.push_back(Point(10 + (i%3), 10 + (i%3)));
    for (int i=0;i<20;++i) points.push_back(Point(60 + (i%4), 15 + (i%2)));
    for (int i=0;i<20;++i) points.push_back(Point(25 + (i%5), 70 + (i%3)));

    int k = 3;
    double eps = 0.2;

    SublinearKMedian solver(points, k, eps);
    auto [centers, gamma] = solver.solve();

    cout << fixed << setprecision(4);
    cout << "[Returned centers]\n";
    for (size_t i=0;i<centers.size(); ++i) {
        cout << "  C[" << i << "] = (" << centers[i].x << ", " << centers[i].y << ")\n";
    }
    cout << "Gamma (cost on representatives): " << gamma << "\n";

    // 원본 P에서 비용 평가(테스트용)
    auto cost_on_P = [&](const vector<Center>& cs){
        double cost = 0.0;
        for (auto &p : points) {
            double best = 1e100;
            for (auto &c : cs) best = min(best, dist2d(p.x, p.y, c.x, c.y));
            cost += best;
        }
        return cost;
    };
    cout << "Cost on original P (unit weights): " << cost_on_P(centers) << "\n";
    return 0;
}


