#ifndef K_MEDIAN_PP_H
#define K_MEDIAN_PP_H

#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include <algorithm>
#include "point.h"  // struct Point { int x, y; double w; ... }

using namespace std;

// ---------------------------
// 유틸리티: 거리 및 비용 계산
// ---------------------------

// 유클리드 거리 (L2)
inline double euclidean_distance(const Point& a, const Point& b) {
    double dx = (double)a.x - (double)b.x;
    double dy = (double)a.y - (double)b.y;
    return sqrt(dx * dx + dy * dy);
}

// weighted k-median cost: sum_{p} w(p) * dist(p, C)
inline double k_median_cost(const vector<Point>& pts,
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
// 1-median (geometric median) 근사: Weiszfeld
// -------------------------------------------
// 한 cluster의 점들에 대해 weighted geometric median을 근사.
//   - 입력: clusterPoints (cluster에 속한 representative points)
//   - 초기 위치: (x0, y0) (대개 현재 center나 weighted mean)
//   - 출력: 근사된 geometric median (Point, weight는 합계로 줘도 되고 1.0으로 둬도 됨)
inline Point geometric_median_weiszfeld(const vector<Point>& clusterPoints,
                                        double x0, double y0,
                                        int maxIter = 100,
                                        double tol = 1e-3) {
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

            // center와 거의 같은 위치인 점이 있으면, 그 점 근처에서 멈추는 처리를 해줌
            if (dist < 1e-9) {
                // 그 점 근처로 그냥 스냅하고 끝내도 됨
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
// k-median++ 초기화 (Rj 위에서 동작)
// -------------------------------------------
// 입력: pts = R_j (weighted representative points)
// 출력: k개의 초기 center (Point 벡터, w는 1.0으로 두어도 됨)
inline vector<Point> k_median_pp_init(const vector<Point>& pts,
                                      int k,
                                      mt19937& rng) {
    vector<Point> centers;
    int n = (int)pts.size();
    if (n == 0 || k <= 0) return centers;

    // k가 점 개수보다 클 수 있으므로, 최대 n개까지만 선택
    k = min(k, n);

    // 1. 첫 center: weight 고려해서 랜덤 샘플링 (w에 비례해서)
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

    // 2. 나머지 centers: D(x)에 비례해서 샘플링 (k-median 스타일)
    while ((int)centers.size() < k) {
        vector<double> distWeights(n);
        double sum = 0.0;

        for (int i = 0; i < n; ++i) {
            const auto& p = pts[i];
            // p와 가장 가까운 기존 center와의 거리
            double best = numeric_limits<double>::infinity();
            for (const auto& c : centers) {
                double d = euclidean_distance(p, c);
                if (d < best) best = d;
            }
            // 샘플링 weight ~ w(p) * D(p)
            double val = p.w * best;
            distWeights[i] = val;
            sum += val;
        }

        if (sum <= 0.0) {
            // 모든 점이 같은 위치인 경우 등: 남은 center들은 아무 점이나 골라서 채우기
            // (중복 허용)
            while ((int)centers.size() < k) {
                uniform_int_distribution<int> ui(0, n - 1);
                centers.push_back(pts[ui(rng)]);
            }
            break;
        }

        // 누적 분포에서 하나 샘플링
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
// 전체 k-median++ 알고리즘 (R_j 위에서)
// -------------------------------------------
// 입력:
//   - pts: R_j (weighted representative points)
//   - k:   center 개수
//   - maxIters: Lloyd-style 반복 횟수
// 출력:
//   - 최종 center 집합 (vector<Point>, w는 1.0으로 표기)
//   - 필요하면 cost는 k_median_cost(pts, centers)로 따로 계산
inline vector<Point> k_median_pp(const vector<Point>& pts,
                                 int k,
                                 int maxIters = 100,
                                 int maxWeiszfeldIters = 50,
                                 double tol = 1e-3,
                                 uint32_t seed = 12345) {
    vector<Point> centers;
    int n = (int)pts.size();
    if (n == 0 || k <= 0) return centers;

    mt19937 rng(seed);

    // 1. k-median++ 초기화
    centers = k_median_pp_init(pts, k, rng);
    int csz = (int)centers.size();
    if (csz == 0) return centers;

    // 2. Lloyd-style 반복: 할당 → 1-median 업데이트
    vector<int> assignment(n, -1);

    for (int it = 0; it < maxIters; ++it) {
        // 2-1. 각 점을 가장 가까운 center에 할당
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

        // 변화가 거의 없으면 조기 종료 가능 (assignment 변동 없음)
        if (!changed && it > 0) break;

        // 2-2. 각 cluster에 대해 새로운 center = weighted geometric median 근사
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
                // 빈 cluster: 아무 점이나 랜덤하게 다시 center로 설정 (collapse 방지)
                uniform_int_distribution<int> ui(0, n - 1);
                centers[c] = pts[ui(rng)];
                centersMovedLittle = false;
                continue;
            }

            // 초기 위치: cluster의 weighted mean (평균) 사용
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

            centers[c] = newCenter; // w는 여기서는 의미 없으니 1.0으로 고정
        }

        if (centersMovedLittle) break;
    }

    return centers;
}

#endif // K_MEDIAN_PP_H
