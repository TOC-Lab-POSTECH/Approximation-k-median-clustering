#ifndef REPRESENTATIVE_SET_BUILDER_H
#define REPRESENTATIVE_SET_BUILDER_H

#include <vector>
#include <cmath>
#include "RangeCountingOracle.h" // RangeTree, Point 정의 포함

using namespace std;

// 쿼드트리 셀 구조체
struct QuadCell {
    int x0, y0;   // 셀의 왼쪽 아래 모서리 (inclusive)
    int size;     // 한 변의 길이 (2^level)
    int level;    // 그리드 레벨 i (size = 2^i)
};

// R_j 빌더 설정값
struct RjConfig {
    int n;             // 점 개수 |P|
    int k;             // k-median의 k
    double eps;        // ε
    double rj;         // OPT_k-med의 guess 값 r_j = (1+eps)^j
};

// R_j를 RangeTree 기반으로 만드는 클래스
class RepresentativeSetBuilder {
public:
    RepresentativeSetBuilder(const RangeTree& tree, const RjConfig& cfg)
        : tree(tree), cfg(cfg)
    {
        // 입력 점들의 bounding box
        minX = tree.getMinX();
        maxX = tree.getMaxX();
        minY = tree.getMinY();
        maxY = tree.getMaxY();

        /*bounding box를 포함하는 최소의 2의 거듭제곱 길이(논문에서는 domain이 [2𝑛]^2이지만, 
        여기서는 실제 점들의 bounding box에 딱 맞는 정사각형을 잡고, 그 정사각형의 한 변을 2의 거듭제곱으로 올려주는 방식)*/ 
        int widthX = maxX - minX + 1;
        int widthY = maxY - minY + 1;
        int rootSize = 1;
        while (rootSize < widthX || rootSize < widthY) {
            rootSize <<= 1;
        }

        rootCell.x0 = minX;
        rootCell.y0 = minY;
        rootCell.size = rootSize;
        rootCell.level = 0;
        while ((1 << rootCell.level) < rootSize) {
            rootCell.level++;
        }
        // 이 상태에서 size = 2^(rootCell.level) = rootSize

        // δ_k-med = 2^20 * (k log n) / ε^3
        double logn = log2((double)cfg.n); // 밑 2 로그 (밑은 상수 차이만 나므로 크게 중요 X)
        // delta_kmed = (double)(1 << 20) * (cfg.k * logn)
        //              / (cfg.eps * cfg.eps * cfg.eps);

        delta_kmed = 100000 * (cfg.k * logn)
                     / (cfg.eps * cfg.eps * cfg.eps);
        // |K_j| 제한: 대략 O(k ε^-3 log n) 스케일
        Kj_limit = (int)(4.0 * cfg.k * logn
                  / (cfg.eps * cfg.eps * cfg.eps)) + 10;
    }

    // R_j를 생성한다.
    // - 성공적으로 run(j)가 유지되면 (true, R_j)
    // - |K_j|가 너무 커져서 run(j)를 포기하면 (false, 빈 벡터)
    pair<bool, vector<Point>> build() {
        Kj.clear();
        aborted = false;

        processCell(rootCell);

        if (aborted) {
            return {false, {}};
        }

        // sparse cell 집합 K_j → representative point set R_j
        vector<Point> Rj;
        Rj.reserve(Kj.size());

        for (const QuadCell& c : Kj) {
            int cnt = countPointsInCell(c);
            if (cnt == 0) continue;

            // 셀 중심 좌표 (정수 좌표가 아니어도 되지만, 여기선 double → int로 반올림)
            double cx = (double)c.x0 + (double)c.size / 2.0;
            double cy = (double)c.y0 + (double)c.size / 2.0;
            Point rep((int)round(cx), (int)round(cy), (double)cnt); // w = cnt
            Rj.push_back(rep);
        }

        return {true, Rj};
    }

private:
    const RangeTree& tree;
    RjConfig cfg;

    QuadCell rootCell;
    double delta_kmed;
    int Kj_limit;
    bool aborted = false;

    int minX, maxX, minY, maxY;

    vector<QuadCell> Kj; // sparse cell들의 집합 K_j

    // 셀 c 안의 점 개수 n_c = |P ∩ c|
    int countPointsInCell(const QuadCell& c) const {
        int x1 = c.x0;
        int x2 = c.x0 + c.size - 1; // inclusive
        int y1 = c.y0;
        int y2 = c.y0 + c.size - 1;
        return tree.range_count(x1, x2, y1, y2); //RangeCountingOracle.h의 range_count 함수 사용
    }

    //dense / sparse 판정 및 재귀적 셀 분할
    void processCell(const QuadCell& c) {
        if (aborted) return;

        int n_c = countPointsInCell(c);
        if (n_c == 0) return; // 비어 있는 셀은 무시

        // threshold T_{i,j} = δ_k-med * r_j / 2^i
        // 여기서 size = 2^i (quadtree 한 변 길이)
        double threshold = delta_kmed * cfg.rj / (double)c.size;

        // leaf 조건: size == 1이면 더 이상 쪼갤 수 없으므로 무조건 sparse 취급
        bool isLeaf = (c.size == 1);
        bool isSparse = (n_c < threshold) || isLeaf;

        if (isSparse) {
            Kj.push_back(c);
            if ((int)Kj.size() > Kj_limit) {
                // 이 guess r_j는 실제 OPT보다 너무 작다고 보고 run(j) 포기
                aborted = true;
            }
            return;
        }

        // dense 셀 → 4개의 child로 분할 후 재귀
        int half = c.size / 2;
        if (half == 0) {
            // 이론상 size는 항상 2^level이므로 half==0은 발생하지 않지만,
            // 혹시라도 방어적으로 sparse로 처리
            Kj.push_back(c);
            return;
        }

        QuadCell c1{c.x0,         c.y0,         half, c.level - 1}; // 좌하
        QuadCell c2{c.x0,         c.y0 + half,  half, c.level - 1}; // 좌상
        QuadCell c3{c.x0 + half,  c.y0,         half, c.level - 1}; // 우하
        QuadCell c4{c.x0 + half,  c.y0 + half,  half, c.level - 1}; // 우상

        processCell(c1);
        processCell(c2);
        processCell(c3);
        processCell(c4);
    }
};

#endif // REPRESENTATIVE_SET_BUILDER_H
