#include "Representative.h"

using namespace std;

RepresentativeSetBuilder::RepresentativeSetBuilder(const RangeTree& tree,
                                                   const RjConfig& cfg)
    : tree(tree), cfg(cfg)
{
    // Bounding box of the input points
    minX = tree.getMinX();
    maxX = tree.getMaxX();
    minY = tree.getMinY();
    maxY = tree.getMaxY();

    /*
     * Smallest power-of-two square that covers the bounding box.
     * In the paper the domain is [2n]^2, but here we use the actual
     * bounding box and round the side length up to the next power of two.
     */
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
    // Now size = 2^(rootCell.level) = rootSize

    // δ_k-med ≈ 2^20 * (k log n) / ε^3 (scaled down here)
    double logn = log2((double)cfg.n); // log base 2 (base does not matter up to constants)

    // Example scaled-down constant for experiments:
    delta_kmed = 100 * (cfg.k * logn)
                 / (cfg.eps * cfg.eps * cfg.eps);

    // Limit on |K_j|: on the order of O(k ε^-3 log n)
    Kj_limit = (int)(4.0 * cfg.k * logn
              / (cfg.eps * cfg.eps * cfg.eps)) + 10;
}

pair<bool, vector<Point>> RepresentativeSetBuilder::build() {
    Kj.clear();
    aborted = false;

    processCell(rootCell);

    if (aborted) {
        return {false, {}};
    }

    // Sparse cells K_j → representative set R_j
    vector<Point> Rj;
    Rj.reserve(Kj.size());

    for (const QuadCell& c : Kj) {
        int cnt = countPointsInCell(c);
        if (cnt == 0) continue;

        // Cell center (rounded to integer coordinates)
        double cx = (double)c.x0 + (double)c.size / 2.0;
        double cy = (double)c.y0 + (double)c.size / 2.0;
        Point rep((int)round(cx), (int)round(cy), (double)cnt); // w = cnt
        Rj.push_back(rep);
    }

    return {true, Rj};
}

// Number of points in cell c, n_c = |P ∩ c|
int RepresentativeSetBuilder::countPointsInCell(const QuadCell& c) const {
    int x1 = c.x0;
    int x2 = c.x0 + c.size - 1; // inclusive
    int y1 = c.y0;
    int y2 = c.y0 + c.size - 1;
    // Use RangeTree::range_count(x1, x2, y1, y2)
    return tree.range_count(x1, x2, y1, y2);
}

// Dense/sparse classification and recursive subdivision
void RepresentativeSetBuilder::processCell(const QuadCell& c) {
    if (aborted) return;

    int n_c = countPointsInCell(c);
    if (n_c == 0) return; // ignore empty cell

    // Threshold T_{i,j} = δ_k-med * r_j / 2^i
    // Here size = 2^i for this quadtree level.
    double threshold = delta_kmed * cfg.rj / (double)c.size;

    // Leaf condition: size == 1 cannot be split further → treat as sparse
    bool isLeaf = (c.size == 1);
    bool isSparse = (n_c < threshold) || isLeaf;

    if (isSparse) {
        Kj.push_back(c);
        if ((int)Kj.size() > Kj_limit) {
            // If |K_j| is too large, treat this r_j as too small and abort run(j)
            aborted = true;
        }
        return;
    }

    // Dense cell: split into 4 children and recurse
    int half = c.size / 2;
    if (half == 0) {
        // Theoretically size is always 2^level, so half==0 should not happen.
        // As a safeguard, mark this cell as sparse and stop splitting.
        Kj.push_back(c);
        return;
    }

    QuadCell c1{c.x0,         c.y0,         half, c.level - 1}; // bottom-left
    QuadCell c2{c.x0,         c.y0 + half,  half, c.level - 1}; // top-left
    QuadCell c3{c.x0 + half,  c.y0,         half, c.level - 1}; // bottom-right
    QuadCell c4{c.x0 + half,  c.y0 + half,  half, c.level - 1}; // top-right

    processCell(c1);
    processCell(c2);
    processCell(c3);
    processCell(c4);
}
