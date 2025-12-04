#ifndef REPRESENTATIVE_SET_BUILDER_H
#define REPRESENTATIVE_SET_BUILDER_H

#include <vector>
#include <cmath>
#include "RangeCountingOracle.h" // includes RangeTree and Point

using namespace std;

// Quadtree cell
struct QuadCell {
    int x0, y0;   // bottom-left corner (inclusive)
    int size;     // side length (2^level)
    int level;    // grid level i (size = 2^i)
};

// Configuration for building R_j
struct RjConfig {
    int n;             // number of points |P|
    int k;             // k in k-median
    double eps;        // ε parameter
    double rj;         // guess for OPT_k-med: r_j = (1+eps)^j
};

// Class that builds R_j using a RangeTree (RCO)
class RepresentativeSetBuilder {
public:
    RepresentativeSetBuilder(const RangeTree& tree, const RjConfig& cfg);

    // Build R_j.
    // - If run(j) finishes without aborting: return (true, R_j).
    // - If |K_j| becomes too large: abort run(j) and return (false, empty).
    pair<bool, vector<Point>> build();

private:
    const RangeTree& tree;
    RjConfig cfg;

    QuadCell rootCell;
    double delta_kmed;
    int Kj_limit;
    bool aborted = false;

    int minX, maxX, minY, maxY;

    vector<QuadCell> Kj; // set of sparse cells K_j

    // Number of points in cell c, n_c = |P ∩ c|
    int countPointsInCell(const QuadCell& c) const;

    // Dense/sparse classification and recursive subdivision
    void processCell(const QuadCell& c);
};

#endif // REPRESENTATIVE_SET_BUILDER_H
