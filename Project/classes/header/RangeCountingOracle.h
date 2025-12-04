// Created by Sanghwa Han
#ifndef RANGE_COUNTING_ORACLE_H
#define RANGE_COUNTING_ORACLE_H

#include <vector>
#include "point.h"

using namespace std;

// Node for 1D range tree on x,
// storing sorted y-coordinates for all points in this node.
struct RangeTreeNode {
    int x_left, x_right; // x range covered by this node
    vector<int> ys;      // sorted y-coordinates of points in this node
    RangeTreeNode* left = nullptr;
    RangeTreeNode* right = nullptr;
    RangeTreeNode() = default;
};

// Range counting oracle implemented as a range tree.
// Supports queries: count points in [x1, x2] × [y1, y2].
class RangeTree {
public:
    // Build tree from input points P.
    explicit RangeTree(const vector<Point>& pts);

    // Destructor: free all nodes.
    ~RangeTree();

    // Count points in axis-aligned rectangle [x1, x2] × [y1, y2].
    int range_count(int x1, int x2, int y1, int y2) const;

    // Accessors for bounding box of input points.
    int getMinX() const { return minX; }
    int getMaxX() const { return maxX; }
    int getMinY() const { return minY; }
    int getMaxY() const { return maxY; }

private:
    RangeTreeNode* root = nullptr;

    int minX = 0, maxX = 0;
    int minY = 0, maxY = 0;

    // Store a copy of points sorted by x for building.
    vector<Point> sortedPts;

    // Build tree recursively on [l, r) in sortedPts.
    RangeTreeNode* build(int l, int r);

    // Free nodes recursively.
    void destroy(RangeTreeNode* node);

    // Internal query on subtree rooted at node.
    int query(RangeTreeNode* node, int x1, int x2, int y1, int y2) const;
};

#endif // RANGE_COUNTING_ORACLE_H
