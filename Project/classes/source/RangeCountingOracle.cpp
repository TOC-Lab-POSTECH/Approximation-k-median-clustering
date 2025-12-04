#include "RangeCountingOracle.h"

#include <algorithm>
#include <limits>

using namespace std;

RangeTree::RangeTree(const vector<Point>& pts) {
    if (pts.empty()) {
        root = nullptr;
        minX = maxX = minY = maxY = 0;
        return;
    }

    // Compute bounding box of the input point set.
    minX = maxX = pts[0].x;
    minY = maxY = pts[0].y;
    for (const auto& p : pts) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
    }

    // Copy and sort points by x-coordinate.
    sortedPts = pts;
    sort(sortedPts.begin(), sortedPts.end(),
         [](const Point& a, const Point& b) {
             if (a.x != b.x) return a.x < b.x;
             return a.y < b.y;
         });

    // Build the tree on [0, n).
    root = build(0, (int)sortedPts.size());
}

RangeTree::~RangeTree() {
    destroy(root);
    root = nullptr;
}

// Recursively free all nodes.
void RangeTree::destroy(RangeTreeNode* node) {
    if (!node) return;
    destroy(node->left);
    destroy(node->right);
    delete node;
}

// Build a node that covers sortedPts[l..r-1].
RangeTreeNode* RangeTree::build(int l, int r) {
    if (l >= r) return nullptr;

    auto* node = new RangeTreeNode();

    // x-range for this node.
    node->x_left  = sortedPts[l].x;
    node->x_right = sortedPts[r - 1].x;

    // Collect all y-coordinates in this range and sort.
    node->ys.reserve(r - l);
    for (int i = l; i < r; ++i) {
        node->ys.push_back(sortedPts[i].y);
    }
    sort(node->ys.begin(), node->ys.end());

    // If only one point, this is a leaf.
    if (r - l == 1) {
        node->left = node->right = nullptr;
        return node;
    }

    // Split at the midpoint in index space.
    int mid = (l + r) / 2;
    node->left  = build(l, mid);
    node->right = build(mid, r);

    return node;
}

// Public query: count points in rectangle [x1, x2] × [y1, y2].
int RangeTree::range_count(int x1, int x2, int y1, int y2) const {
    if (!root) return 0;

    // Normalize ranges so that x1 <= x2 and y1 <= y2.
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);

    return query(root, x1, x2, y1, y2);
}

// Internal query on subtree rooted at node.
int RangeTree::query(RangeTreeNode* node,
                     int x1, int x2,
                     int y1, int y2) const {
    if (!node) return 0;

    // If node's x-range is completely outside query range, no contribution.
    if (node->x_right < x1 || node->x_left > x2) {
        return 0;
    }

    // If node's x-range is fully inside query range,
    // count y in [y1, y2] via binary search on sorted ys.
    if (x1 <= node->x_left && node->x_right <= x2) {
        auto hi = upper_bound(node->ys.begin(), node->ys.end(), y2);
        auto lo = lower_bound(node->ys.begin(), node->ys.end(), y1);
        return (int)(hi - lo);
    }

    // Otherwise, recurse into both children and sum.
    return query(node->left,  x1, x2, y1, y2)
         + query(node->right, x1, x2, y1, y2);
}
