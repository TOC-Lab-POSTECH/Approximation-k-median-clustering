// Created by Sanghwa Han
#ifndef RANGE_COUNTING_ORACLE_H
#define RANGE_COUNTING_ORACLE_H
#include <iostream>
#include <vector>
#include <algorithm>
#include "point.h"
using namespace std;

struct RangeTreeNode {
    int x_left, x_right; // x range
    vector<int> ys;      // sorted y-coordinates of all points under this node
    RangeTreeNode *left = nullptr, *right = nullptr;
    RangeTreeNode() = default;
};

class RangeTree {
public:
    RangeTree(const vector<Point>& points) {
        vector<Point> sorted_points = points;
        sort(sorted_points.begin(), sorted_points.end(),
             [](const Point& a, const Point& b) { return a.x < b.x; });
        root = build(sorted_points, 0, (int)sorted_points.size() - 1);
        // store bounds
        minX =  INT_MAX; maxX = INT_MIN;
        minY =  INT_MAX; maxY = INT_MIN;
        for (auto &p : points) {
            minX = min(minX, p.x); maxX = max(maxX, p.x);
            minY = min(minY, p.y); maxY = max(maxY, p.y);
        }
        if (minX == INT_MAX) { minX = minY = 0; maxX = maxY = 0; }
    }

    // number of points in [x1, x2] x [y1, y2]
    int range_count(int x1, int x2, int y1, int y2) const {
        return query(root, x1, x2, y1, y2);
    }

    int getMinX() const { return minX; }
    int getMaxX() const { return maxX; }
    int getMinY() const { return minY; }
    int getMaxY() const { return maxY; }

private:
    RangeTreeNode* root = nullptr;
    int minX, maxX, minY, maxY;

    RangeTreeNode* build(const vector<Point>& pts, int l, int r) {
        if (l > r) return nullptr;
        RangeTreeNode* node = new RangeTreeNode();
        node->x_left = pts[l].x;
        node->x_right = pts[r].x;
        node->ys.reserve(r - l + 1);
        for (int i = l; i <= r; ++i) node->ys.push_back(pts[i].y);
        sort(node->ys.begin(), node->ys.end());
        if (l == r) return node;
        int m = (l + r) / 2;
        node->left = build(pts, l, m);
        node->right = build(pts, m + 1, r);
        return node;
    }

    int query(RangeTreeNode* node, int x1, int x2, int y1, int y2) const {
        if (!node || node->x_right < x1 || node->x_left > x2) return 0;
        if (x1 <= node->x_left && node->x_right <= x2) {
            auto hi = upper_bound(node->ys.begin(), node->ys.end(), y2);
            auto lo = lower_bound(node->ys.begin(), node->ys.end(), y1);
            return (int)(hi - lo);
        }
        return query(node->left, x1, x2, y1, y2)
             + query(node->right, x1, x2, y1, y2);
    }
};
#endif