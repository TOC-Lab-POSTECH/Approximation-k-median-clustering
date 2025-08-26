// Created by Chanho Song

struct Point {
    int x, y;
    double w; // weight (for coreset / representative points)
    Point() : x(0), y(0), w(1.0) {}
    Point(int _x, int _y) : x(_x), y(_y), w(1.0) {}
    Point(int _x, int _y, double _w) : x(_x), y(_y), w(_w) {}
};
