//
// Created by nuc on 2021/8/19.
//
#include "Layout.h"

using namespace std;

int layoutCnt = 1;
Layout::Layout(const float &x, const float &y, const float &z)
{
    pos.resize(3);
    pos[0] = x;
    pos[1] = y;
    pos[2] = z;
    confidence = 0.0;
}

Layout::Layout(const float &x, const float &y, const float &z, const int &id1, const int &id2, const int &id3)
{
    pos.resize(3);
    pos[0] = x;
    pos[1] = y;
    pos[2] = z;
    confidence = 0.0;
    mvPlaneId.push_back(id1);
    mvPlaneId.push_back(id2);
    mvPlaneId.push_back(id3);
}
