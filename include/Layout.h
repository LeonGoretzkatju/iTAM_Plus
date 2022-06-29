//
// Created by nuc on 2021/8/19.
//

#ifndef ITAM_LAYOUT_H
#define ITAM_LAYOUT_H
#include <vector>
#include <unordered_set>

class Layout
{
public:
    Layout(const float &x, const float &y, const float &z);
    Layout(const float &x, const float &y, const float &z, const int &id1, const int &id2, const int &id3);

    std::vector<float> pos;
    float confidence;
    int mnId;
    std::vector<int> mvPlaneId;

    bool operator==(const Layout &node)
    {
        if (node.pos[0] == this->pos[0] && node.pos[1] == this->pos[1] && node.pos[2] == this->pos[2] && node.confidence == this->confidence)
            return true;
        else
            return false;
    }
};
#endif //ITAM_LAYOUT_H
