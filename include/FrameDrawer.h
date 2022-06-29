/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include "Tracking.h"
#include "MapPoint.h"
#include "Map.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include<mutex>


namespace ORB_SLAM2
{

    class Tracking;
    class Viewer;

    class FrameDrawer
    {
    public:
        FrameDrawer(Map* pMap);

        // Update info from the last processed frame.
        void Update(Tracking *pTracker);

        // Draw last processed frame.
        cv::Mat DrawFrame();
        cv::Mat DrawSegImg();

    protected:

        void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

        // Info of the frame to be drawn
        cv::Mat mIm;
        cv::Mat mSeg;
        int N;
        std::vector<cv::KeyPoint> mvCurrentKeys;     //当前帧的特征点
        std::vector<bool> mvbMap, mvbVO;
        bool mbOnlyTracking;
        int mnTracked, mnTrackedVO;
        std::vector<cv::KeyPoint> mvIniKeys; //初始化时的特征点
        std::vector<int> mvIniMatches;   //跟踪初始化时，前两帧的特征点匹配
        int mState; //跟踪状态

        // 自己添加的
        int NL;
        std::vector<cv::line_descriptor::KeyLine> mvCurrentKeyLines;
        std::vector<bool> mvbLineMap, mvbLineVO;
        std::vector<cv::line_descriptor::KeyLine> mvIniKeyLines;  //初始化时的特征线

        Map* mpMap;

        std::mutex mMutex;
    };

} //namespace ORB_SLAM

#endif // FRAMEDRAWER_H