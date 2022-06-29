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

#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include"Map.h"
#include"MapPoint.h"
#include"KeyFrame.h"
#include<pangolin/pangolin.h>
#include "open3d/geometry/PointCloud.h"
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>

#include<mutex>

namespace ORB_SLAM2
{

class MapDrawer
{
public:
    MapDrawer(Map* pMap, const string &strSettingPath);

    Map* mpMap;

    void DrawMapPoints();
    void DrawMesh();
    void TextureDenseMap();
    void DrawMapLines();
    void DrawMapPlanes();
    void DrawPlanePolygon();
    void DrawDiscretePoints();
    void DrawOuterPLanes();
    void DrawCubePoints();
    void DrawCubeBoundingBox();
    void DrawNoPlaneArea();
    void DrawInlierLines();
    void DrawBoundaryPoints();
    void DrawPlaneIntersections();
    void DrawNonPlaneArea();
    void DrawLayouts();
    void SaveMeshMode(const string& filename);
    void DrawMapPlaneBoundaries();
    void DrawSurfels();
    void DrawCrossLine();
    void DrawCrossPoint();
    void DrawCrossPointInMap();
    void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph);
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
    void SetCurrentCameraPose(const cv::Mat &Tcw);
    void SetReferenceKeyFrame(KeyFrame *pKF);
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);

protected:
    std::shared_ptr<open3d::geometry::PointCloud> globalMap;
    uint16_t                lastKeyframeSize =0;

private:
    float mLineWidth;    //线特征的尺寸
    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;
    float DepthMapFactor;

    cv::Mat mCameraPose;

    std::mutex mMutexCamera;
};

} //namespace ORB_SLAM

#endif // MAPDRAWER_H
