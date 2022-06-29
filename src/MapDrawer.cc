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

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>

#include "open3d/pipelines/integration/ScalableTSDFVolume.h"
#include "open3d/pipelines/integration/TSDFVolume.h"

#include "open3d/geometry/Image.h"
#include "open3d/geometry/RGBDImage.h"
#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/visualization/DrawGeometry.h"
#include "open3d/io/FilePLY.h"
#include <pcl/surface/texture_mapping.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>

#include <Eigen/Dense>

#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;
#define xjRandom(a,b) (rand()%(b-a)+a)
namespace ORB_SLAM2
{


    MapDrawer::MapDrawer(Map* pMap, const string &strSettingPath):mpMap(pMap)
    {
        globalMap = std::make_shared<open3d::geometry::PointCloud>();
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
        mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
        mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
        mPointSize = fSettings["Viewer.PointSize"];
        mCameraSize = fSettings["Viewer.CameraSize"];
        mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];
        mLineWidth = fSettings["Viewer.LineWidth"];
        DepthMapFactor = fSettings["DepthMapFactor"];
    }

    void MapDrawer::DrawMapPoints()
    {
        const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
        const vector<MapPoint*> &vpRefMPs = mpMap->GetReferenceMapPoints();

        set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

        if(vpMPs.empty())
            return;

        glPointSize(mPointSize);
        glBegin(GL_POINTS);
        glColor3f(0.0,0.0,0.0);     //黑色

        for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
        {
            if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
                continue;
            cv::Mat pos = vpMPs[i]->GetWorldPos();
//        cout << "point\t" << i << ": " << pos.at<float>(0) << ", " << pos.at<float>(1) << ", " << pos.at<float>(2) << endl;
            glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
        }
        glEnd();

        glPointSize(mPointSize);
        glBegin(GL_POINTS);
        glColor3f(0.0,0.0,0.0);     //红色

        for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
        {
            if((*sit)->isBad())
                continue;
            cv::Mat pos = (*sit)->GetWorldPos();
            glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
        }

        glEnd();
    }

    void MapDrawer::DrawBoundaryPoints() {
        const vector<pcl::PointXYZRGB> &vpMPs = mpMap->GetAllBoundaryPoints();

        if(vpMPs.empty())
            return;

        glPointSize(7);
        glBegin(GL_POINTS);
        glColor3f(1.0,0.0,0.0);     //黑色

        for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
        {
//            if(vpMPs[i]->isBad())
//                continue;
            pcl::PointXYZRGB pos = vpMPs[i];
//        cout << "point\t" << i << ": " << pos.at<float>(0) << ", " << pos.at<float>(1) << ", " << pos.at<float>(2) << endl;
            glVertex3f(pos.x,pos.y,pos.z);
        }
        glEnd();
    }

    void MapDrawer::DrawInlierLines() {
        const vector<pcl::PointXYZRGB> &vpMPs = mpMap->GetAllInlierLines();
        if(vpMPs.empty())
            return;

        glPointSize(7);
        glBegin(GL_POINTS);
        glColor3f(0.0,0.0,1.0);     //黑色

        for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
        {
//            if(vpMPs[i]->isBad())
//                continue;
            pcl::PointXYZRGB pos = vpMPs[i];
//        cout << "point\t" << i << ": " << pos.at<float>(0) << ", " << pos.at<float>(1) << ", " << pos.at<float>(2) << endl;
            glVertex3f(pos.x,pos.y,pos.z);
        }
        glEnd();
    }

    void MapDrawer::DrawMapLines()
    {
        const vector<MapLine*> &vpMLs = mpMap->GetAllMapLines();
        const vector<MapLine*> &vpRefMLs = mpMap->GetReferenceMapLines();

        set<MapLine*> spRefMLs(vpRefMLs.begin(), vpRefMLs.end());//set中不包含任何的重复元素

        if(vpMLs.empty())
            return;

        glLineWidth(mLineWidth);
        glBegin ( GL_LINES );
//    glColor3f(0.4, 0.35, 0.8);  //紫色
        glColor3f(0.0,0.0,0.0);    //黑色

//    cout << "vpMLs.size() = " << vpMLs.size() << endl;
        for(size_t i=0, iend=vpMLs.size(); i<iend; i++)
        {
            if(vpMLs[i]->isBad() || spRefMLs.count(vpMLs[i]))//.count() return true if the value can be found in the set.
                continue;
            Vector6d pos = vpMLs[i]->GetWorldPos();
//        cout << "line = " << pos.head(3).transpose() << "\n" << pos.tail(3).transpose() << endl;

            glVertex3f(pos(0), pos(1), pos(2));
            glVertex3f(pos(3), pos(4), pos(5));

        }
        glEnd();

        glLineWidth(mLineWidth);
        glBegin ( GL_LINES );
        glColor3f(0.0,0.0,0.0); //红色
//    cout << "spRefMLs.size() = " << spRefMLs.size() << endl;

        for(set<MapLine*>::iterator sit=spRefMLs.begin(), send=spRefMLs.end(); sit!=send; sit++)
        {
//        cout << "(*sit)->isBad() = " << (*sit)->isBad() << endl;
            if((*sit)->isBad())
                continue;
            Vector6d pos = (*sit)->GetWorldPos();
//        cout << "pos = " << pos.head(3).transpose() << "\n" << pos.tail(3).transpose() << endl;
            glVertex3f(pos(0), pos(1), pos(2));
            glVertex3f(pos(3), pos(4), pos(5));
        }
        glEnd();
    }

    void MapDrawer::DrawPlaneIntersections() {
        const vector<Eigen::Matrix<double, 6, 1>> &vpMLs = mpMap->GetAllPlaneIntersections();

        if(vpMLs.empty())
            return;

        glLineWidth(5);
        glBegin ( GL_LINES );
        glColor3f(0.0,1.0,0.0);    //黑色
        for(size_t i=0, iend=vpMLs.size(); i<iend; i++)
        {

            Vector6d pos = vpMLs[i];
            glVertex3f(pos(0), pos(1), pos(2));
            glVertex3f(pos(3), pos(4), pos(5));

        }
        glEnd();
    }

    void MapDrawer::DrawDiscretePoints() {
        glPointSize(mPointSize*2);
        glBegin(GL_POINTS);
        float ir = 0.0;
        float ig = 1.0;
        float ib = 0.0;
        float norm = sqrt(ir*ir + ig*ig + ib*ib);
        glColor3f(ir/norm, ig/norm, ib/norm);
        if (!mpMap->DrawDiscretePoints.empty())
        {
            for (auto p: mpMap->DrawDiscretePoints) {
                glVertex3f(p.x,p.y,p.z);
            }
            glEnd();
        }
    }

    void MapDrawer::DrawCrossPointInMap(){
        const vector<cv::Mat> &vpMPs = mpMap->GetAllCrossPointInMap();
        if(vpMPs.empty())
            return;
        glPointSize(mPointSize*5);
        glBegin(GL_POINTS);

        for(auto pMP : vpMPs){
            float ir = 255.0;
            float ig = 255.0;
            float ib = 0.0;
            float norm = sqrt(ir*ir + ig*ig + ib*ib);
            glColor3f(ir/norm, ig/norm, ib/norm);
            glVertex3f(pMP.at<float>(0),pMP.at<float>(1),pMP.at<float>(2));
        }
        glEnd();
    }

    void MapDrawer::DrawOuterPLanes() {
        const vector<MapPlane*> &vpMPs = mpMap->mspOuterPlanes;
        if(vpMPs.empty())
            return;
        glPointSize(mPointSize*2);
        glBegin(GL_POINTS);
        for(auto pMP : vpMPs){
            float ir = 0.0;
            float ig = 1.0;
            float ib = 0.0;
            float norm = sqrt(ir*ir + ig*ig + ib*ib);
            glColor3f(ir/norm, ig/norm, ib/norm);
            for(auto& p : pMP->mvPlanePoints.get()->points){
                glVertex3f(p.x,p.y,p.z);
            }
        }
        glEnd();
    }

    void MapDrawer::DrawMapPlanes() {
        const vector<MapPlane*> &vpMPs = mpMap->GetAllMapPlanes();
        if(vpMPs.empty())
            return;
        glPointSize(mPointSize*2);
        glBegin(GL_POINTS);

        for(auto pMP : vpMPs){
            float ir = pMP->mRed;
            float ig = pMP->mGreen;
            float ib = pMP->mBlue;
            float norm = sqrt(ir*ir + ig*ig + ib*ib);
            glColor3f(ir/norm, ig/norm, ib/norm);
            for(auto& p : pMP->mvPlanePoints.get()->points){
                glVertex3f(p.x,p.y,p.z);
            }
        }
        glEnd();
    }

    void MapDrawer::DrawPlanePolygon() {
        const vector<MapPlane*> &vpMapPlanes = mpMap->GetAllMapPlanes();
        glPolygonMode(GL_FRONT, GL_FILL);
//        glPolygonMode(GL_BACK, GL_FILL);
        for (auto plane : vpMapPlanes)
        {
            if (plane->msLayout.size() != 4)
                continue;
            glBegin(GL_POLYGON);
            float r = static_cast<float>(plane->mRed) / 255.0;
            float g = static_cast<float>(plane->mGreen) / 255.0;
            float b = static_cast<float>(plane->mBlue) / 255.0;
            glColor3f(r, g, b);
            glVertex3f(plane->mPw1.x(), plane->mPw1.y(), plane->mPw1.z());
            glVertex3f(plane->mPw2.x(), plane->mPw2.y(), plane->mPw2.z());
            glVertex3f(plane->mPw3.x(), plane->mPw3.y(), plane->mPw3.z());
            glVertex3f(plane->mPw4.x(), plane->mPw4.y(), plane->mPw4.z());
            glEnd();
        }
    }

    void MapDrawer::DrawCubeBoundingBox() {
        const vector<std::tuple<pcl::PointXYZRGB,pcl::PointXYZRGB>> &vCube = mpMap->GetAllTupleCubes();
        if (vCube.empty())
            return;
        Eigen::Vector3f whd;
        glLineWidth(5);
        glBegin ( GL_LINES );
        glColor3f(0.0,0.0,0.0);    //黑色
        for (auto cube : vCube) {
            auto minPoint = std::get<0>(cube);
            auto maxPoint = std::get<1>(cube);
            glVertex3f(minPoint.x,minPoint.y,minPoint.z);
            glVertex3f(maxPoint.x,minPoint.y,minPoint.z);

            glVertex3f(minPoint.x,minPoint.y,minPoint.z);
            glVertex3f(minPoint.x,maxPoint.y,minPoint.z);

            glVertex3f(minPoint.x,minPoint.y,minPoint.z);
            glVertex3f(minPoint.x,minPoint.y,maxPoint.z);

            glVertex3f(maxPoint.x,maxPoint.y,maxPoint.z);
            glVertex3f(maxPoint.x,minPoint.y,maxPoint.z);

            glVertex3f(maxPoint.x,maxPoint.y,maxPoint.z);
            glVertex3f(minPoint.x,maxPoint.y,maxPoint.z);

            glVertex3f(maxPoint.x,maxPoint.y,maxPoint.z);
            glVertex3f(maxPoint.x,maxPoint.y,minPoint.z);

            glVertex3f(maxPoint.x,minPoint.y,maxPoint.z);
            glVertex3f(minPoint.x,minPoint.y,maxPoint.z);

            glVertex3f(maxPoint.x,minPoint.y,maxPoint.z);
            glVertex3f(maxPoint.x,minPoint.y,minPoint.z);

            glVertex3f(minPoint.x,maxPoint.y,maxPoint.z);
            glVertex3f(minPoint.x,maxPoint.y,minPoint.z);

            glVertex3f(minPoint.x,maxPoint.y,maxPoint.z);
            glVertex3f(minPoint.x,minPoint.y,maxPoint.z);

            glVertex3f(maxPoint.x,maxPoint.y,minPoint.z);
            glVertex3f(maxPoint.x,minPoint.y,minPoint.z);

            glVertex3f(maxPoint.x,maxPoint.y,minPoint.z);
            glVertex3f(minPoint.x,maxPoint.y,minPoint.z);
        }
        glEnd();
    }

    void MapDrawer::DrawCubePoints() {
        const vector<pcl::PointXYZRGB> &vCube = mpMap->GetAllCubes();
        if (vCube.empty())
            return;
        glPointSize(mPointSize*4);
        glBegin(GL_POINTS);
        for (auto cube : vCube) {
            float ir = 0.0;
            float ig = 0.0;
            float ib = 0.0;
            float norm = sqrt(ir*ir + ig*ig + ib*ib);
            glColor3f(ir/norm, ig/norm, ib/norm);
            glVertex3f(cube.x,cube.y,cube.z);
        }
        glEnd();
    }

    void MapDrawer::DrawNoPlaneArea() {
        const vector<MapPlane*> &vpMPs = mpMap->GetAllMapPlanes();
        if(vpMPs.empty())
            return;
        glPointSize(mPointSize*2);
        glBegin(GL_POINTS);

        for(auto pMP : vpMPs){
            float ir = 0.0;
            float ig = 0.0;
            float ib = 0.0;
            float norm = sqrt(ir*ir + ig*ig + ib*ib);
            glColor3f(ir/norm, ig/norm, ib/norm);
            for(auto& p : pMP->mvNoPlanePoints.get()->points){
                glVertex3f(p.x,p.y,p.z);
            }
        }
        glEnd();
    }

    void MapDrawer::SaveMeshMode(const string& filename) {
        const vector<MapPlane*> &vpMPs = mpMap->GetAllMapPlanes();
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr meshCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        for (auto pMP : vpMPs) {
            int ir = 255.0;
            int ig = 255.0;
            int ib = 255.0;
            float norm = sqrt(ir*ir + ig*ig + ib*ib);
            for (auto &planePoint : pMP->mvPlanePoints.get()->points) {
                int i = 0;
                pcl::PointXYZRGB p;
                p.x = planePoint.x;
                p.y = planePoint.y;
                p.z = planePoint.z;
                p.r = ir/norm;
                p.g = ig/norm;
                p.b = ib/norm;
                meshCloud->points.push_back(p);
            }
        }
        if (meshCloud->points.size() > 0) {
            pcl::PolygonMesh cloud_mesh;
            pcl::io::savePLYFile(filename, *meshCloud);
        }
    }

    void MapDrawer::DrawNonPlaneArea() {
        glPointSize(mPointSize*2);
        glBegin(GL_POINTS);
        float ir = 0.0;
        float ig = 0.0;
        float ib = 0.0;
        float norm = sqrt(ir*ir + ig*ig + ib*ib);
        glColor3f(ir/norm, ig/norm, ib/norm);
        for (auto p: mpMap->DrawNonPlaneArea.points) {
            glVertex3f(p.x,p.y,p.z);
        }
        glEnd();
    }

    void MapDrawer::DrawLayouts() {
        auto lLayouts = mpMap->DrawLayout;
        glPointSize(mPointSize*4);
        glBegin(GL_POINTS);
        int i = 1;
        for (auto node : lLayouts)
        {
            node->mnId = i;
            i++;
        }
        for (auto node : lLayouts)
        {
            // 只显示yomposition)
//            {
//                if (node->mnId != menuLayoutId)
//                    continue;
//            }

            // layout节点
            glPointSize(4 * 4);
            glBegin(GL_POINTS);
            if (node->confidence == 1.0)
                glColor3f(0.0, 0.0, 0.0);
            else
                glColor3f(1.0, 0.0, 0.0);
            glVertex3f(node->pos[0], node->pos[1], node->pos[2]);

            // 在该节点处显示组成它的3个平面的id
//            if (true)
//            {
//                glColor3f(0.0, 0.0, 0.0);
//                if (!node->mvPlaneId.empty())
//                {
//                    string composedPlane = to_string(node->mvPlaneId[0]) + " " + to_string(node->mvPlaneId[1]) + " " + to_string(node->mvPlaneId[2]);
//                    pangolin::GlFont::I().Text(composedPlane).Draw(node->pos[0], node->pos[1], node->pos[2]);
//                }
//            }
        }
        glEnd();
    }

    void MapDrawer::DrawMapPlaneBoundaries() {
        const vector<MapPlane*> &vpMPs = mpMap->GetAllMapPlanes();
        if(vpMPs.empty())
            return;
        glPointSize(mPointSize*2);
        glBegin(GL_POINTS);

        for(auto pMP : vpMPs){
            float ir = 255.0;
            float ig = 0.0;
            float ib = 0.0;
            float norm = sqrt(ir*ir + ig*ig + ib*ib);
            glColor3f(ir/norm, ig/norm, ib/norm);
            if (pMP->cloud_boundary.get()->points.size() > 0)
            {
                for(auto& p : pMP->cloud_boundary.get()->points){
                    glVertex3f(p.x,p.y,p.z);
                }
            }
            else
            {
                continue;
            }
        }
        glEnd();
    }

    void MapDrawer::DrawCrossLine() {
        const vector<cv::Mat> &vpMLs = mpMap->GetAllCrossLines();
        if(vpMLs.empty())
            return;

        glLineWidth(mLineWidth);
        glBegin ( GL_LINES );
//    glColor3f(0.4, 0.35, 0.8);  //紫色
        glColor3f(0.0,0.0,0.0);    //黑色

//    cout << "vpMLs.size() = " << vpMLs.size() << endl;
        for(size_t i=0, iend=vpMLs.size(); i<iend; i++)
        {
            glVertex3f(vpMLs[i].at<float>(0), vpMLs[i].at<float>(1), vpMLs[i].at<float>(2));
        }
        glEnd();

        glLineWidth(mLineWidth);
        glBegin ( GL_LINES );
        glColor3f(0.0,0.0,0.0); //红色
        glEnd();
    }

    void MapDrawer::DrawCrossPoint() {
        const vector<cv::Mat> &vpMPs = mpMap->GetAllCrossPoints();
        if(vpMPs.empty())
            return;
        glPointSize(mPointSize*5);
        glBegin(GL_POINTS);

        for(auto pMP : vpMPs){
            float ir = 255.0;
            float ig = 255.0;
            float ib = 0.0;
            float norm = sqrt(ir*ir + ig*ig + ib*ib);
            glColor3f(ir/norm, ig/norm, ib/norm);
            glVertex3f(pMP.at<float>(0),pMP.at<float>(1),pMP.at<float>(2));
        }
        glEnd();
    }

    void MapDrawer::DrawSurfels() {
        const vector<SurfelElement> &vSurfels = mpMap->mvLocalSurfels;
        const vector<SurfelElement> &vInactiveSurfels = mpMap->mvInactiveSurfels;

        if(vSurfels.empty() && vInactiveSurfels.empty())
            return;

        glPointSize(mPointSize/2);
        glBegin(GL_POINTS);

        for(auto& p : vSurfels){
            float norm = sqrt(p.r*p.r + p.g*p.g + p.b*p.b);
            glColor3f(p.r/norm, p.g/norm, p.b/norm);
            glVertex3f(p.px,p.py,p.pz);
        }

        for(auto& p : vInactiveSurfels){
            float norm = sqrt(p.r*p.r + p.g*p.g + p.b*p.b);
            glColor3f(p.r/norm, p.g/norm, p.b/norm);
            glVertex3f(p.px,p.py,p.pz);
        }

        glEnd();
    }

    int num_j = 0;

    void MapDrawer::DrawMesh()
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glBegin(GL_POINTS);
        if (!mpMap->GetAllKeyFrames().empty())
        {
            double length = 4.0;
            int resolution = 216.0;
            double sdf_trunc_percentage = 0.01;
            open3d::pipelines::integration::ScalableTSDFVolume volume(
                    length / (double)resolution, length * sdf_trunc_percentage,
                    open3d::pipelines::integration::TSDFVolumeColorType::RGB8);
            const vector<KeyFrame*> keyframes = mpMap->GetAllKeyFrames();
            size_t N = keyframes.size();
            for(size_t i=lastKeyframeSize; i<N ; i++)
            {
                if(keyframes[i]->isBad())
                    continue;

                open3d::geometry::Image depth, color;

                color.FromCVMatRGB(keyframes[i]->mRGB);
                depth.FromCVMatRGB(keyframes[i]->mDepth);


                float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;
                fx = keyframes[i]->fx; fy = keyframes[i]->fy;
                cx = keyframes[i]->cx; cy = keyframes[i]->cy;
                cout << "depth " << depth.width_ << " " << depth.height_ << endl;

                auto rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(
                        color, depth, 5000.0, 7.0, false);
                open3d::camera::PinholeCameraIntrinsic intrinsic_
                        (depth.width_, depth.height_, fx, fy, cx, cy );
    //                intrinsic_.SetIntrinsics(640,480,
    //                                         481.2,-480.0,319.5,239.50);
                cv::Mat Twc = keyframes[i]->GetPose();// Inverse();
                Eigen::Matrix4d extrinsic;
                cv::cv2eigen(Twc,extrinsic);

                volume.Integrate(*rgbd, intrinsic_,   extrinsic);
            }

            auto pcd = volume.ExtractPointCloud();
            if (num_j == 0)
                globalMap = pcd;
            else
                *globalMap += *pcd;
            num_j++;
            lastKeyframeSize = N;
            for (int i = 0; i < globalMap->points_.size(); ++i) {
                Eigen::Vector3d color = globalMap->colors_[i];
                glColor3f(color[0], color[1], color[2]);
                Eigen::Vector3d point = globalMap->points_[i];
                glVertex3f(point[0], point[1], point[2]);
            }
            glEnd();
        }
    }

    void MapDrawer::TextureDenseMap()
    {
        std::string textureImg = "texture/";
        if (access(textureImg.c_str(), 0) == 0)
        {
            rmdir(textureImg.c_str());
        }

        mkdir(textureImg.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
        //mesh
        pcl::PolygonMesh triangles;
        pcl::io::loadPolygonFilePLY("mesh_.ply", triangles);
        //读取位姿
        vector<Eigen::Affine3f, aligned_allocator<Eigen::Affine3f> > T_vector;
        vector<string> img_index;
        const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
        pcl::texture_mapping::CameraVector my_cams;
        int saveKeyFrameId = 0;
        for(const auto pKF:vpKFs)
        {
            // camera
            pcl::TextureMapping<pcl::PointXYZ>::Camera cam;
            cv::Mat Tcw = pKF->GetPose();// Inverse();
//            cv::Mat Twc= Tcw.clone();
//            Twc.rowRange(0,3).colRange(0,3) = Tcw.rowRange(0,3).colRange(0,3).t();
//            cv::Mat tcw = -Tcw.rowRange(0,3).colRange(0,3).t().t()*Tcw.rowRange(0,3).col(3);
//            //cout<<tcw<<endl;
//            //cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);
//            tcw.copyTo(Twc.rowRange(0,3).col(3));
            cv::Mat Twc =  pKF->GetPoseInverse();

            Eigen::Affine3f T;

            Matrix4f M;
            M << Twc.at<float>(0, 0), Twc.at<float>(0, 1),Twc.at<float>(0, 2), Twc.at<float>(0, 3),
                    Twc.at<float>(1, 0), Twc.at<float>(1, 1),Twc.at<float>(1, 2), Twc.at<float>(1, 3),
                    Twc.at<float>(2, 0), Twc.at<float>(2, 1),Twc.at<float>(2, 2), Twc.at<float>(2, 3),
                    Twc.at<float>(3, 0), Twc.at<float>(3, 1),Twc.at<float>(3, 2), Twc.at<float>(3, 3);
            Eigen::Matrix4d extrinsic;
            cv::cv2eigen(Twc,M);
            T.matrix() = M;
            cout<<"M"<<M<<endl<<"T"<<T.matrix()<<endl;
            T_vector.push_back(T);
            cam.pose = T;
            cam.texture_file = "texture/keyframe_"+to_string(saveKeyFrameId)+".png";
            // imwrite keyframe
            cv::imwrite("texture/keyframe_"+to_string(saveKeyFrameId)+".png", pKF->mRGB);
            cam.focal_length_h = pKF->fy;
            cam.focal_length_w = pKF->fx;
            cam.center_h = pKF->cy;
            cam.center_w = pKF->cx;
            cam.height = pKF->mDepth.rows;//480;
            cam.width = pKF->mDepth.cols; // 640;
            my_cams.push_back (cam);
            saveKeyFrameId++;
        }


        pcl::TextureMesh mesh;
        mesh.cloud = triangles.cloud;
        mesh.header = triangles.header;
        mesh.tex_polygons.push_back(triangles.polygons);
        mesh.tex_materials.resize (my_cams.size () + 1);
        for(int i = 0; i <my_cams.size() ; ++i)
        {
            pcl::TexMaterial mesh_material;
            mesh_material.tex_Ka.r = 0.2f;
            mesh_material.tex_Ka.g = 0.2f;
            mesh_material.tex_Ka.b = 0.2f;
            mesh_material.tex_Kd.r = 0.8f;
            mesh_material.tex_Kd.g = 0.8f;
            mesh_material.tex_Kd.b = 0.8f;
            mesh_material.tex_Ks.r = 1.0f;
            mesh_material.tex_Ks.g = 1.0f;
            mesh_material.tex_Ks.b = 1.0f;
            mesh_material.tex_d = 2.5f;
            mesh_material.tex_Ns = 75.0f;
            mesh_material.tex_illum = 2;
            std::stringstream tex_name;
            tex_name << "material_" << i;
            tex_name >> mesh_material.tex_name;
            if(i < my_cams.size ())
                mesh_material.tex_file = my_cams[i].texture_file;
            else
                mesh_material.tex_file = "./occluded.jpg";

            mesh.tex_materials[i] = mesh_material;

        }

        pcl::TextureMapping<pcl::PointXYZ> tm; // TextureMapping object that will perform the sort
        tm.textureMeshwithMultipleCameras(mesh, my_cams);
        // compute normals for the mesh
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromPCLPointCloud2(triangles.cloud, *cloud);
        PCL_INFO ("\nEstimating normals...\n");
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
        pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud (cloud);
        n.setInputCloud (cloud);
        n.setSearchMethod (tree);
        n.setKSearch (20);
        n.compute (*normals);

        // Concatenate XYZ and normal fields
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
        pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
        PCL_INFO ("...Done.\n");

        pcl::toPCLPointCloud2 (*cloud_with_normals, mesh.cloud);
        pcl::io::saveOBJFile("mesh.obj", mesh, 5);

    }

    void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
    {
        const float &w = mKeyFrameSize;
        const float h = w*0.75;
        const float z = w*0.6;

        const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

        if(bDrawKF)
        {
            for(size_t i=0; i<vpKFs.size(); i++)
            {
                KeyFrame* pKF = vpKFs[i];
                cv::Mat Twc = pKF->GetPoseInverse().t();

                glPushMatrix();

                glMultMatrixf(Twc.ptr<GLfloat>(0));

                glLineWidth(mKeyFrameLineWidth);
                glColor3f(0.0f,0.0f,1.0f);
                glBegin(GL_LINES);
                glVertex3f(0,0,0);
                glVertex3f(w,h,z);
                glVertex3f(0,0,0);
                glVertex3f(w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,h,z);

                glVertex3f(w,h,z);
                glVertex3f(w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(-w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(w,h,z);

                glVertex3f(-w,-h,z);
                glVertex3f(w,-h,z);
                glEnd();

                glPopMatrix();
            }
        }

        if(bDrawGraph)
        {
            glLineWidth(mGraphLineWidth);
            glColor4f(0.0f,1.0f,0.0f,0.6f);
            glBegin(GL_LINES);

            for(size_t i=0; i<vpKFs.size(); i++)
            {
                // Covisibility Graph
                const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
                cv::Mat Ow = vpKFs[i]->GetCameraCenter();
                if(!vCovKFs.empty())
                {
                    for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                    {
                        if((*vit)->mnId<vpKFs[i]->mnId)
                            continue;
                        cv::Mat Ow2 = (*vit)->GetCameraCenter();
                        glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                        glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                    }
                }

                // Spanning tree
                KeyFrame* pParent = vpKFs[i]->GetParent();
                if(pParent)
                {
                    cv::Mat Owp = pParent->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    glVertex3f(Owp.at<float>(0),Owp.at<float>(1),Owp.at<float>(2));
                }

                // Loops
                set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
                for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
                {
                    if((*sit)->mnId<vpKFs[i]->mnId)
                        continue;
                    cv::Mat Owl = (*sit)->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    glVertex3f(Owl.at<float>(0),Owl.at<float>(1),Owl.at<float>(2));
                }
            }

            glEnd();
        }
    }

    void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
    {
        const float &w = mCameraSize;
        const float h = w*0.75;
        const float z = w*0.6;

        glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

        glLineWidth(mCameraLineWidth);
        glColor3f(0.0f,1.0f,0.0f);
        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();

        glPopMatrix();
    }


    void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
    {
        unique_lock<mutex> lock(mMutexCamera);
        mCameraPose = Tcw.clone();
    }

    void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
    {
        if(!mCameraPose.empty())
        {
            cv::Mat Rwc(3,3,CV_32F);
            cv::Mat twc(3,1,CV_32F);
            {
                unique_lock<mutex> lock(mMutexCamera);
                Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
                twc = -Rwc*mCameraPose.rowRange(0,3).col(3);
            }

            M.m[0] = Rwc.at<float>(0,0);
            M.m[1] = Rwc.at<float>(1,0);
            M.m[2] = Rwc.at<float>(2,0);
            M.m[3]  = 0.0;

            M.m[4] = Rwc.at<float>(0,1);
            M.m[5] = Rwc.at<float>(1,1);
            M.m[6] = Rwc.at<float>(2,1);
            M.m[7]  = 0.0;

            M.m[8] = Rwc.at<float>(0,2);
            M.m[9] = Rwc.at<float>(1,2);
            M.m[10] = Rwc.at<float>(2,2);
            M.m[11]  = 0.0;

            M.m[12] = twc.at<float>(0);
            M.m[13] = twc.at<float>(1);
            M.m[14] = twc.at<float>(2);
            M.m[15]  = 1.0;
        }
        else
            M.SetIdentity();
    }

} //namespace ORB_SLAM