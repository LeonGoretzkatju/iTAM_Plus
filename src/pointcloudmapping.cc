/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "pointcloudmapping.h"
#include "open3d/pipelines/integration/ScalableTSDFVolume.h"
#include "open3d/pipelines/integration/TSDFVolume.h"

#include "open3d/geometry/Image.h"
#include "open3d/geometry/RGBDImage.h"
#include "open3d/camera/PinholeCameraIntrinsic.h"
#include <KeyFrame.h>
#include <opencv2/highgui/highgui.hpp>
#include "Converter.h"

#include <Eigen/Dense>

#include <opencv2/core/eigen.hpp>

#include <boost/make_shared.hpp>

PointCloudMapping::PointCloudMapping(double resolution_)
{
    this->resolution = resolution_;
    globalMap = std::make_shared<open3d::geometry::PointCloud>();
    viewerThread = make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );
}

void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}

void PointCloudMapping::insertKeyFrame(KeyFrame* kf)
{
    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;
    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back( kf );
    colorImgs.push_back( kf->mRGB );
    depthImgs.push_back( kf->mDepth );

    keyFrameUpdated.notify_one();
}

int num_i = 0;

void PointCloudMapping::viewer()
{
    while(1)
    {
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }

        // keyframe is updated
        size_t N=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N = keyframes.size();
        }

        double length = 4.0;
        int resolution = 512.0;
        double sdf_trunc_percentage = 0.01;
        open3d::pipelines::integration::ScalableTSDFVolume volume(
                length / (double)resolution, length * sdf_trunc_percentage,
                open3d::pipelines::integration::TSDFVolumeColorType::RGB8);
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

        lastKeyframeSize = N;

    }
}

