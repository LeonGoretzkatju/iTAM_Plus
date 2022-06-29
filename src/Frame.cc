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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"

#include <thread>
#include <include/LocalMapping.h>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace ORB_SLAM2 {

    long unsigned int Frame::nNextId = 0;
    bool Frame::mbInitialComputations = true;
    float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
    float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
    float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

    Frame::Frame() {}

//Copy Constructor
    Frame::Frame(const Frame &frame)
            : mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft),
              mpORBextractorRight(frame.mpORBextractorRight),
              mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
              mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
              mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
              mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
              mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
              mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
              mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
              mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
              mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
              mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2),
              mLdesc(frame.mLdesc), NL(frame.NL), mvKeylinesUn(frame.mvKeylinesUn),
              mvpMapLines(frame.mvpMapLines),  //线特征相关的类成员变量
              mvbLineOutlier(frame.mvbLineOutlier), mvKeyLineFunctions(frame.mvKeyLineFunctions),
              mvDepthLine(frame.mvDepthLine), mvPlaneCoefficients(frame.mvPlaneCoefficients),
              mbNewPlane(frame.mbNewPlane),
              mvpMapPlanes(frame.mvpMapPlanes), mnPlaneNum(frame.mnPlaneNum), mvbPlaneOutlier(frame.mvbPlaneOutlier),
              mvpParallelPlanes(frame.mvpParallelPlanes), mvpVerticalPlanes(frame.mvpVerticalPlanes),
              mvPlanePoints(frame.mvPlanePoints), mvNoPlanePoints(frame.mvNoPlanePoints) {
        for (int i = 0; i < FRAME_GRID_COLS; i++)
            for (int j = 0; j < FRAME_GRID_ROWS; j++)
                mGrid[i][j] = frame.mGrid[i][j];

        if (!frame.mTcw.empty())
            SetPose(frame.mTcw);
    }


    Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor *extractorLeft,
                 ORBextractor *extractorRight, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf,
                 const float &thDepth)
            : mpORBvocabulary(voc), mpORBextractorLeft(extractorLeft), mpORBextractorRight(extractorRight),
              mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
              mpReferenceKF(static_cast<KeyFrame *>(NULL)) {
        // Frame ID
        mnId = nNextId++;

        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

        // ORB extraction
        thread threadLeft(&Frame::ExtractORB, this, 0, imLeft);
        thread threadRight(&Frame::ExtractORB, this, 1, imRight);
        threadLeft.join();
        threadRight.join();

        N = mvKeys.size();

        if (mvKeys.empty())
            return;

        UndistortKeyPoints();

        ComputeStereoMatches();

        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mvbOutlier = vector<bool>(N, false);


        // This is done only for the first Frame (or after a change in the calibration)
        if (mbInitialComputations) {
            ComputeImageBounds(imLeft);

            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            mbInitialComputations = false;
        }

        mb = mbf / fx;

        AssignFeaturesToGrid();
    }

    Frame::Frame(const cv::Mat &imRGB, const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp,
                 ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf,
                 const float &thDepth, const float &depthMapFactor)
            : mpORBvocabulary(voc), mpORBextractorLeft(extractor),mDepth(imDepth),
              mpORBextractorRight(static_cast<ORBextractor *>(NULL)),
              mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth) {
        // Frame ID
        mnId = nNextId++;
        mRGB = imRGB;
        mDepth = imDepth;
        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        cv::Mat imDepthScaled;
        if (depthMapFactor != 1 || imDepth.type() != CV_32F) {
            imDepth.convertTo(imDepthScaled, CV_32F, depthMapFactor);
        }
//        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        thread threadPoints(&ORB_SLAM2::Frame::ExtractORB, this, 0, imGray);
        thread threadLines(&ORB_SLAM2::Frame::ExtractLSD, this, imGray);
        thread threadPlanes(&ORB_SLAM2::Frame::ExtractPlanes, this, imRGB, imDepth, K, depthMapFactor);
        Params params_;
//        nonPlaneArea = 1 -
//        ExtractInseg(imRGB,imDepth,K,params_);
        threadPoints.join();
        threadLines.join();
        threadPlanes.join();
//        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//        double t12= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

//        std::ofstream fileWrite("FeatureExtraction_opt.dat", std::ios::binary | std::ios::app);
//        fileWrite.write((char*) &t12, sizeof(double));
//        fileWrite.close();

        N = mvKeys.size();
        NL = mvKeylinesUn.size();

        mnPlaneNum = mvPlanePoints.size();
        mvpMapPlanes = vector<MapPlane *>(mnPlaneNum, static_cast<MapPlane *>(nullptr));
        mvpParallelPlanes = vector<MapPlane *>(mnPlaneNum, static_cast<MapPlane *>(nullptr));
        mvpVerticalPlanes = vector<MapPlane *>(mnPlaneNum, static_cast<MapPlane *>(nullptr));
        mvPlanePointMatches = vector<vector<MapPoint *>>(mnPlaneNum);
        mvPlaneLineMatches = vector<vector<MapLine *>>(mnPlaneNum);
        mvbPlaneOutlier = vector<bool>(mnPlaneNum, false);
        mvbVerPlaneOutlier = vector<bool>(mnPlaneNum, false);
        mvbParPlaneOutlier = vector<bool>(mnPlaneNum, false);

        GetLineDepth(imDepthScaled);

        if (mvKeys.empty())
            return;

        UndistortKeyPoints();

        ComputeStereoFromRGBD(imDepthScaled);

        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mvpMapLines = vector<MapLine *>(NL, static_cast<MapLine *>(NULL));
        mvbOutlier = vector<bool>(N, false);
        mvbLineOutlier = vector<bool>(NL, false);


        // This is done only for the first Frame (or after a change in the calibration)
        if (mbInitialComputations) {
            ComputeImageBounds(imGray);

            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            mbInitialComputations = false;
        }

        mb = mbf / fx;

        AssignFeaturesToGrid();
    }


    Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor *extractor, ORBVocabulary *voc,
                 cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
            : mpORBvocabulary(voc), mpORBextractorLeft(extractor),
              mpORBextractorRight(static_cast<ORBextractor *>(NULL)),
              mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth) {
        // Frame ID
        mnId = nNextId++;

        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

        // ORB extraction
        ExtractORB(0, imGray);

        N = mvKeys.size();

        if (mvKeys.empty())
            return;

        UndistortKeyPoints();

        // Set no stereo information
        mvuRight = vector<float>(N, -1);
        mvDepth = vector<float>(N, -1);

        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mvbOutlier = vector<bool>(N, false);

        // This is done only for the first Frame (or after a change in the calibration)
        if (mbInitialComputations) {
            ComputeImageBounds(imGray);

            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            mbInitialComputations = false;
        }

        mb = mbf / fx;

        AssignFeaturesToGrid();
    }

    void Frame::AssignFeaturesToGrid() {
        int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
        for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
            for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
                mGrid[i][j].reserve(nReserve);

        for (int i = 0; i < N; i++) {
            const cv::KeyPoint &kp = mvKeysUn[i];

            int nGridPosX, nGridPosY;
            if (PosInGrid(kp, nGridPosX, nGridPosY))
                mGrid[nGridPosX][nGridPosY].push_back(i);
        }
    }

    void Frame::ExtractLSD(const cv::Mat &im) {
        mpLineSegment->ExtractLineSegment(im, mvKeylinesUn, mLdesc, mvKeyLineFunctions);

    }

    void Frame::ExtractORB(int flag, const cv::Mat &im) {
        if (flag == 0)
            (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors);
        else
            (*mpORBextractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight);
    }

    void Frame::GetLineDepth(const cv::Mat &imDepth) {
        mvDepthLine = std::vector<std::pair<float,float>>(mvKeylinesUn.size(), make_pair(-1.0f, -1.0f));

        for (int i = 0; i < mvKeylinesUn.size(); ++i) {
            mvDepthLine[i] = std::make_pair(imDepth.at<float>(mvKeylinesUn[i].startPointY, mvKeylinesUn[i].startPointX),
                                            imDepth.at<float>(mvKeylinesUn[i].endPointY, mvKeylinesUn[i].endPointX));
        }
    }

    void Frame::SetPose(cv::Mat Tcw) {
        mTcw = Tcw.clone();
        UpdatePoseMatrices();
    }

    void Frame::UpdatePoseMatrices() {
        mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
        mRwc = mRcw.t();
        mtcw = mTcw.rowRange(0, 3).col(3);
        mOw = -mRcw.t() * mtcw;

        mTwc = cv::Mat::eye(4, 4, mTcw.type());
        mRwc.copyTo(mTwc.rowRange(0, 3).colRange(0, 3));
        mOw.copyTo(mTwc.rowRange(0, 3).col(3));
    }

    bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit) {
        pMP->mbTrackInView = false;

        // 3D in absolute coordinates
        cv::Mat P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const cv::Mat Pc = mRcw * P + mtcw;
        const float &PcX = Pc.at<float>(0);
        const float &PcY = Pc.at<float>(1);
        const float &PcZ = Pc.at<float>(2);

        // Check positive depth
        if (PcZ < 0.0f)
            return false;

        // Project in image and check it is not outside
        const float invz = 1.0f / PcZ;
        const float u = fx * PcX * invz + cx;
        const float v = fy * PcY * invz + cy;

        if (u < mnMinX || u > mnMaxX)
            return false;
        if (v < mnMinY || v > mnMaxY)
            return false;

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const cv::Mat PO = P - mOw;
        const float dist = cv::norm(PO);

        if (dist < minDistance || dist > maxDistance)
            return false;

        // Check viewing angle
        cv::Mat Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn) / dist;

        if (viewCos < viewingCosLimit)
            return false;

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist, this);

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = u;
        pMP->mTrackProjXR = u - mbf * invz;
        pMP->mTrackProjY = v;
        pMP->mnTrackScaleLevel = nPredictedLevel;
        pMP->mTrackViewCos = viewCos;

        return true;
    }

    bool Frame::isInFrustum(MapLine *pML, float viewingCosLimit) {
        pML->mbTrackInView = false;

        Vector6d P = pML->GetWorldPos();

        cv::Mat SP = (Mat_<float>(3, 1) << P(0), P(1), P(2));
        cv::Mat EP = (Mat_<float>(3, 1) << P(3), P(4), P(5));

        const cv::Mat SPc = mRcw * SP + mtcw;
        const float &SPcX = SPc.at<float>(0);
        const float &SPcY = SPc.at<float>(1);
        const float &SPcZ = SPc.at<float>(2);

        const cv::Mat EPc = mRcw * EP + mtcw;
        const float &EPcX = EPc.at<float>(0);
        const float &EPcY = EPc.at<float>(1);
        const float &EPcZ = EPc.at<float>(2);

        if (SPcZ < 0.0f || EPcZ < 0.0f)
            return false;

        const float invz1 = 1.0f / SPcZ;
        const float u1 = fx * SPcX * invz1 + cx;
        const float v1 = fy * SPcY * invz1 + cy;

        if (u1 < mnMinX || u1 > mnMaxX)
            return false;
        if (v1 < mnMinY || v1 > mnMaxY)
            return false;

        const float invz2 = 1.0f / EPcZ;
        const float u2 = fx * EPcX * invz2 + cx;
        const float v2 = fy * EPcY * invz2 + cy;

        if (u2 < mnMinX || u2 > mnMaxX)
            return false;
        if (v2 < mnMinY || v2 > mnMaxY)
            return false;


        const float maxDistance = pML->GetMaxDistanceInvariance();
        const float minDistance = pML->GetMinDistanceInvariance();

        const cv::Mat OM = 0.5 * (SP + EP) - mOw;
        const float dist = cv::norm(OM);

        if (dist < minDistance || dist > maxDistance)
            return false;


        Vector3d Pn = pML->GetNormal();
        cv::Mat pn = (Mat_<float>(3, 1) << Pn(0), Pn(1), Pn(2));
        const float viewCos = OM.dot(pn) / dist;

        if (viewCos < viewingCosLimit)
            return false;

        const int nPredictedLevel = pML->PredictScale(dist, mfLogScaleFactor);

        pML->mbTrackInView = true;
        pML->mTrackProjX1 = u1;
        pML->mTrackProjY1 = v1;
        pML->mTrackProjX2 = u2;
        pML->mTrackProjY2 = v2;
        pML->mnTrackScaleLevel = nPredictedLevel;
        pML->mTrackViewCos = viewCos;

        return true;
    }


    vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel,
                                            const int maxLevel) const {
        vector<size_t> vIndices;
        vIndices.reserve(N);

        const int nMinCellX = max(0, (int) floor((x - mnMinX - r) * mfGridElementWidthInv));
        if (nMinCellX >= FRAME_GRID_COLS)
            return vIndices;

        const int nMaxCellX = min((int) FRAME_GRID_COLS - 1, (int) ceil((x - mnMinX + r) * mfGridElementWidthInv));
        if (nMaxCellX < 0)
            return vIndices;

        const int nMinCellY = max(0, (int) floor((y - mnMinY - r) * mfGridElementHeightInv));
        if (nMinCellY >= FRAME_GRID_ROWS)
            return vIndices;

        const int nMaxCellY = min((int) FRAME_GRID_ROWS - 1, (int) ceil((y - mnMinY + r) * mfGridElementHeightInv));
        if (nMaxCellY < 0)
            return vIndices;

        const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

        for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
            for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
                const vector<size_t> vCell = mGrid[ix][iy];
                if (vCell.empty())
                    continue;

                for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
                    const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                    if (bCheckLevels) {
                        if (kpUn.octave < minLevel)
                            continue;
                        if (maxLevel >= 0)
                            if (kpUn.octave > maxLevel)
                                continue;
                    }

                    const float distx = kpUn.pt.x - x;
                    const float disty = kpUn.pt.y - y;

                    if (fabs(distx) < r && fabs(disty) < r)
                        vIndices.push_back(vCell[j]);
                }
            }
        }

        return vIndices;
    }

    vector<size_t>
    Frame::GetLinesInArea(const float &x1, const float &y1, const float &x2, const float &y2, const float &r,
                          const int minLevel, const int maxLevel) const {
        vector<size_t> vIndices;

        vector<KeyLine> vkl = this->mvKeylinesUn;

        const bool bCheckLevels = (minLevel > 0) || (maxLevel > 0);

        for (size_t i = 0; i < vkl.size(); i++) {
            KeyLine keyline = vkl[i];

            // 1.对比中点距离
            float distance = (0.5 * (x1 + x2) - keyline.pt.x) * (0.5 * (x1 + x2) - keyline.pt.x) +
                             (0.5 * (y1 + y2) - keyline.pt.y) * (0.5 * (y1 + y2) - keyline.pt.y);
            if (distance > r * r)
                continue;

            float slope = (y1 - y2) / (x1 - x2) - keyline.angle;
            if (slope > r * 0.01)
                continue;

            if (bCheckLevels) {
                if (keyline.octave < minLevel)
                    continue;
                if (maxLevel >= 0 && keyline.octave > maxLevel)
                    continue;
            }

            vIndices.push_back(i);
        }

        return vIndices;
    }


    bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY) {
        posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
        posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

        //Keypoint's coordinates are undistorted, which could cause to go out of the image
        if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
            return false;

        return true;
    }


    void Frame::ComputeBoW() {
        if (mBowVec.empty()) {
            vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
            mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
        }
    }

    void Frame::UndistortKeyPoints() {
        if (mDistCoef.at<float>(0) == 0.0) {
            mvKeysUn = mvKeys;
            return;
        }

        // Fill matrix with points
        cv::Mat mat(N, 2, CV_32F);
        for (int i = 0; i < N; i++) {
            mat.at<float>(i, 0) = mvKeys[i].pt.x;
            mat.at<float>(i, 1) = mvKeys[i].pt.y;
        }

        // Undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        // Fill undistorted keypoint vector
        mvKeysUn.resize(N);
        for (int i = 0; i < N; i++) {
            cv::KeyPoint kp = mvKeys[i];
            kp.pt.x = mat.at<float>(i, 0);
            kp.pt.y = mat.at<float>(i, 1);
            mvKeysUn[i] = kp;
        }
    }

    void Frame::ComputeImageBounds(const cv::Mat &imLeft) {
        if (mDistCoef.at<float>(0) != 0.0) {
            cv::Mat mat(4, 2, CV_32F);
            mat.at<float>(0, 0) = 0.0;
            mat.at<float>(0, 1) = 0.0;
            mat.at<float>(1, 0) = imLeft.cols;
            mat.at<float>(1, 1) = 0.0;
            mat.at<float>(2, 0) = 0.0;
            mat.at<float>(2, 1) = imLeft.rows;
            mat.at<float>(3, 0) = imLeft.cols;
            mat.at<float>(3, 1) = imLeft.rows;

            // Undistort corners
            mat = mat.reshape(2);
            cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
            mat = mat.reshape(1);

            mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
            mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
            mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
            mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));

        } else {
            mnMinX = 0.0f;
            mnMaxX = imLeft.cols;
            mnMinY = 0.0f;
            mnMaxY = imLeft.rows;
        }
    }

    void Frame::ComputePlaneBoundary(PointCloud::Ptr &inputCloud) {
        int min_plane = Config::Get<int>("Plane.MinSize");
        float AngTh = Config::Get<float>("Plane.AngleThreshold");
        float DisTh = Config::Get<float>("Plane.DistanceThreshold");

        cout << "if the inputCloud is organized point cloud" << "   " << inputCloud->isOrganized() << endl;
        if (inputCloud->isOrganized())
        {
            vector<pcl::ModelCoefficients> coefficients;
            vector<pcl::PointIndices> inliers;
            pcl::PointCloud<pcl::Label>::Ptr labels ( new pcl::PointCloud<pcl::Label> );
            vector<pcl::PointIndices> label_indices;
            vector<pcl::PointIndices> boundary;

            pcl::NormalEstimation<PointT, pcl::Normal> NormalEst;
            NormalEst.setInputCloud(inputCloud);
            pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
            NormalEst.setSearchMethod (tree);
            pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
            NormalEst.setRadiusSearch (0.03);
            NormalEst.compute (*cloud_normals);


            pcl::OrganizedMultiPlaneSegmentation< PointT, pcl::Normal, pcl::Label > mps;
            mps.setMinInliers (min_plane);
            mps.setAngularThreshold (0.017453 * AngTh);
            mps.setDistanceThreshold (DisTh);
            mps.setInputNormals (cloud_normals);
            mps.setInputCloud (inputCloud);
            std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT>>> regions;
            mps.segmentAndRefine (regions, coefficients, inliers, labels, label_indices, boundary);


            pcl::ExtractIndices<PointT> extract;
            extract.setInputCloud(inputCloud);
            extract.setKeepOrganized(true);
            extract.setNegative(false);

            for (int i = 0; i < inliers.size(); ++i) {
                PointCloud::Ptr planeCloud(new PointCloud());
                extract.setIndices(boost::make_shared<pcl::PointIndices>(inliers[i]));
                extract.filter(*planeCloud);
                mvPlanePoints.push_back(*planeCloud);

                pcl::VoxelGrid<PointT> voxel;
                voxel.setLeafSize(0.2, 0.2, 0.2);

                PointCloud::Ptr coarseCloud(new PointCloud());
                voxel.setInputCloud(planeCloud);
                voxel.filter(*coarseCloud);

                PointCloud::Ptr boundaryPoints(new PointCloud());
                boundaryPoints->points = regions[i].getContour();
                mvBoundaryPoints.push_back(*boundaryPoints);

                cv::Mat coef = (cv::Mat_<float>(4,1) << coefficients[i].values[0],
                        coefficients[i].values[1],
                        coefficients[i].values[2],
                        coefficients[i].values[3]);
                mvPlaneCoefficients.push_back(coef);
            }
        }
        else
        {
            pcl::PointCloud<pcl::Boundary> boundaries;
            pcl::BoundaryEstimation<pcl::PointXYZRGB,pcl::Normal,pcl::Boundary> boundEst;
            pcl::NormalEstimation<pcl::PointXYZRGB,pcl::Normal> normEst;
            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_boundary (new pcl::PointCloud<pcl::PointXYZRGB>);
            normEst.setInputCloud(inputCloud);
            normEst.setRadiusSearch(0.01);
            normEst.compute(*normals);
            boundEst.setInputCloud(inputCloud);
            boundEst.setInputNormals(normals);
            boundEst.setRadiusSearch(0.01);
            boundEst.setAngleThreshold(M_PI/4);
            boundEst.setSearchMethod(pcl::search::KdTree<pcl::PointXYZRGB>::Ptr (new pcl::search::KdTree<pcl::PointXYZRGB>));
            boundEst.compute(boundaries);
            PointCloud::Ptr boundaryPoints(new PointCloud());
//            for (int i = 0; i < inputCloud.get()->points.size(); ++i) {
//                if (boundaries[i].boundary_point > 0)
//                {
////                    boundaryPoints = inputCloud->points[i];
//                    mvBoundaryPoints.push_back((inputCloud->points[i]));
//                }
//            }
        }
    }

    void Frame::ComputeStereoMatches() {
        mvuRight = vector<float>(N, -1.0f);
        mvDepth = vector<float>(N, -1.0f);

        const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;

        const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

        //Assign keypoints to row table
        vector<vector<size_t> > vRowIndices(nRows, vector<size_t>());

        for (int i = 0; i < nRows; i++)
            vRowIndices[i].reserve(200);

        const int Nr = mvKeysRight.size();

        for (int iR = 0; iR < Nr; iR++) {
            const cv::KeyPoint &kp = mvKeysRight[iR];
            const float &kpY = kp.pt.y;
            const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
            const int maxr = ceil(kpY + r);
            const int minr = floor(kpY - r);

            for (int yi = minr; yi <= maxr; yi++)
                vRowIndices[yi].push_back(iR);
        }

        // Set limits for search
        const float minZ = mb;
        const float minD = 0;
        const float maxD = mbf / minZ;

        // For each left keypoint search a match in the right image
        vector<pair<int, int> > vDistIdx;
        vDistIdx.reserve(N);

        for (int iL = 0; iL < N; iL++) {
            const cv::KeyPoint &kpL = mvKeys[iL];
            const int &levelL = kpL.octave;
            const float &vL = kpL.pt.y;
            const float &uL = kpL.pt.x;

            const vector<size_t> &vCandidates = vRowIndices[vL];

            if (vCandidates.empty())
                continue;

            const float minU = uL - maxD;
            const float maxU = uL - minD;

            if (maxU < 0)
                continue;

            int bestDist = ORBmatcher::TH_HIGH;
            size_t bestIdxR = 0;

            const cv::Mat &dL = mDescriptors.row(iL);

            // Compare descriptor to right keypoints
            for (size_t iC = 0; iC < vCandidates.size(); iC++) {
                const size_t iR = vCandidates[iC];
                const cv::KeyPoint &kpR = mvKeysRight[iR];

                if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
                    continue;

                const float &uR = kpR.pt.x;

                if (uR >= minU && uR <= maxU) {
                    const cv::Mat &dR = mDescriptorsRight.row(iR);
                    const int dist = ORBmatcher::DescriptorDistance(dL, dR);

                    if (dist < bestDist) {
                        bestDist = dist;
                        bestIdxR = iR;
                    }
                }
            }

            // Subpixel match by correlation
            if (bestDist < thOrbDist) {
                // coordinates in image pyramid at keypoint scale
                const float uR0 = mvKeysRight[bestIdxR].pt.x;
                const float scaleFactor = mvInvScaleFactors[kpL.octave];
                const float scaleduL = round(kpL.pt.x * scaleFactor);
                const float scaledvL = round(kpL.pt.y * scaleFactor);
                const float scaleduR0 = round(uR0 * scaleFactor);

                // sliding window search
                const int w = 5;
                cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL - w,
                                                                                     scaledvL + w + 1).colRange(
                        scaleduL - w, scaleduL + w + 1);
                IL.convertTo(IL, CV_32F);
                IL = IL - IL.at<float>(w, w) * cv::Mat::ones(IL.rows, IL.cols, CV_32F);

                int bestDist = INT_MAX;
                int bestincR = 0;
                const int L = 5;
                vector<float> vDists;
                vDists.resize(2 * L + 1);

                const float iniu = scaleduR0 + L - w;
                const float endu = scaleduR0 + L + w + 1;
                if (iniu < 0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                    continue;

                for (int incR = -L; incR <= +L; incR++) {
                    cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL - w,
                                                                                          scaledvL + w + 1).colRange(
                            scaleduR0 + incR - w, scaleduR0 + incR + w + 1);
                    IR.convertTo(IR, CV_32F);
                    IR = IR - IR.at<float>(w, w) * cv::Mat::ones(IR.rows, IR.cols, CV_32F);

                    float dist = cv::norm(IL, IR, cv::NORM_L1);
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestincR = incR;
                    }

                    vDists[L + incR] = dist;
                }

                if (bestincR == -L || bestincR == L)
                    continue;

                // Sub-pixel match (Parabola fitting)
                const float dist1 = vDists[L + bestincR - 1];
                const float dist2 = vDists[L + bestincR];
                const float dist3 = vDists[L + bestincR + 1];

                const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

                if (deltaR < -1 || deltaR > 1)
                    continue;

                // Re-scaled coordinate
                float bestuR = mvScaleFactors[kpL.octave] * ((float) scaleduR0 + (float) bestincR + deltaR);

                float disparity = (uL - bestuR);

                if (disparity >= minD && disparity < maxD) {
                    if (disparity <= 0) {
                        disparity = 0.01;
                        bestuR = uL - 0.01;
                    }
                    mvDepth[iL] = mbf / disparity;
                    mvuRight[iL] = bestuR;
                    vDistIdx.push_back(pair<int, int>(bestDist, iL));
                }
            }
        }

        sort(vDistIdx.begin(), vDistIdx.end());
        const float median = vDistIdx[vDistIdx.size() / 2].first;
        const float thDist = 1.5f * 1.4f * median;

        for (int i = vDistIdx.size() - 1; i >= 0; i--) {
            if (vDistIdx[i].first < thDist)
                break;
            else {
                mvuRight[vDistIdx[i].second] = -1;
                mvDepth[vDistIdx[i].second] = -1;
            }
        }
    }


    void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth) {
        mvuRight = vector<float>(N, -1);
        mvDepth = vector<float>(N, -1);

        for (int i = 0; i < N; i++) {
            const cv::KeyPoint &kp = mvKeys[i];
            const cv::KeyPoint &kpU = mvKeysUn[i];

            const float &v = kp.pt.y;
            const float &u = kp.pt.x;

            const float d = imDepth.at<float>(v, u);

            if (d > 0) {
                mvDepth[i] = d;
                mvuRight[i] = kpU.pt.x - mbf / d;
            }
        }
    }

    cv::Mat Frame::UnprojectStereo(const int &i) {
        const float z = mvDepth[i];
        if (z > 0) {
            const float u = mvKeysUn[i].pt.x;
            const float v = mvKeysUn[i].pt.y;
            const float x = (u - cx) * z * invfx;
            const float y = (v - cy) * z * invfy;
            cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);
            return mRwc * x3Dc + mOw;
        } else
            return cv::Mat();
    }

    Vector6d Frame::obtain3DLine(const int &i, const cv::Mat &imDepth) {
        double len = cv::norm(mvKeylinesUn[i].getStartPoint() - mvKeylinesUn[i].getEndPoint());

        vector<cv::Point3d> pts3d;
        // iterate through a line
        double numSmp = (double) min((int) len, 100); //number of line points sampled

        pts3d.reserve(numSmp);

        for (int j = 0; j <= numSmp; ++j) {
            // use nearest neighbor to querry depth value
            // assuming position (0,0) is the top-left corner of image, then the
            // top-left pixel's center would be (0.5,0.5)
            cv::Point2d pt = mvKeylinesUn[i].getStartPoint() * (1 - j / numSmp) +
                             mvKeylinesUn[i].getEndPoint() * (j / numSmp);
            if (pt.x < 0 || pt.y < 0 || pt.x >= imDepth.cols || pt.y >= imDepth.rows) continue;
            int row, col; // nearest pixel for pt
            if ((floor(pt.x) == pt.x) && (floor(pt.y) == pt.y)) { // boundary issue
                col = max(int(pt.x - 1), 0);
                row = max(int(pt.y - 1), 0);
            } else {
                col = int(pt.x);
                row = int(pt.y);
            }

            float d = -1;
            if (imDepth.at<float>(row, col) <= 0.01) { // no depth info
                continue;
            } else {
                d = imDepth.at<float>(row, col);
            }
            cv::Point3d p;

            p.z = d;
            p.x = (col - cx) * p.z * invfx;
            p.y = (row - cy) * p.z * invfy;

            pts3d.push_back(p);

        }

        if (pts3d.size() < 10.0)
            return static_cast<Vector6d>(NULL);

        RandomLine3d tmpLine;
        vector<RandomPoint3d> rndpts3d;
        rndpts3d.reserve(pts3d.size());

        cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx,
                0, fy, cy,
                0, 0, 1);

        // compute uncertainty of 3d points
        for (auto & j : pts3d) {
            rndpts3d.push_back(compPt3dCov(j, K, 1));
        }
        // using ransac to extract a 3d line from 3d pts
        tmpLine = extract3dline_mahdist(rndpts3d);

        if (tmpLine.pts.size() / len > 0.4 && cv::norm(tmpLine.A - tmpLine.B) > 0.02) {
            //this line is reliable

            Vector6d line3D;
            line3D << tmpLine.A.x, tmpLine.A.y, tmpLine.A.z, tmpLine.B.x, tmpLine.B.y, tmpLine.B.z;

            cv::Mat Ac = (Mat_<float>(3, 1) << line3D(0), line3D(1), line3D(2));
            cv::Mat A = mRwc * Ac + mOw;
            cv::Mat Bc = (Mat_<float>(3, 1) << line3D(3), line3D(4), line3D(5));
            cv::Mat B = mRwc * Bc + mOw;
            line3D << A.at<float>(0, 0), A.at<float>(1, 0), A.at<float>(2, 0),
                    B.at<float>(0, 0), B.at<float>(1,0), B.at<float>(2, 0);
            return line3D;
        } else {
            return static_cast<Vector6d>(NULL);
        }
    }

    int ite = 0;

    void Frame::ExtractPlanes(const cv::Mat &imRGB, const cv::Mat &imDepth, const cv::Mat &K, const float &depthMapFactor) {
        planeDetector.readColorImage(imRGB);
        planeDetector.readDepthImage(imDepth, K, depthMapFactor);
        cout << "depthmap factor is " << depthMapFactor << endl;
        mask_img = planeDetector.runPlaneDetection(imDepth,imRGB);
        seg_img = planeDetector.seg_img_;
        cv::Mat depth_img = imDepth;
        cout << mask_img.size << endl;
        int vertex_id = 0;
        planeDetector.NoPlaneAreaCloud.vertices.resize(240 * 320);
        planeDetector.NoPlaneAreaCloud.verticesColour.resize(240 * 320);
        planeDetector.NoPlaneAreaCloud.w = 320;
        planeDetector.NoPlaneAreaCloud.h = 240;
        cv::Mat maskCopy_img = cv::Mat(240, 320, CV_8UC1);
        for (int i = 0; i < mask_img.rows; ++i) {
            for (int j = 0; j < mask_img.cols; ++j) {
//                if ((i+2) < mask_img.rows && (j+2) < mask_img.cols && (i-2) >= 0 && (j-2) >= 0)
//                {
//                    if ((int)mask_img.at<uchar>(i,j) == 0 && (int)mask_img.at<uchar>(i,j+2) == 0 &&
//                            (int)mask_img.at<uchar>(i+2,j) == 0 && (int)mask_img.at<uchar>(i+2,j+2) == 0 &&
//                            (int)mask_img.at<uchar>(i-2,j) == 0 && (int)mask_img.at<uchar>(i,j-2) == 0 &&
//                            (int)mask_img.at<uchar>(i-2,j-2) == 0 && 2*i <= mask_img.rows && 2*j <= mask_img.cols)
                    if ((int)mask_img.at<uchar>(i,j) == 0)
                    {
                        double z = (double)(depth_img.at<unsigned short>(2*i, 2*j)) * depthMapFactor;
                        if (_isnan(z))
                        {
                            planeDetector.NoPlaneAreaCloud.vertices[vertex_id++] = VertexType(0, 0, z);
                            continue;
                        }
                        double x = ((double)2*j - K.at<float>(0, 2)) * z / K.at<float>(0, 0);
                        double y = ((double)2*i - K.at<float>(1, 2)) * z / K.at<float>(1, 1);
                        planeDetector.NoPlaneAreaCloud.vertices[vertex_id++] = VertexType(x, y, z);
                        maskCopy_img.at<uchar>(i,j) = 255;
                    }
                    else
                        maskCopy_img.at<uchar>(i,j) = 0;
//                }
            }
        }
//        cout << "vertices size" << "       "  << planeDetector.NoPlaneAreaCloud.vertices.size() << endl;
//        imwrite("/home/nuc/NYU2/maskCopyimg/"+ to_string(ite)+".png", maskCopy_img);
//        ite ++;

        PointCloud::Ptr inputCloudNoPlane(new PointCloud());
        for (int w = 0; w < vertex_id; ++w) {
            PointT p1;
            p1.x = (float) planeDetector.NoPlaneAreaCloud.vertices[w][0];
            p1.y = (float) planeDetector.NoPlaneAreaCloud.vertices[w][1];
            p1.z = (float) planeDetector.NoPlaneAreaCloud.vertices[w][2];
            p1.r = static_cast<uint8_t>(0.0);
            p1.g = static_cast<uint8_t>(0.0);
            p1.b = static_cast<uint8_t>(0.0);
//            for (int i = 0; i < planeDetector.plane_num_; i++) {
//                auto extractedPlane = planeDetector.plane_filter.extractedPlanes[i];
//                double nx = extractedPlane->normal[0];
//                double ny = extractedPlane->normal[1];
//                double nz = extractedPlane->normal[2];
//                double cx = extractedPlane->center[0];
//                double cy = extractedPlane->center[1];
//                double cz = extractedPlane->center[2];
//
//                float d = (float) -(nx * cx + ny * cy + nz * cz);
//                float result = p1.x * nx + p1.y * ny + p1.z * nz + d;
//                if (result >= -0.1 && result <= 0.1)
//                {
//                    continue;
//                }
//                else
//                {
//                    inputCloudNoPlane->points.push_back(p1);
//                }
//            }
            inputCloudNoPlane->points.push_back(p1);
        }
//        for (auto & vertice : planeDetector.NoPlaneAreaCloud.vertices) {
//            PointT p1;
//            p1.x = (float) vertice[0];
//            p1.y = (float) vertice[1];
//            p1.z = (float) vertice[2];
//            p1.r = static_cast<uint8_t>(0.0);
//            p1.g = static_cast<uint8_t>(0.0);
//            p1.b = static_cast<uint8_t>(0.0);
//            inputCloudNoPlane->points.push_back(p1);
//        }

        pcl::VoxelGrid<PointT> voxel1;
        voxel1.setLeafSize(0.1, 0.1, 0.1);

        PointCloud::Ptr coarseCloudNoPlane(new PointCloud());
        voxel1.setInputCloud(inputCloudNoPlane);
        voxel1.filter(*coarseCloudNoPlane);
//        pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud(new pcl::PointCloud<pcl::PointXYZ>());
        mvNoPlanePoints = *coarseCloudNoPlane;
//        pcl::PointXYZ p;
//        for (auto &noplanepoint : mvNoPlanePoints.points) {
//            int i = 0;
//            p.x = noplanepoint.x;
//            p.y = noplanepoint.y;
//            p.z = noplanepoint.z;
//            meshCloud->points.push_back(p);
//        }

        auto disTh = Config::Get<double>("Plane.DistanceThreshold");

        for (int i = 0; i < planeDetector.plane_num_; i++) {
            auto &indices = planeDetector.plane_vertices_[i];
            PointCloud::Ptr inputCloud(new PointCloud());
            for (int j : indices) {
                PointT p;
                p.x = (float) planeDetector.cloud.vertices[j][0];
                p.y = (float) planeDetector.cloud.vertices[j][1];
                p.z = (float) planeDetector.cloud.vertices[j][2];
                p.r = static_cast<uint8_t>(planeDetector.cloud.verticesColour[j][0]);
                p.g = static_cast<uint8_t>(planeDetector.cloud.verticesColour[j][1]);
                p.b = static_cast<uint8_t>(planeDetector.cloud.verticesColour[j][2]);

                inputCloud->points.push_back(p);
            }

            auto extractedPlane = planeDetector.plane_filter.extractedPlanes[i];
            double nx = extractedPlane->normal[0];
            double ny = extractedPlane->normal[1];
            double nz = extractedPlane->normal[2];
            double cx = extractedPlane->center[0];
            double cy = extractedPlane->center[1];
            double cz = extractedPlane->center[2];

            float d = (float) -(nx * cx + ny * cy + nz * cz);

            pcl::VoxelGrid<PointT> voxel;
            voxel.setLeafSize(0.2, 0.2, 0.2);

            PointCloud::Ptr coarseCloud(new PointCloud());
            voxel.setInputCloud(inputCloud);
            voxel.filter(*coarseCloud);

            cv::Mat coef = (cv::Mat_<float>(4, 1) << nx, ny, nz, d);

            // bool valid = MaxPointDistanceFromPlane(coef, coarseCloud);

            // if (!valid) {
            //     continue;
            // }

//            bool matched = false;
//            for (int j = 0, jend = mvPlaneCoefficients.size(); j < jend; j++) {
//                cv::Mat plane = mvPlaneCoefficients[j];
//                double angle = nx * plane.at<float>(0) +
//                        ny * plane.at<float>(1) +
//                        nz * plane.at<float>(2);
//                if (angle > 0.999 || angle < -0.999) {
//                    double min = MinPointDistanceFromPlane(plane, coarseCloud);
//                    if (min < 0.01) {
//                        mvPlanePoints[j] += *coarseCloud;
//                        matched = true;
//                        break;
//                    }
//                }
//            }
//
//            if (matched)
//                continue;



            mvPlanePoints.push_back(*coarseCloud);
//            ComputePlaneBoundary(inputCloud);
            mvPlaneCoefficients.push_back(coef);
//            for (auto &planepoint : (coarseCloud->points)) {
//                int i = 0;
//                p.x = planepoint.x;
//                p.y = planepoint.y;
//                p.z = planepoint.z;
//                meshCloud->points.push_back(p);
//            }

//            planeDetector.NoPlaneAreaCloud.vertices.clear();
//            planeDetector.seg_img_.release();
//            planeDetector.color_img_.release();
        }

////        int r = 107;
////        int g = 240;
////        int b = 90;
//
//        PointCloud::Ptr printCloud(new PointCloud());
//
//        for (int i = 0; i < mvPlanePoints.size(); i++) {
////            r = (r + 60) % 255;
////            g = (g + 130) % 255;
////            b = (b + 20) % 255;
//
//            auto &points = mvPlanePoints[i].points;
//            for (auto &p : points) {
////                p.r = r;
////                p.g = g;
////                p.b = b;
//                printCloud->points.push_back(p);
//            }
//        }
//
//        PlaneViewer::cloudPoints = printCloud;
//        if (meshCloud->points.size() > 0) {
//            pcl::PolygonMesh cloud_mesh;
//            pcl::io::savePLYFile("/home/nuc/second.ply", *meshCloud);
//        }
//        static const std::string kWindowName = "seg_img";
//        cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
//        imshow(kWindowName, planeDetector.seg_img_);
//        cv::waitKey();
    }

    void Frame::ExtractInseg(const cv::Mat& rgb_image, const cv::Mat& depth_image,
                             const cv::Mat& depth_intrinsics,Params& params)
    {
        cout<<"ss"<<endl;

        cv::Mat* label_map;
        cv::Mat * normal_map;
        std::vector<cv::Mat>segment_masks;

        std::vector<Segment> segments;
//
//        LOG(INFO)<<"segmentSingleFrame";
//        CHECK(!rgb_image.empty());
//        CHECK(!depth_image.empty());
//        CHECK_NOTNULL(label_map);
//        CHECK_NOTNULL(normal_map);
//        CHECK_NOTNULL(segment_masks);
//        CHECK_NOTNULL(segments);


        DepthCamera depth_camera;
        DepthSegmenter depth_segmenter(depth_camera, params);
        cout<<"ss"<<endl;
        depth_camera.initialize(depth_image.rows, depth_image.cols, CV_32FC1,
                                depth_intrinsics);
        depth_segmenter.initialize();
        cout<<"ss"<<endl;
        cv::Mat rescaled_depth = cv::Mat(depth_image.size(), CV_32FC1);
        if (depth_image.type() == CV_16UC1) {
            cv::rgbd::rescaleDepth(depth_image, CV_32FC1, rescaled_depth);
        } else if (depth_image.type() != CV_32FC1) {
            cout<< "Depth image is of unknown type.";
        } else {
            rescaled_depth = depth_image;
        }
        cout<<"ss"<<endl;
        // Compute depth map from rescaled depth image.
        cv::Mat depth_map(rescaled_depth.size(), CV_32FC3);
        depth_segmenter.computeDepthMap(rescaled_depth, &depth_map);
        cout<<"computeDepthMap ss"<<endl;
        // Compute normals based on specified method.
        cv::Mat normal = cv::Mat(depth_map.size(), CV_32FC3, 0.0f);
        cout<<"computeDepthMap normal 1"<<endl;
        cout<<depth_map.size()<<endl;
        normal_map= &normal;
        cout<<"computeDepthMap normal 1"<<endl;
        //*normal_map = cv::Mat(depth_map.size(), CV_32FC3, 0.0f);
        cout<<"computeDepthMap normal"<<endl;
        if (params.normals.method ==
            SurfaceNormalEstimationMethod::kFals ||
            params.normals.method ==
            SurfaceNormalEstimationMethod::kSri ||
            params.normals.method ==
            SurfaceNormalEstimationMethod::
            kDepthWindowFilter) {
            depth_segmenter.computeNormalMap(depth_map, normal_map);
        } else if (params.normals.method ==
                   SurfaceNormalEstimationMethod::kLinemod) {
            depth_segmenter.computeNormalMap(depth_image, normal_map);
        }

        // Compute depth discontinuity map.
        cv::Mat discontinuity_map = cv::Mat::zeros(rescaled_depth.size(), CV_32FC1);
        if (params.depth_discontinuity.use_discontinuity) {
            depth_segmenter.computeDepthDiscontinuityMap(rescaled_depth,
                                                         &discontinuity_map);
        }
        cout<<"computeDepthMap convexity_map"<<endl;
        // Compute maximum distance map.
        cv::Mat distance_map = cv::Mat::zeros(rescaled_depth.size(), CV_32FC1);
        if (params.max_distance.use_max_distance) {
            depth_segmenter.computeMaxDistanceMap(depth_map, &distance_map);
        }

        // Compute minimum convexity map.
        cv::Mat convexity_map = cv::Mat::zeros(rescaled_depth.size(), CV_32FC1);
        if (params.min_convexity.use_min_convexity) {
            depth_segmenter.computeMinConvexityMap(depth_map, *normal_map,
                                                   &convexity_map);
        }
        cout<<"computeDepthMap convexity_map"<<endl;

        // Compute final edge map.
        cv::Mat edge_map(rescaled_depth.size(), CV_32FC1);
        depth_segmenter.computeFinalEdgeMap(convexity_map, distance_map,
                                            discontinuity_map, &edge_map);
        cout<<"computeDepthMap convexity_map"<<endl;

        // Label the remaning segments.
        cv::Mat remove_no_values = cv::Mat::zeros(edge_map.size(), edge_map.type());
        edge_map.copyTo(remove_no_values, rescaled_depth == rescaled_depth);
        edge_map = remove_no_values;
        // cout<<"computeDepthMap labelMap"<<endl;
        depth_segmenter.labelMap(rgb_image, rescaled_depth, depth_map, edge_map,
                                 *normal_map, label_map, &segment_masks, &segments);
        // cout<<"computeDepthMap labelMap"<<endl;

    }

    cv::Mat Frame::ComputePlaneWorldCoeff(const int &idx) {
        cv::Mat temp;
        cv::transpose(mTcw, temp);// transform to the world position
        return temp * mvPlaneCoefficients[idx];
    }

    double Frame::MinPointDistanceFromPlane(const cv::Mat &plane, PointCloud::Ptr pointCloud) {
        double min = INT_MAX;
        for (auto p : pointCloud->points) {
            double dis = abs(plane.at<float>(0, 0) * p.x +
                             plane.at<float>(1, 0) * p.y +
                             plane.at<float>(2, 0) * p.z +
                             plane.at<float>(3, 0));
            if (dis < min)
                min = dis;
        }

        return min;
    }

    bool Frame::MaxPointDistanceFromPlane(cv::Mat &plane, PointCloud::Ptr pointCloud) {
        auto disTh = Config::Get<double>("Plane.DistanceThreshold");
        bool erased = false;
//        double max = -1;
        double threshold = 0.04;
        int i = 0;
        auto &points = pointCloud->points;

//        map<float, vector<int>> bin;

//        std::cout << "points before: " << points.size() << std::endl;
        for (auto &p : points) {
            double absDis = abs(plane.at<float>(0) * p.x +
                             plane.at<float>(1) * p.y +
                             plane.at<float>(2) * p.z +
                             plane.at<float>(3));

//            float dis = plane.at<float>(0) * p.x +
//                             plane.at<float>(1) * p.y +
//                             plane.at<float>(2) * p.z +
//                             plane.at<float>(3);

            if (absDis > disTh)
                return false;

//            float val = roundf(dis * 1000) / 1000;
//
//            bin[val].push_back(i);

//            if (absDis > threshold) {
//                points.erase(points.begin() + i);
//                erased = true;
//                continue;
//            }

            i++;
        }

//        float maxVal;
//        int max = 0;
//        for (auto &kv : bin) {
////            std::cout << "bin val: " << kv.first << std::endl;
////            std::cout << "bin size: " << kv.second.size() << std::endl;
//            if (kv.second.size() > max) {
//                max = kv.second.size();
//                maxVal = kv.first;
//            }
//        }
//
//        vector<int> indices = bin[maxVal];
//
//        PointCloud::Ptr temp (new PointCloud());
//
//        for (int &idx : indices) {
//            temp->points.push_back(points[idx]);
//        }
//
//        pointCloud->clear();
//        *pointCloud += *temp;

//        std::cout << "points after: " << points.size() << std::endl;

//        if (erased) {
//            if (points.size() < 3) {
//                return false;
//            }
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            // Create the segmentation object
            pcl::SACSegmentation<PointT> seg;
            // Optional
            seg.setOptimizeCoefficients(true);
            // Mandatory
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(disTh);

            seg.setInputCloud(pointCloud);
            seg.segment(*inliers, *coefficients);

            float oldVal = plane.at<float>(3);
            float newVal = coefficients->values[3];

            cv::Mat oldPlane = plane.clone();

//            std::cout << "old plane: " << plane.at<float>(0) << " "
//                      << plane.at<float>(1) << " "
//                      << plane.at<float>(2) << " "
//                      << plane.at<float>(3) << std::endl;
//
//            std::cout << "new plane: " << coefficients->values[0] << " "
//                      << coefficients->values[1] << " "
//                      << coefficients->values[2] << " "
//                      << coefficients->values[3] << std::endl;

            plane.at<float>(0) = coefficients->values[0];
            plane.at<float>(1) = coefficients->values[1];
            plane.at<float>(2) = coefficients->values[2];
            plane.at<float>(3) = coefficients->values[3];

            if ((newVal < 0 && oldVal > 0) || (newVal > 0 && oldVal < 0)) {
                plane = -plane;
//                double dotProduct = plane.dot(oldPlane) / sqrt(plane.dot(plane) * oldPlane.dot(oldPlane));
//                std::cout << "Flipped plane: " << plane.t() << std::endl;
//                std::cout << "Flip plane: " << dotProduct << std::endl;
            }
//        }

        return true;
    }
} //namespace ORB_SLAM
