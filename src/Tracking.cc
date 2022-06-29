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


#include "Tracking.h"

#include "ORBmatcher.h"

#include "Optimizer.h"
#include "PnPsolver.h"


using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace ORB_SLAM2 {

    Tracking::Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap,
                       KeyFrameDatabase *pKFDB, const string &strSettingPath, const int sensor) :
            mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
            mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer *>(NULL)), mpSystem(pSys),
            mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0) {
// Load camera parameters from settings file

        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0) {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        int img_width = fSettings["Camera.width"];
        int img_height = fSettings["Camera.height"];

        cout << "img_width = " << img_width << endl;
        cout << "img_height = " << img_height << endl;

        initUndistortRectifyMap(mK, mDistCoef, Mat_<double>::eye(3, 3), mK, Size(img_width, img_height), CV_32F,
                                mUndistX, mUndistY);

        cout << "mUndistX size = " << mUndistX.size << endl;
        cout << "mUndistY size = " << mUndistY.size << endl;

        mbf = fSettings["Camera.bf"];

        float fps = fSettings["Camera.fps"];
        if (fps == 0)
            fps = 30;

// Max/Min Frames to insert keyframes and to check relocalisation
        mMinFrames = 0;
        mMaxFrames = fps;

        cout << endl << "Camera Parameters: " << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;
        cout << "- k1: " << DistCoef.at<float>(0) << endl;
        cout << "- k2: " << DistCoef.at<float>(1) << endl;
        if (DistCoef.rows == 5)
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        cout << "- fps: " << fps << endl;


        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if (mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;

// Load ORB parameters

        int nFeatures = fSettings["ORBextractor.nFeatures"];
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
        int nLevels = fSettings["ORBextractor.nLevels"];
        int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
        int fMinThFAST = fSettings["ORBextractor.minThFAST"];

        mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (sensor == System::STEREO)
            mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (sensor == System::MONOCULAR)
            mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        cout << endl << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

        if (sensor == System::STEREO || sensor == System::RGBD) {
            mThDepth = mbf * (float) fSettings["ThDepth"] / fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }

        if (sensor == System::RGBD) {
            mDepthMapFactor = fSettings["DepthMapFactor"];
            if (fabs(mDepthMapFactor) < 1e-5)
                mDepthMapFactor = 1;
            else
                mDepthMapFactor = 1.0f / mDepthMapFactor;
        }

        mfDThRef = fSettings["Plane.AssociationDisRef"];
        mfDThMon = fSettings["Plane.AssociationDisMon"];
        mfAThRef = fSettings["Plane.AssociationAngRef"];
        mfAThMon = fSettings["Plane.AssociationAngMon"];

        mfVerTh = fSettings["Plane.VerticalThreshold"];
        mfParTh = fSettings["Plane.ParallelThreshold"];

        manhattanCount = 0;
        fullManhattanCount = 0;

        fullManhattanFound = false;
//        mpPointCloudMapping = make_shared<MeshViewer>(mpMap);

    }


    void Tracking::SetLocalMapper(LocalMapping *pLocalMapper) {
        mpLocalMapper = pLocalMapper;
    }

    void Tracking::SetSurfelMapper(SurfelMapping *pSurfelMapper) {
        mpSurfelMapper = pSurfelMapper;
    }

    void Tracking::SetLoopClosing(LoopClosing *pLoopClosing) {
        mpLoopClosing = pLoopClosing;
    }

    void Tracking::SetViewer(Viewer *pViewer) {
        mpViewer = pViewer;
    }


    cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp) {
        mImGray = imRectLeft;
        cv::Mat imGrayRight = imRectRight;

        if (mImGray.channels() == 3) {
            if (mbRGB) {
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
            } else {
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
            }
        } else if (mImGray.channels() == 4) {
            if (mbRGB) {
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
            } else {
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
            }
        }

        mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary,
                              mK, mDistCoef, mbf, mThDepth);

        Track();

        return mCurrentFrame.mTcw.clone();
    }


    cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp) {
        mImRGB = imRGB;
        mImGray = imRGB;
        mImDepth = imD;

        if (mImGray.channels() == 3) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        } else if (mImGray.channels() == 4) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        mCurrentFrame = Frame(mImRGB, mImGray, mImDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK,
                              mDistCoef, mbf, mThDepth, mDepthMapFactor);

        if (mDepthMapFactor != 1 || mImDepth.type() != CV_32F) {
            mImDepth.convertTo(mImDepth, CV_32F, mDepthMapFactor);
        }

        Track();
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double t12= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        std::ofstream fileWrite12("Track.dat", std::ios::binary | std::ios::app);
        fileWrite12.write((char*) &t12, sizeof(double));
        fileWrite12.close();

        cout << "Track time: " << t12 << endl;

        return mCurrentFrame.mTcw.clone();
    }


    cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp) {
        mImGray = im;

        if (mImGray.channels() == 3) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        } else if (mImGray.channels() == 4) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }

        if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
            mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
        else
            mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf,
                                  mThDepth);

        Track();

        return mCurrentFrame.mTcw.clone();
    }

    void Tracking::Track() {

        if (mState == NO_IMAGES_YET) {
            mState = NOT_INITIALIZED;
        }

        mLastProcessedState = mState;

        // Get Map Mutex -> Map cannot be changed
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        if (mState == NOT_INITIALIZED) {
            if (mSensor == System::STEREO || mSensor == System::RGBD) {
                StereoInitialization();
                // mpSurfelMapper->InsertKeyFrame(mImGray.clone(), mImDepth.clone(), mCurrentFrame.planeDetector.plane_filter.membershipImg.clone(), mCurrentFrame.mTwc.clone(), 0, true);
            } else
                MonocularInitialization();

            mpFrameDrawer->Update(this);

            if (mState != OK)
                return;
        } else {
            bool bOK = false;
            bool bManhattan = false;
            // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
            if (!mbOnlyTracking) {

                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
                    mCurrentFrame.SetPose(mLastFrame.mTcw);
                } else {
                    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
                }

                PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);
                pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());
//                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
                bManhattan = DetectManhattan();
//                std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//                double t12= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

//                std::ofstream fileWrite12("MFrotation_opt.dat", std::ios::binary | std::ios::app);
//                fileWrite12.write((char*) &t12, sizeof(double));
//                fileWrite12.close();

                cout << "bManhattan: " << bManhattan << endl;
//                mCurrentFrame.mnId;

                if (bManhattan) {
                    // Translation (only) estimation
                    if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
                        bOK = TranslationEstimation();
                        if (!bOK) {
                            mCurrentFrame.SetPose(mLastFrame.mTcw);
                        }
                    } else {
//                        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
                        bOK = TranslationWithMotionModel();
//                        std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
//                        double t34= std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();

//                        std::ofstream fileWrite34("MFtranslation_opt.dat", std::ios::binary | std::ios::app);
//                        fileWrite34.write((char*) &t34, sizeof(double));
//                        fileWrite34.close();

                        if (!bOK) {
                            mCurrentFrame.SetPose(mLastFrame.mTcw);
                            bOK = TranslationEstimation();
                            if (!bOK) {
                                mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
                            }
                        }
                    }
                }

                if (bOK) {
                    if (bManhattan)
                        ++manhattanCount;

                    if (fullManhattanFound)
                        ++fullManhattanCount;
                }

                cout << "manhattanCount: " << manhattanCount << endl;
                cout << "fullManhattanCount: " << fullManhattanCount << endl;

                // Pose refinement
                if (!bOK) {
                    if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
                        bOK = TrackReferenceKeyFrame();
                        if (!bOK) {
                            mCurrentFrame.SetPose(mLastFrame.mTcw);
                        }
                    } else {
//                        std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();
                        bOK = TrackWithMotionModel();
//                        std::chrono::steady_clock::time_point t6 = std::chrono::steady_clock::now();
//                        double t56 = std::chrono::duration_cast<std::chrono::duration<double> >(t6 - t5).count();

//                        std::ofstream fileWrite56("PoseEstimation_opt.dat", std::ios::binary | std::ios::app);
//                        fileWrite56.write((char*) &t56, sizeof(double));
//                        fileWrite56.close();
                        if (!bOK) {
                            mCurrentFrame.SetPose(mLastFrame.mTcw);
                            bOK = TrackReferenceKeyFrame();
                            if (!bOK) {
                                mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
                            }
                        }
                    }

//                    mCurrentFrame.mpReferenceKF = mpReferenceKF;
//
//                    if (bOK) {
//                        bOK = TrackLocalMap();
//                    }
                } else {
//                    mCurrentFrame.mpReferenceKF = mpReferenceKF;
                }

                mCurrentFrame.mpReferenceKF = mpReferenceKF;

                if (bOK) {
//                    std::chrono::steady_clock::time_point t7 = std::chrono::steady_clock::now();
                    bOK = TrackLocalMap();
//                    std::chrono::steady_clock::time_point t8 = std::chrono::steady_clock::now();
//                    double t78 = std::chrono::duration_cast<std::chrono::duration<double> >(t8 - t7).count();

//                    std::ofstream fileWrite78("PoseRefinement_opt.dat", std::ios::binary | std::ios::app);
//                    fileWrite78.write((char*) &t78, sizeof(double));
//                    fileWrite78.close();
                } else {
                    bOK = Relocalization();
                }
            }

            if (bOK)
                mState = OK;
            else
                mState = LOST;


            // Update drawer
            mpFrameDrawer->Update(this);

            mpMap->FlagMatchedPlanePoints(mCurrentFrame, mfDThRef);

            //Update Planes
            for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
                MapPlane *pMP = mCurrentFrame.mvpMapPlanes[i];//i th mapplane in current frame
                if (pMP) {
                    pMP->UpdateCoefficientsAndPoints(mCurrentFrame, i);
                } else if (!mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mbNewPlane = true;
                }
            }

//            mpPointCloudMapping->print();
            if (bOK) {
                // Update motion model
                if (!mLastFrame.mTcw.empty()) {
                    cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                    mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                    mVelocity = mCurrentFrame.mTcw * LastTwc;
                    cv::Mat trans = mLastFrame.mTcw.rowRange(0, 3).col(3) - mCurrentFrame.mTcw.rowRange(0, 3).col(3);

                    // To store the sum of squares of the
                    // elements of the given matrix
                    float sumSq = 0;
//                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 2; j++) {
                            sumSq += pow(trans.at<float>(j), 2);
                        }
//                    }

                    // Return the square root of
                    // the sum of squares
                    double res = sqrt(sumSq);
                    cout << "trans: " << trans << endl;
                    cout << "trans norm: " << res << endl;
                    if (res > 0.1000) {
                        cout << "Faulty" << endl;
                    }
                } else
                    mVelocity = cv::Mat();

                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
                // Clean VO matches
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (pMP)
                        if (pMP->Observations() < 1) {
                            mCurrentFrame.mvbOutlier[i] = false;
                            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                        }
                }
                for (int i = 0; i < mCurrentFrame.NL; i++) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    if (pML)
                        if (pML->Observations() < 1) {
                            mCurrentFrame.mvbLineOutlier[i] = false;
                            mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                        }
                }

                // Delete temporal MapPoints
                for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end();
                     lit != lend; lit++) {
                    MapPoint *pMP = *lit;
                    delete pMP;
                }
                for (list<MapLine *>::iterator lit = mlpTemporalLines.begin(), lend = mlpTemporalLines.end();
                     lit != lend; lit++) {
                    MapLine *pML = *lit;
                    delete pML;
                }
                mlpTemporalPoints.clear();
                mlpTemporalLines.clear();

                int referenceIndex = 0;

                double timeDiff = 1e9;
                vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
                for(int i=0; i<vpKFs.size(); i++)
                {
                    double diff = fabs(vpKFs[i]->mTimeStamp - mpReferenceKF->mTimeStamp);
                    if (diff < timeDiff)
                    {
                        referenceIndex = i;
                        timeDiff = diff;
                    }
                }

                // Check if we need to insert a new keyframe
                bool isKeyFrame = NeedNewKeyFrame();

                if (isKeyFrame) {
                    CreateNewKeyFrame();
                }

                if (isKeyFrame)
                    // mpSurfelMapper->InsertKeyFrame(mImGray.clone(), mImDepth.clone(), mCurrentFrame.planeDetector.plane_filter.membershipImg.clone(), mCurrentFrame.mTwc.clone(), referenceIndex, isKeyFrame);

                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame.
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }

                for (int i = 0; i < mCurrentFrame.NL; i++) {
                    if (mCurrentFrame.mvpMapLines[i] && mCurrentFrame.mvbLineOutlier[i])
                        mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                }
            }

            // Reset if the camera get lost soon after initialization
            if (mState == LOST) {
                if (mpMap->KeyFramesInMap() <= 5) {
                    cout << "Track lost soon after initialisation, reseting..." << endl;
                    mpSystem->Reset();
                    return;
                }
            }

            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            mLastFrame = Frame(mCurrentFrame);
        }

        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if (!mCurrentFrame.mTcw.empty()) {
            cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr);
            mlpReferences.push_back(mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState == LOST);
        } else {
            // This can happen if tracking is lost
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState == LOST);
        }

    }

    void Tracking::StereoInitialization() {
        if (mCurrentFrame.N > 50 || mCurrentFrame.NL > 15) {
            // Set Frame pose to the origin
            mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

            // Create KeyFrame
            KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

            // Insert KeyFrame in the map
            mpMap->AddKeyFrame(pKFini);

            // Create MapPoints and asscoiate to KeyFrame
            for (int i = 0; i < mCurrentFrame.N; i++) {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0) {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap);
                    pNewMP->AddObservation(pKFini, i);
                    pKFini->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);
                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                }
            }


            for (int i = 0; i < mCurrentFrame.NL; i++) {

                pair<float,float> z = mCurrentFrame.mvDepthLine[i];

                if (z.first > 0 && z.second > 0) {
                    Vector6d line3D = mCurrentFrame.obtain3DLine(i, mImDepth);
                    if (line3D == static_cast<Vector6d>(NULL)) {
                        continue;
                    }
                    MapLine *pNewML = new MapLine(line3D, pKFini, mpMap);
                    pNewML->AddObservation(pKFini, i);
                    pKFini->AddMapLine(pNewML, i);
                    pNewML->ComputeDistinctiveDescriptors();
                    pNewML->UpdateAverageDir();
                    mpMap->AddMapLine(pNewML);
                    mCurrentFrame.mvpMapLines[i] = pNewML;
                }
            }

            for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
                cv::Mat p3D = mCurrentFrame.ComputePlaneWorldCoeff(i);
                MapPlane *pNewMP = new MapPlane(p3D, pKFini, mpMap);// create new plane in the Map,mpMap is the Map
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPlane(pNewMP, i);
                pNewMP->UpdateCoefficientsAndPoints();
                mpMap->AddMapPlane(pNewMP);
                mCurrentFrame.mvpMapPlanes[i] = pNewMP;
            }

//            mpPointCloudMapping->print();

            mpLocalMapper->InsertKeyFrame(pKFini);

            mLastFrame = Frame(mCurrentFrame);
            mnLastKeyFrameId = mCurrentFrame.mnId;
            mpLastKeyFrame = pKFini;

            mvpLocalKeyFrames.push_back(pKFini);
            mvpLocalMapPoints = mpMap->GetAllMapPoints();
            mvpLocalMapLines = mpMap->GetAllMapLines();

            mpReferenceKF = pKFini;
            mCurrentFrame.mpReferenceKF = pKFini;

            mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
            mpMap->SetReferenceMapLines(mvpLocalMapLines);

            mpMap->mvpKeyFrameOrigins.push_back(pKFini);

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            mState = OK;
        }
    }

    void Tracking::MonocularInitialization() {
        int num = 100;
        // 如果单目初始器还没有没创建，则创建单目初始器
        if (!mpInitializer) {
            // Set Reference Frame
            if (mCurrentFrame.mvKeys.size() > num) {
                // step 1：得到用于初始化的第一帧，初始化需要两帧
                mInitialFrame = Frame(mCurrentFrame);
                // 记录最近的一帧
                mLastFrame = Frame(mCurrentFrame);
                // mvbPreMatched最大的情况就是当前帧所有的特征点都被匹配上
                mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
                for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                    mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

                if (mpInitializer)
                    delete mpInitializer;

                // 由当前帧构造初始化器， sigma:1.0    iterations:200
                mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

                return;
            }
        } else {
            // Try to initialize
            // step2：如果当前帧特征点数大于100，则得到用于单目初始化的第二帧
            // 如果当前帧特征点太少，重新构造初始器
            // 因此只有连续两帧的特征点个数都大于100时，才能继续进行初始化过程
            if ((int) mCurrentFrame.mvKeys.size() <= num) {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
                return;
            }

            // Find correspondences
            // step3：在mInitialFrame与mCurrentFrame中找匹配的特征点对
            // mvbPrevMatched为前一帧的特征点，存储了mInitialFrame中哪些点将进行接下来的匹配,类型  std::vector<cv::Point2f> mvbPrevMatched;
            // mvIniMatches存储mInitialFrame, mCurrentFrame之间匹配的特征点，类型为std::vector<int> mvIniMatches; ????
            ORBmatcher matcher(0.9, true);
            int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches,
                                                           100);

            LSDmatcher lmatcher;   //建立线特征之间的匹配
            int lineMatches = lmatcher.SerachForInitialize(mInitialFrame, mCurrentFrame, mvLineMatches);
//        cout << "Tracking::MonocularInitialization(), lineMatches = " << lineMatches << endl;
//
//        cout << "Tracking::MonocularInitialization(), mvLineMatches size = " << mvLineMatches.size() << endl;

            // Check if there are enough correspondences
            // step4：如果初始化的两帧之间的匹配点太少，重新初始化
            if (nmatches < 100) {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                return;
            }

            cv::Mat Rcw; // Current Camera Rotation
            cv::Mat tcw; // Current Camera Translation
            vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

            // step5：通过H或者F进行单目初始化，得到两帧之间相对运动，初始化MapPoints
#if 0
                                                                                                                                    if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
    {
        for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
        {
            if(mvIniMatches[i]>=0 && !vbTriangulated[i])
            {
                mvIniMatches[i]=-1;
                nmatches--;
            }
        }

        // Set Frame Poses
        // 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
        mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
        cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
        Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
        tcw.copyTo(Tcw.rowRange(0,3).col(3));
        mCurrentFrame.SetPose(Tcw);

        // step6：将三角化得到的3D点包装成MapPoints
        /// 如果要修改，应该是从这个函数开始
        CreateInitialMapMonocular();
    }
#else
            if (0)//mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated, mvLineMatches, mvLineS3D, mvLineE3D, mvbLineTriangulated))
            {
                for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
                    if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
                        mvIniMatches[i] = -1;
                        nmatches--;
                    }
                }

                // Set Frame Poses
                // 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
                mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
                cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
                Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
                tcw.copyTo(Tcw.rowRange(0, 3).col(3));
                mCurrentFrame.SetPose(Tcw);

                // step6：将三角化得到的3D点包装成MapPoints
                /// 如果要修改，应该是从这个函数开始
//            CreateInitialMapMonocular();
                CreateInitialMapMonoWithLine();
            }
#endif
        }
    }

    void Tracking::CreateInitialMapMonocular() {
        // Create KeyFrames
        KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
        KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();

        // Insert KFs in the map
        mpMap->AddKeyFrame(pKFini);
        mpMap->AddKeyFrame(pKFcur);

        // Create MapPoints and asscoiate to keyframes
        for (size_t i = 0; i < mvIniMatches.size(); i++) {
            if (mvIniMatches[i] < 0)
                continue;

            //Create MapPoint.
            cv::Mat worldPos(mvIniP3D[i]);

            MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, mvIniMatches[i]);

            // b.从众多观测到该MapPoint的特征点中挑选出区分度最高的描述子
            pMP->ComputeDistinctiveDescriptors();

            // c.更新该MapPoint的平均观测方向以及观测距离的范围
            pMP->UpdateNormalAndDepth();

            //Fill Current Frame structure
            mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
            mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

            //Add to Map
            mpMap->AddMapPoint(pMP);
        }

        // Update Connections
        pKFini->UpdateConnections();
        pKFcur->UpdateConnections();

        // Bundle Adjustment
        cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

        Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

        // Set median depth to 1
        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth = 1.0f / medianDepth;

        if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100) {
            cout << "Wrong initialization, reseting..." << endl;
            Reset();
            return;
        }

        // Scale initial baseline
        cv::Mat Tc2w = pKFcur->GetPose();
        Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
        pKFcur->SetPose(Tc2w);

        // Scale points
        vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
        for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
            if (vpAllMapPoints[iMP]) {
                MapPoint *pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
            }
        }

        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);

        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;

        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;

        mLastFrame = Frame(mCurrentFrame);

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mState = OK;  //至此，初始化成功
    }

#if 1

/**
* @brief 为单目摄像头三角化生成带有线特征的Map，包括MapPoints和MapLine
*/
    void Tracking::CreateInitialMapMonoWithLine() {
        // step1:创建关键帧，即用于初始化的前两帧
        KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
        KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        // step2：将两个关键帧的描述子转为BoW，这里的BoW只有ORB的词袋
        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();

        // step3：将关键帧插入到地图，凡是关键帧，都要插入地图
        mpMap->AddKeyFrame(pKFini);
        mpMap->AddKeyFrame(pKFcur);

        // step4：将特征点的3D点包装成MapPoints
        for (size_t i = 0; i < mvIniMatches.size(); i++) {
            if (mvIniMatches[i] < 0)
                continue;

            // Create MapPoint
            cv::Mat worldPos(mvIniP3D[i]);

            // step4.1：用3D点构造MapPoint
            MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

            // step4.2：为该MapPoint添加属性：
            // a.观测到该MapPoint的关键帧
            // b.该MapPoint的描述子
            // c.该MapPoint的平均观测方向和深度范围

            // step4.3：表示该KeyFrame的哪个特征点对应到哪个3D点
            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

            // a.表示该MapPoint可以被哪个KeyFrame观测到，以及对应的第几个特征点
            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, mvIniMatches[i]);

            // b.从众多观测到该MapPoint的特征点中挑选出区分度最高的描述子
            pMP->UpdateNormalAndDepth();

            // Fill Current Frame structure
            mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
            mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

            // Add to Map
            // step4.4：在地图中添加该MapPoint
            mpMap->AddMapPoint(pMP);
        }

        // step5：将特征线包装成MapLines
        for (size_t i = 0; i < mvLineMatches.size(); i++) {
            if (!mvbLineTriangulated[i])
                continue;

            // Create MapLine
            Vector6d worldPos;
            worldPos << mvLineS3D[i].x, mvLineS3D[i].y, mvLineS3D[i].z, mvLineE3D[i].x, mvLineE3D[i].y, mvLineE3D[i].z;

            //step5.1：用线段的两个端点构造MapLine
            MapLine *pML = new MapLine(worldPos, pKFcur, mpMap);

            //step5.2：为该MapLine添加属性：
            // a.观测到该MapLine的关键帧
            // b.该MapLine的描述子
            // c.该MapLine的平均观测方向和深度范围？

            //step5.3：表示该KeyFrame的哪个特征点可以观测到哪个3D点
            pKFini->AddMapLine(pML, i);
            pKFcur->AddMapLine(pML, i);

            //a.表示该MapLine可以被哪个KeyFrame观测到，以及对应的第几个特征线
            pML->AddObservation(pKFini, i);
            pML->AddObservation(pKFcur, i);

            //b.MapPoint中是选取区分度最高的描述子，pl-slam直接采用前一帧的描述子,这里先按照ORB-SLAM的过程来
            pML->ComputeDistinctiveDescriptors();

            //c.更新该MapLine的平均观测方向以及观测距离的范围
            pML->UpdateAverageDir();

            // Fill Current Frame structure
            mCurrentFrame.mvpMapLines[i] = pML;
            mCurrentFrame.mvbLineOutlier[i] = false;

            // step5.4: Add to Map
            mpMap->AddMapLine(pML);
        }

        // step6：更新关键帧间的连接关系
        // 1.最初是在3D点和关键帧之间建立边，每一个边有一个权重，边的权重是该关键帧与当前关键帧公共3D点的个数
        // 2.加入线特征后，这个关系应该和特征线也有一定的关系，或者就先不加关系，只是单纯的添加线特征

        // step7：全局BA优化，这里需要再进一步修改优化函数，参照OptimizePose函数
        cout << "this Map created with " << mpMap->MapPointsInMap() << " points, and " << mpMap->MapLinesInMap()
             << " lines." << endl;
        //Optimizer::GlobalBundleAdjustemnt(mpMap, 20, true); //true代表使用有线特征的BA

        // step8：将MapPoints的中值深度归一化到1，并归一化两帧之间的变换
        // Q：MapPoints的中值深度归一化为1，MapLine是否也归一化？
        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth = 1.0f / medianDepth;

        cout << "medianDepth = " << medianDepth << endl;
        cout << "pKFcur->TrackedMapPoints(1) = " << pKFcur->TrackedMapPoints(1) << endl;

        if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 80) {
            cout << "Wrong initialization, reseting ... " << endl;
            Reset();
            return;
        }

        // Scale initial baseline
        cv::Mat Tc2w = pKFcur->GetPose();
        Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
        pKFcur->SetPose(Tc2w);

        // Scale Points
        vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
        for (size_t iMP = 0; iMP < vpAllMapPoints.size(); ++iMP) {
            if (vpAllMapPoints[iMP]) {
                MapPoint *pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
            }
        }

        // Scale Line Segments
        vector<MapLine *> vpAllMapLines = pKFini->GetMapLineMatches();
        for (size_t iML = 0; iML < vpAllMapLines.size(); iML++) {
            if (vpAllMapLines[iML]) {
                MapLine *pML = vpAllMapLines[iML];
                pML->SetWorldPos(pML->GetWorldPos() * invMedianDepth);
            }
        }

        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);

        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;

        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mvpLocalMapLines = mpMap->GetAllMapLines();
        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;

        mLastFrame = Frame(mCurrentFrame);

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->SetReferenceMapLines(mvpLocalMapLines);

        mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mState = OK;
    }

#endif

    void Tracking::CheckReplacedInLastFrame() {
        for (int i = 0; i < mLastFrame.N; i++) {
            MapPoint *pMP = mLastFrame.mvpMapPoints[i];

            if (pMP) {
                MapPoint *pRep = pMP->GetReplaced();
                if (pRep) {
                    mLastFrame.mvpMapPoints[i] = pRep;
                }
            }
        }

        for (int i = 0; i < mLastFrame.NL; i++) {
            MapLine *pML = mLastFrame.mvpMapLines[i];

            if (pML) {
                MapLine *pReL = pML->GetReplaced();
                if (pReL) {
                    mLastFrame.mvpMapLines[i] = pReL;
                }
            }
        }

//        for (int i = 0; i < mLastFrame.mnPlaneNum; i++) {
//            MapPlane *pMP = mLastFrame.mvpMapPlanes[i];
//
//            if (pMP) {
//                MapPlane *pRep = pMP->GetReplaced();
//                if (pRep) {
//                    mLastFrame.mvpMapPlanes[i] = pRep;
//                }
//            }
//        }
    }

    bool Tracking::DetectManhattan() {

        auto verTh = Config::Get<double>("Plane.MFVerticalThreshold");
        KeyFrame * pKFCandidate = nullptr;
        int maxScore = 0;
        cv::Mat pMFc1, pMFc2, pMFc3, pMFm1, pMFm2, pMFm3;
        fullManhattanFound = false;

        int id1, id2, id3 = -1;

        for (size_t i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            cv::Mat p3Dc1 = mCurrentFrame.mvPlaneCoefficients[i];
            MapPlane *pMP1 = mCurrentFrame.mvpMapPlanes[i];

            if (!pMP1 || pMP1->isBad()) {
                continue;
            }

            //cout << "MF detection id1: " << pMP1->mnId << endl;

            for (size_t j = i + 1; j < mCurrentFrame.mnPlaneNum; j++) {
                cv::Mat p3Dc2 = mCurrentFrame.mvPlaneCoefficients[j];
                MapPlane *pMP2 = mCurrentFrame.mvpMapPlanes[j];

                if (!pMP2 || pMP2->isBad()) {
                    continue;
                }

                //cout << "MF detection id2: " << pMP2->mnId << endl;

                float angle12 = p3Dc1.at<float>(0) * p3Dc2.at<float>(0) +
                              p3Dc1.at<float>(1) * p3Dc2.at<float>(1) +
                              p3Dc1.at<float>(2) * p3Dc2.at<float>(2);

                if (angle12 > verTh || angle12 < -verTh) {
                    continue;
                }

                //cout << "MF detection angle12: " << angle12 << endl;

                for (size_t k = j+1; k < mCurrentFrame.mnPlaneNum; k++) {
                    cv::Mat p3Dc3 = mCurrentFrame.mvPlaneCoefficients[k];
                    MapPlane *pMP3 = mCurrentFrame.mvpMapPlanes[k];

                    if (!pMP3 || pMP3->isBad()) {
                        continue;
                    }

                    //cout << "MF detection id3: " << pMP3->mnId << endl;

                    float angle13 = p3Dc1.at<float>(0) * p3Dc3.at<float>(0) +
                                    p3Dc1.at<float>(1) * p3Dc3.at<float>(1) +
                                    p3Dc1.at<float>(2) * p3Dc3.at<float>(2);

                    float angle23 = p3Dc2.at<float>(0) * p3Dc3.at<float>(0) +
                                    p3Dc2.at<float>(1) * p3Dc3.at<float>(1) +
                                    p3Dc2.at<float>(2) * p3Dc3.at<float>(2);

                    if (angle13 > verTh || angle13 < -verTh || angle23 > verTh || angle23 < -verTh) {
                        continue;
                    }

                    //cout << "MF detection angle13: " << angle13 << endl;
                    //cout << "MF detection angle23: " << angle23 << endl;

                    KeyFrame* pKF = mpMap->GetManhattanObservation(pMP1, pMP2, pMP3);

                    if (!pKF) {
                        continue;
                    }

                    auto idx1 = pMP1->GetIndexInKeyFrame(pKF);
                    auto idx2 = pMP2->GetIndexInKeyFrame(pKF);
                    auto idx3 = pMP3->GetIndexInKeyFrame(pKF);

                    if (idx1 == -1 || idx2 == -1 || idx3 == -1) {
                        continue;
                    }

                    int score = pKF->mvPlanePoints[idx1].size() +
                                pKF->mvPlanePoints[idx2].size() +
                                pKF->mvPlanePoints[idx3].size() +
                                mCurrentFrame.mvPlanePoints[i].size() +
                                mCurrentFrame.mvPlanePoints[j].size() +
                                mCurrentFrame.mvPlanePoints[k].size();

                    if (score > maxScore) {
                        maxScore = score;

                        pKFCandidate = pKF;
                        pMFc1 = p3Dc1;
                        pMFc2 = p3Dc2;
                        pMFc3 = p3Dc3;
                        pMFm1 = pKF->mvPlaneCoefficients[idx1];
                        pMFm2 = pKF->mvPlaneCoefficients[idx2];
                        pMFm3 = pKF->mvPlaneCoefficients[idx3];

                        id1 = pMP1->mnId;
                        id2 = pMP2->mnId;
                        id3 = pMP3->mnId;

                        fullManhattanFound = true;
                        //cout << "Full MF detection found!" << endl;
                    }
                }

                KeyFrame* pKF = mpMap->GetPartialManhattanObservation(pMP1, pMP2);

                if (!pKF) {
                    continue;
                }

                auto idx1 = pMP1->GetIndexInKeyFrame(pKF);
                auto idx2 = pMP2->GetIndexInKeyFrame(pKF);

                if (idx1 == -1 || idx2 == -1) {
                    continue;
                }

                int score = pKF->mvPlanePoints[idx1].size() +
                            pKF->mvPlanePoints[idx2].size() +
                            mCurrentFrame.mvPlanePoints[i].size() +
                            mCurrentFrame.mvPlanePoints[j].size();

                if (score > maxScore) {
                    maxScore = score;

                    pKFCandidate = pKF;
                    pMFc1 = p3Dc1;
                    pMFc2 = p3Dc2;
                    pMFm1 = pKF->mvPlaneCoefficients[idx1];
                    pMFm2 = pKF->mvPlaneCoefficients[idx2];

                    id1 = pMP1->mnId;
                    id2 = pMP2->mnId;

                    fullManhattanFound = false;
                    //cout << "Partial MF detection found!" << endl;
                }
            }
        }

        if (pKFCandidate==nullptr) {
            return false;
        }

        //cout << "Manhattan found!" << endl;

        //cout << "Ref MF frame id: " << pKFCandidate->mnFrameId<< endl;

        //cout << "Manhattan id1: " << id1 << endl;
        //cout << "Manhattan id2: " << id2 << endl;

        if (!fullManhattanFound) {
            cv::Mat pMFc1n = (cv::Mat_<float>(3, 1) << pMFc1.at<float>(0), pMFc1.at<float>(1), pMFc1.at<float>(2));
            cv::Mat pMFc2n = (cv::Mat_<float>(3, 1) << pMFc2.at<float>(0), pMFc2.at<float>(1), pMFc2.at<float>(2));
            pMFc3 = pMFc1n.cross(pMFc2n);

            cv::Mat pMFm1n = (cv::Mat_<float>(3, 1) << pMFm1.at<float>(0), pMFm1.at<float>(1), pMFm1.at<float>(2));
            cv::Mat pMFm2n = (cv::Mat_<float>(3, 1) << pMFm2.at<float>(0), pMFm2.at<float>(1), pMFm2.at<float>(2));
            pMFm3 = pMFm1n.cross(pMFm2n);
        } else {
            //cout << "Manhattan id3: " << id3 << endl;
        }

//        cout << "Manhattan pMFc1: " << pMFc1.t() << endl;
//        cout << "Manhattan pMFc2: " << pMFc2.t() << endl;
//        cout << "Manhattan pMFc3: " << pMFc3.t() << endl;
//
//        cout << "Manhattan pMFm1: " << pMFm1.t() << endl;
//        cout << "Manhattan pMFm2: " << pMFm2.t() << endl;
//        cout << "Manhattan pMFm3: " << pMFm3.t() << endl;

        cv::Mat MFc, MFm;
        MFc = cv::Mat::eye(cv::Size(3, 3), CV_32F);
        MFm = cv::Mat::eye(cv::Size(3, 3), CV_32F);

        MFc.at<float>(0, 0) = pMFc1.at<float>(0);
        MFc.at<float>(1, 0) = pMFc1.at<float>(1);
        MFc.at<float>(2, 0) = pMFc1.at<float>(2);
        MFc.at<float>(0, 1) = pMFc2.at<float>(0);
        MFc.at<float>(1, 1) = pMFc2.at<float>(1);
        MFc.at<float>(2, 1) = pMFc2.at<float>(2);
        MFc.at<float>(0, 2) = pMFc3.at<float>(0);
        MFc.at<float>(1, 2) = pMFc3.at<float>(1);
        MFc.at<float>(2, 2) = pMFc3.at<float>(2);

        if (!fullManhattanFound && std::abs(cv::determinant(MFc) + 1) < 0.5) {
            MFc.at<float>(0, 2) = -pMFc3.at<float>(0);
            MFc.at<float>(1, 2) = -pMFc3.at<float>(1);
            MFc.at<float>(2, 2) = -pMFc3.at<float>(2);
        }

        cv::Mat Uc, Wc, VTc;

        cv::SVD::compute(MFc, Wc, Uc, VTc);

        MFc = Uc * VTc;

        MFm.at<float>(0, 0) = pMFm1.at<float>(0);
        MFm.at<float>(1, 0) = pMFm1.at<float>(1);
        MFm.at<float>(2, 0) = pMFm1.at<float>(2);
        MFm.at<float>(0, 1) = pMFm2.at<float>(0);
        MFm.at<float>(1, 1) = pMFm2.at<float>(1);
        MFm.at<float>(2, 1) = pMFm2.at<float>(2);
        MFm.at<float>(0, 2) = pMFm3.at<float>(0);
        MFm.at<float>(1, 2) = pMFm3.at<float>(1);
        MFm.at<float>(2, 2) = pMFm3.at<float>(2);

        if (!fullManhattanFound && std::abs(cv::determinant(MFm) + 1) < 0.5) {
            MFm.at<float>(0, 2) = -pMFm3.at<float>(0);
            MFm.at<float>(1, 2) = -pMFm3.at<float>(1);
            MFm.at<float>(2, 2) = -pMFm3.at<float>(2);
        }

        cv::Mat Um, Wm, VTm;

        cv::SVD::compute(MFm, Wm, Um, VTm);

        MFm = Um * VTm;

        //cout << "MFc: " << MFc << endl;
        //cout << "MFm: " << MFm << endl;

        cv::Mat Rwc = pKFCandidate->GetPoseInverse().rowRange(0,3).colRange(0,3) * MFm * MFc.t();
        manhattanRcw = Rwc.t();

        return true;
    }

    bool Tracking::TranslationEstimation() {

        // Compute Bag of Words vector
        mCurrentFrame.ComputeBoW();

        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.7, true);
        LSDmatcher lmatcher;
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        vector<MapPoint *> vpMapPointMatches;
        vector<MapLine *> vpMapLineMatches;
        vector<pair<int, int>> vLineMatches;

        int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);
        int lmatches = lmatcher.SearchByDescriptor(mpReferenceKF, mCurrentFrame, vpMapLineMatches);
        int planeMatches = pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

        int initialMatches = nmatches + lmatches + planeMatches;

        cout << "TranslationEstimation: Before: Point Matches: " << nmatches << " , Line Matches:"
             << lmatches << ", Plane Matches:" << planeMatches << endl;

        if (initialMatches < 5) {
            cout << "TranslationEstimation: Before: Not enough matches" << endl;
            return false;
        }

        mCurrentFrame.mvpMapPoints = vpMapPointMatches;
        mCurrentFrame.mvpMapLines = vpMapLineMatches;

        manhattanRcw.copyTo(mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3));

        //cout << "translation reference,pose before opti" << mCurrentFrame.mTcw << endl;
        Optimizer::TranslationOptimization(&mCurrentFrame);
        //cout << "translation reference,pose after opti" << mCurrentFrame.mTcw << endl;

        float nmatchesMap = 0;
        int nmatchesLineMap = 0;
        int nmatchesPlaneMap = 0;

        // Discard outliers
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
//                    nmatches--;

                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (mCurrentFrame.mvbLineOutlier[i]) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    mCurrentFrame.mvbLineOutlier[i] = false;
                    pML->mbTrackInView = false;
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    lmatches--;
                } else if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                    nmatchesLineMap++;

            }
        }

        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
                    planeMatches--;
                } else
                    nmatchesPlaneMap++;
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

        cout << "TranslationEstimation: After: Matches: " << nmatchesMap << " , Line Matches:"
             << nmatchesLineMap << " , Plane Matches:" << nmatchesPlaneMap << endl;

        float finalMatches = nmatchesMap + nmatchesLineMap + nmatchesPlaneMap;

        mnMatchesInliers = finalMatches;

        double ratioThresh = Config::Get<double>("MFTrackingThreshold");

        if (nmatchesMap < 3 || nmatchesMap / nmatches < ratioThresh) {
            //cout << "TranslationEstimation: After: Not enough matches" << endl;
            return false;
        }

        return true;
    }

    bool Tracking::TranslationWithMotionModel() {
        ORBmatcher matcher(0.9, true);
        LSDmatcher lmatcher;
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        UpdateLastFrame();

        // Project points seen in previous frame
        int th;
        if (mSensor != System::STEREO)
            th = 15;
        else
            th = 7;
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

        fill(mCurrentFrame.mvpMapLines.begin(), mCurrentFrame.mvpMapLines.end(), static_cast<MapLine *>(NULL));
        int lmatches = lmatcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

//        vector<MapLine *> vpMapLineMatches;
//        int lmatches = lmatcher.SearchByDescriptor(mpReferenceKF, mCurrentFrame, vpMapLineMatches);
//        mCurrentFrame.mvpMapLines = vpMapLineMatches;

        if (nmatches < 40) {
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
            nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 4 * th, mSensor == System::MONOCULAR);
        }

        int planeMatches = pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

        int initialMatches = nmatches + lmatches + planeMatches;

        //cout << "TranslationWithMotionModel: Before: Point matches: " << nmatches << " , Line Matches:"
         //    << lmatches << ", Plane Matches:" << planeMatches << endl;

        if (initialMatches < 5) {
            //cout << "TranslationWithMotionModel: Before: Not enough matches" << endl;
            return false;
        }

        manhattanRcw.copyTo(mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3));

        // Optimize frame pose with all matches
        //cout << "translation motion model,pose before opti" << mCurrentFrame.mTcw << endl;
        Optimizer::TranslationOptimization(&mCurrentFrame);
        //cout << "translation motion model,pose after opti" << mCurrentFrame.mTcw << endl;

        // Discard outliers
        float nmatchesMap = 0;
        int nmatchesLineMap = 0;
        int nmatchesPlaneMap = 0;

        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
//                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (mCurrentFrame.mvbLineOutlier[i]) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    mCurrentFrame.mvbLineOutlier[i] = false;
                    pML->mbTrackInView = false;
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    lmatches--;
                } else if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                    nmatchesLineMap++;
            }

        }

        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
                    planeMatches--;
                } else
                    nmatchesPlaneMap++;
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

        //cout << "TranslationWithMotionModel: After: Matches: " << nmatchesMap << " , Line Matches:"
        //     << nmatchesLineMap << ", Plane Matches:" << nmatchesPlaneMap << endl;

        if (mbOnlyTracking) {
            mbVO = nmatchesMap < 10;
            return nmatches > 20;
        }

        float finalMatches = nmatchesMap + nmatchesLineMap + nmatchesPlaneMap;

        mnMatchesInliers = finalMatches;

        double ratioThresh = Config::Get<double>("MFTrackingThreshold");

        if (nmatchesMap < 3 || nmatchesMap / nmatches < ratioThresh) {
            //cout << "TranslationWithMotionModel: After: Not enough matches" << endl;
            return false;
        }

        return true;
    }

    void Tracking::UpdateLastFrame() {
        // Update pose according to reference keyframe
        KeyFrame *pRef = mLastFrame.mpReferenceKF;
        cv::Mat Tlr = mlRelativeFramePoses.back();

        mLastFrame.SetPose(Tlr * pRef->GetPose());

        if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || !mbOnlyTracking)
            return;

        // Create "visual odometry" MapPoints
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float, int> > vDepthIdx;
        vDepthIdx.reserve(mLastFrame.N);
        for (int i = 0; i < mLastFrame.N; i++) {
            float z = mLastFrame.mvDepth[i];
            if (z > 0) {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        sort(vDepthIdx.begin(), vDepthIdx.end());

        // We insert all close points (depth<mThDepth)
        // If less than 100 close points, we insert the 100 closest ones.
        int nPoints = 0;
        for (size_t j = 0; j < vDepthIdx.size(); j++) {
            int i = vDepthIdx[j].second;

            bool bCreateNew = false;

            MapPoint *pMP = mLastFrame.mvpMapPoints[i];
            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1) {
                bCreateNew = true;
            }

            if (bCreateNew) {
                cv::Mat x3D = mLastFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

                mLastFrame.mvpMapPoints[i] = pNewMP;

                mlpTemporalPoints.push_back(pNewMP);
                nPoints++;
            } else {
                nPoints++;
            }

            if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                break;
        }


        // Create "visual odometry" MapLines
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float, int> > vLineDepthIdx;
        vLineDepthIdx.reserve(mLastFrame.NL);
        int nLines = 0;
        for (int i = 0; i < mLastFrame.NL; i++) {
            pair<float,float> z = mLastFrame.mvDepthLine[i];

            if (z.first > 0 && z.second > 0) {
                bool bCreateNew = false;
                vLineDepthIdx.push_back(make_pair(min(z.first, z.second), i));
                MapLine *pML = mLastFrame.mvpMapLines[i];
                if (!pML)
                    bCreateNew = true;
                else if (pML->Observations() < 1) {
                    bCreateNew = true;
                }
                if (bCreateNew) {
                    Vector6d line3D = mLastFrame.obtain3DLine(i, mImDepth);
                    if (line3D == static_cast<Vector6d>(NULL)) {
                        continue;
                    }
                    MapLine *pNewML = new MapLine(line3D, mpMap, &mLastFrame, i);

                    mLastFrame.mvpMapLines[i] = pNewML;

                    mlpTemporalLines.push_back(pNewML);
                    nLines++;
                } else {
                    nLines++;
                }

                if (nLines > 30)
                    break;

            }
        }
    }

    bool Tracking::TrackReferenceKeyFrame() {

        // Compute Bag of Words vector
        mCurrentFrame.ComputeBoW();
        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.7, true);
        LSDmatcher lmatcher;
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        vector<MapPoint *> vpMapPointMatches;
        vector<MapLine *> vpMapLineMatches;
        vector<pair<int, int>> vLineMatches;

        int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);
        int lmatches = lmatcher.SearchByDescriptor(mpReferenceKF, mCurrentFrame, vpMapLineMatches);
        int planeMatches = pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

        float initialMatches = nmatches + lmatches + planeMatches;

//        cout << "TrackReferenceKeyFrame: Before: Point matches: " << nmatches << " , Line Matches:"
//             << lmatches << ", Plane Matches:" << planeMatches << endl;

        if (initialMatches < 5) {
//            cout << "TrackReferenceKeyFrame: Before: Not enough matches" << endl;
            return false;
        }

        mCurrentFrame.mvpMapLines = vpMapLineMatches;
        mCurrentFrame.mvpMapPoints = vpMapPointMatches;


//        cout << "tracking reference,pose before opti" << mCurrentFrame.mTcw << endl;
        Optimizer::PoseOptimization(&mCurrentFrame);
//        cout << "tracking reference,pose after opti" << mCurrentFrame.mTcw << endl;

        // Discard outliers

        float nmatchesMap = 0;
        int nmatchesLineMap = 0;
        int nmatchesPlaneMap = 0;

        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (mCurrentFrame.mvbLineOutlier[i]) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    mCurrentFrame.mvbLineOutlier[i] = false;
                    pML->mbTrackInView = false;
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    lmatches--;
                } else if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                    nmatchesLineMap++;

            }
        }

        int nDiscardPlane = 0;
        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
                    planeMatches--;
                    nDiscardPlane++;
                } else
                    nmatchesPlaneMap++;
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

//        cout << "TrackReferenceKeyFrame: After: Matches: " << nmatchesMap << " , Line Matches:"
//             << nmatchesLineMap << ", Plane Matches:" << nmatchesPlaneMap << endl;

        float finalMatches = nmatchesMap + nmatchesLineMap + nmatchesPlaneMap;

//        if (finalMatches < 10) {
        if (nmatchesMap < 3 || nmatchesMap / nmatches < 0.1) {
//            cout << "TrackReferenceKeyFrame: After: Not enough matches" << endl;
            return false;
        }

        return true;
    }

    bool Tracking::TrackWithMotionModel() {
        ORBmatcher matcher(0.9, true);
        LSDmatcher lmatcher;
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        UpdateLastFrame();

        // Project points seen in previous frame
        int th;
        if (mSensor != System::STEREO)
            th = 15;
        else
            th = 7;
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

        fill(mCurrentFrame.mvpMapLines.begin(), mCurrentFrame.mvpMapLines.end(), static_cast<MapLine *>(NULL));
        int lmatches = lmatcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

//        vector<MapLine *> vpMapLineMatches;
//        int lmatches = lmatcher.SearchByDescriptor(mpReferenceKF, mCurrentFrame, vpMapLineMatches);
//        mCurrentFrame.mvpMapLines = vpMapLineMatches;

        // If few matches, uses a wider window search
        if (nmatches < 40) {
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
                 static_cast<MapPoint *>(NULL));
            nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 4 * th, mSensor == System::MONOCULAR);
        }

        int planeMatches = pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

        int initialMatches = nmatches + lmatches + planeMatches;

//        cout << "TrackWithMotionModel: Before: Point matches: " << nmatches << " , Line Matches:"
//             << lmatches << ", Plane Matches:" << planeMatches << endl;

        if (initialMatches < 5) {
//            cout << "TrackWithMotionModel: Before: Not enough matches" << endl;
            return false;
        }

        // Optimize frame pose with all matches
//        cout << "tracking motion model,pose before opti" << mCurrentFrame.mTcw << endl;
        Optimizer::PoseOptimization(&mCurrentFrame);
//        cout << "tracking motion model,pose after opti" << mCurrentFrame.mTcw << endl;

        // Discard outliers
        float nmatchesMap = 0;
        int nmatchesLineMap = 0;
        int nmatchesPlaneMap = 0;

        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (mCurrentFrame.mvbLineOutlier[i]) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    mCurrentFrame.mvbLineOutlier[i] = false;
                    pML->mbTrackInView = false;
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    lmatches--;
                } else if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                    nmatchesLineMap++;
            }

        }

        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
                    planeMatches--;
                } else
                    nmatchesPlaneMap++;
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

//        cout << "TrackWithMotionModel: After: Matches: " << nmatchesMap << " , Line Matches:"
//             << nmatchesLineMap << ", Plane Matches:" << nmatchesPlaneMap << endl;

        float finalMatches = nmatchesMap + nmatchesLineMap + nmatchesPlaneMap;

        mnMatchesInliers = finalMatches;

        if (mbOnlyTracking) {
            mbVO = nmatchesMap < 10;
            return nmatches > 20;
        }

//        if (finalMatches < 10) {
        if (nmatchesMap < 3 || nmatchesMap / nmatches < 0.1) {
//            cout << "TrackWithMotionModel: After: Not enough matches" << endl;
            return false;
        }

        return true;
    }

    bool Tracking::TrackLocalMap() {
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        UpdateLocalMap();

        thread threadPoints(&Tracking::SearchLocalPoints, this);
        thread threadLines(&Tracking::SearchLocalLines, this);
        thread threadPlanes(&Tracking::SearchLocalPlanes, this);
        threadPoints.join();
        threadLines.join();
        threadPlanes.join();

        pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());
//        mpMap->ComputeCrossLine(mpMap->GetAllMapPlanes(), 2.8, 0.5);
//        cout << "tracking localmap, pose before opti" << endl << mCurrentFrame.mTcw << endl;
        Optimizer::PoseOptimization(&mCurrentFrame);
//        cout << "tracking localmap, pose after opti" << mCurrentFrame.mTcw << endl;

        mnMatchesInliers = 0;

        // Update MapPoints Statistics
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (!mCurrentFrame.mvbOutlier[i]) {
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                    if (!mbOnlyTracking) {
                        if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                            mnMatchesInliers++;
                    } else
                        mnMatchesInliers++;
                } else if (mSensor == System::STEREO)
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);

            }
        }

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (!mCurrentFrame.mvbLineOutlier[i]) {
                    mCurrentFrame.mvpMapLines[i]->IncreaseFound();
                    if (!mbOnlyTracking) {
                        if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                            mnMatchesInliers++;
                    } else
                        mnMatchesInliers++;
                } else if (mSensor == System::STEREO)
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
            }
        }

        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
                } else {
                    mCurrentFrame.mvpMapPlanes[i]->IncreaseFound();
                    mnMatchesInliers++;
                }
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

//        cout << "TrackLocalMap: After: Matches: " << mnMatchesInliers << endl;

        // Decide if the tracking was succesful
        // More restrictive if there was a relocalization recently
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 20) {
//            cout << "TrackLocalMap: After: Not enough matches" << endl;
            return false;
        }

        if (mnMatchesInliers < 5) {
//            cout << "TrackLocalMapWithLines: After: Not enough matches" << endl;
            return false;
        } else
            return true;
    }


    bool Tracking::NeedNewKeyFrame() {
        if (mbOnlyTracking)
            return false;

// If Local Mapping is freezed by a Loop Closure do not insert keyframes
        if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
            return false;

        const int nKFs = mpMap->KeyFramesInMap();

// Do not insert keyframes if not enough frames have passed from last relocalisation
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
            return false;

// Tracked MapPoints in the reference keyframe
        int nMinObs = 3;
        if (nKFs <= 2)
            nMinObs = 2;
        int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

// Local Mapping accept keyframes?
        bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

// Stereo & RGB-D: Ratio of close "matches to map"/"total matches"
// "total matches = matches to map + visual odometry matches"
// Visual odometry matches will become MapPoints if we insert a keyframe.
// This ratio measures how many MapPoints we could create if we insert a keyframe.
        int nMap = 0; //nTrackedClose
        int nTotal = 0;
        int nNonTrackedClose = 0;
        if (mSensor != System::MONOCULAR) {
            for (int i = 0; i < mCurrentFrame.N; i++) {
                if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth) {
                    nTotal++;
                    if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                        nMap++;
                    else
                        nNonTrackedClose++;
                }
            }
        } else {
            // There are no visual odometry matches in the monocular case
            nMap = 1;
            nTotal = 1;
        }

        const float ratioMap = (float) nMap / fmax(1.0f, nTotal);

// Thresholds
        float thRefRatio = 0.75f;
        if (nKFs < 2)
            thRefRatio = 0.4f;

        if (mSensor == System::MONOCULAR)
            thRefRatio = 0.9f;

        float thMapRatio = 0.35f;
        if (mnMatchesInliers > 300)
            thMapRatio = 0.20f;

// Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
        const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
// Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
        const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
//Condition 1c: tracking is weak
        const bool c1c = mSensor != System::MONOCULAR && (mnMatchesInliers < nRefMatches * 0.25 || ratioMap < 0.3f);
// Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
        const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || ratioMap < thMapRatio) &&
                         mnMatchesInliers > 15);

        if (((c1a || c1b || c1c) && c2) || mCurrentFrame.mbNewPlane) {
            // If the mapping accepts keyframes, insert keyframe.
            // Otherwise send a signal to interrupt BA
            if (bLocalMappingIdle) {
                return true;
            } else {
                mpLocalMapper->InterruptBA();
                if (mSensor != System::MONOCULAR) {
                    if (mpLocalMapper->KeyframesInQueue() < 3)
                        return true;
                    else
                        return false;
                } else
                    return false;
            }
        }

        return false;
    }

    void Tracking::CreateNewKeyFrame() {
        if (!mpLocalMapper->SetNotStop(true))
            return;

        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;

//        cout << "New Keyframe!" << endl;

        if (mSensor != System::MONOCULAR) {

            mCurrentFrame.UpdatePoseMatrices();

            // We sort points by the measured depth by the stereo/RGBD sensor.
            // We create all those MapPoints whose depth < mThDepth.
            // If there are less than 100 close points we create the 100 closest.
            vector<pair<float, int> > vDepthIdx;
            vDepthIdx.reserve(mCurrentFrame.N);

            for (int i = 0; i < mCurrentFrame.N; i++) {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0) {
                    vDepthIdx.push_back(make_pair(z, i));
                }
            }

            if (!vDepthIdx.empty()) {
                sort(vDepthIdx.begin(), vDepthIdx.end());

                int nPoints = 0;
                for (size_t j = 0; j < vDepthIdx.size(); j++) {
                    int i = vDepthIdx[j].second;

                    bool bCreateNew = false;

                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (!pMP)
                        bCreateNew = true;
                    else if (pMP->Observations() < 1) {
                        bCreateNew = true;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }

                    if (bCreateNew) {
                        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                        MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap);
                        pNewMP->AddObservation(pKF, i);
                        pKF->AddMapPoint(pNewMP, i);
                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpMap->AddMapPoint(pNewMP);

                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                        nPoints++;
                    } else {
                        nPoints++;
                    }

                    if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                        break;
                }
            }

            vector<pair<float, int>> vLineDepthIdx;
            vLineDepthIdx.reserve(mCurrentFrame.NL);

            for (int i = 0; i < mCurrentFrame.NL; i++) {
                pair<float,float> z = mCurrentFrame.mvDepthLine[i];
                if (z.first > 0 && z.second > 0) {
                    vLineDepthIdx.push_back(make_pair(min(z.first, z.second), i));
                }
            }

            if (!vLineDepthIdx.empty()) {
                sort(vLineDepthIdx.begin(),vLineDepthIdx.end());

                int nLines = 0;
                for (size_t j = 0; j < vLineDepthIdx.size(); j++) {
                    int i = vLineDepthIdx[j].second;

                    bool bCreateNew = false;

                    MapLine *pMP = mCurrentFrame.mvpMapLines[i];
                    if (!pMP)
                        bCreateNew = true;
                    else if (pMP->Observations() < 1) {
                        bCreateNew = true;
                        mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    }

                    if (bCreateNew) {
                        Vector6d line3D = mCurrentFrame.obtain3DLine(i, mImDepth);
                        if (line3D == static_cast<Vector6d>(NULL)) {
                            continue;
                        }
                        MapLine *pNewML = new MapLine(line3D, pKF, mpMap);
                        pNewML->AddObservation(pKF, i);
                        pKF->AddMapLine(pNewML, i);
                        pNewML->ComputeDistinctiveDescriptors();
                        pNewML->UpdateAverageDir();
                        mpMap->AddMapLine(pNewML);
                        mCurrentFrame.mvpMapLines[i] = pNewML;
                        nLines++;
                    } else {
                        nLines++;
                    }

                    if (nLines > 30)
                        break;
                }
            }

            for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
                if (mCurrentFrame.mvpParallelPlanes[i] && !mCurrentFrame.mvbPlaneOutlier[i]) {
//                if (mCurrentFrame.mvpParallelPlanes[i]) {
                    mCurrentFrame.mvpParallelPlanes[i]->AddParObservation(pKF, i);
                }
                if (mCurrentFrame.mvpVerticalPlanes[i] && !mCurrentFrame.mvbPlaneOutlier[i]) {
//                if (mCurrentFrame.mvpVerticalPlanes[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i]->AddVerObservation(pKF, i);
                }

                if (mCurrentFrame.mvpMapPlanes[i]) {
                    mCurrentFrame.mvpMapPlanes[i]->AddObservation(pKF, i);
                    continue;
                }

                if (mCurrentFrame.mvbPlaneOutlier[i]) {
//                    mCurrentFrame.mvbPlaneOutlier[i] = false;
                    continue;
                }

//                if (mCurrentFrame.mvpMapPlanes[i] || mCurrentFrame.mvbPlaneOutlier[i]) {
//                    continue;
//                }
//
//                if (mCurrentFrame.mvpParallelPlanes[i]) {
//                    mCurrentFrame.mvpParallelPlanes[i]->AddParObservation(pKF, i);
//                }
//                if (mCurrentFrame.mvpVerticalPlanes[i]) {
//                    mCurrentFrame.mvpVerticalPlanes[i]->AddVerObservation(pKF, i);
//                }

                pKF->SetNotErase();

                cv::Mat p3D = mCurrentFrame.ComputePlaneWorldCoeff(i);
                MapPlane *pNewMP = new MapPlane(p3D, pKF, mpMap);
                pNewMP->AddObservation(pKF,i);
                pKF->AddMapPlane(pNewMP, i);
                pNewMP->UpdateCoefficientsAndPoints();
//                mpMap->ComputeCrossLine(mpMap->GetAllMapPlanes(), 2.8, 0.5);
                // pNewMP->UpdateComputePlaneBoundary();
                mpMap->AddMapPlane(pNewMP);
//                mpMap->ComputeCrossLine(mpMap->GetAllMapPlanes(), 51, 0.00002);
            }
//            mpPointCloudMapping->print();
        }

        mpLocalMapper->InsertKeyFrame(pKF);

        mpLocalMapper->SetNotStop(false);

        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKF;
    }

    void Tracking::SearchLocalPoints() {
// Do not search map points already matched
        for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end();
             vit != vend; vit++) {
            MapPoint *pMP = *vit;
            if (pMP) {
                if (pMP->isBad()) {
                    *vit = static_cast<MapPoint *>(NULL);
                } else {
                    pMP->IncreaseVisible();
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    pMP->mbTrackInView = false;
                }
            }
        }

        int nToMatch = 0;

// Project points in frame and check its visibility
        for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end();
             vit != vend; vit++) {
            MapPoint *pMP = *vit;
            if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if (pMP->isBad())
                continue;
            // Project (this fills MapPoint variables for matching)
            if (mCurrentFrame.isInFrustum(pMP, 0.5)) {
                pMP->IncreaseVisible();
                nToMatch++; //将要match的
            }
        }

        if (nToMatch > 0) {
            ORBmatcher matcher(0.8);
            int th = 1;
            if (mSensor == System::RGBD)
                th = 3;
            // If the camera has been relocalised recently, perform a coarser search
            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                th = 5;
            matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
        }
    }

    void Tracking::SearchLocalLines() {
        for (vector<MapLine *>::iterator vit = mCurrentFrame.mvpMapLines.begin(), vend = mCurrentFrame.mvpMapLines.end();
             vit != vend; vit++) {
            MapLine *pML = *vit;
            if (pML) {
                if (pML->isBad()) {
                    *vit = static_cast<MapLine *>(NULL);
                } else {
                    pML->IncreaseVisible();
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    pML->mbTrackInView = false;
                }
            }
        }

        int nToMatch = 0;

        for (vector<MapLine *>::iterator vit = mvpLocalMapLines.begin(), vend = mvpLocalMapLines.end();
             vit != vend; vit++) {
            MapLine *pML = *vit;

            if (pML->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if (pML->isBad())
                continue;

            if (mCurrentFrame.isInFrustum(pML, 0.6)) {
                pML->IncreaseVisible();
                nToMatch++;
            }
        }

        if (nToMatch > 0) {
            LSDmatcher matcher;
            int th = 1;

            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                th = 5;

            matcher.SearchByProjection(mCurrentFrame, mvpLocalMapLines, th);
        }
    }

    void Tracking::SearchLocalPlanes() {
        for (vector<MapPlane *>::iterator vit = mCurrentFrame.mvpMapPlanes.begin(), vend = mCurrentFrame.mvpMapPlanes.end();
             vit != vend; vit++) {
            MapPlane *pMP = *vit;
            if (pMP) {
                if (pMP->isBad()) {
                    *vit = static_cast<MapPlane *>(NULL);
                } else {
                    pMP->IncreaseVisible();
                }
            }
        }
    }


    void Tracking::UpdateLocalMap() {
// This is for visualization
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->SetReferenceMapLines(mvpLocalMapLines);

// Update
        UpdateLocalKeyFrames();

        UpdateLocalPoints();
        UpdateLocalLines();
    }

    void Tracking::UpdateLocalLines() {
        mvpLocalMapLines.clear();

        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            KeyFrame *pKF = *itKF;
            const vector<MapLine *> vpMLs = pKF->GetMapLineMatches();

            for (vector<MapLine *>::const_iterator itML = vpMLs.begin(), itEndML = vpMLs.end();
                 itML != itEndML; itML++) {
                MapLine *pML = *itML;
                if (!pML)
                    continue;
                if (pML->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                if (!pML->isBad()) {
                    mvpLocalMapLines.push_back(pML);
                    pML->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }

    void Tracking::UpdateLocalPoints() {
        mvpLocalMapPoints.clear();

        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            KeyFrame *pKF = *itKF;
            const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

            for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end();
                 itMP != itEndMP; itMP++) {
                MapPoint *pMP = *itMP;
                if (!pMP)
                    continue;
                if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                if (!pMP->isBad()) {
                    mvpLocalMapPoints.push_back(pMP);
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }


    void Tracking::UpdateLocalKeyFrames() {
// Each map point vote for the keyframes in which it has been observed
        map<KeyFrame *, int> keyframeCounter;
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (!pMP->isBad()) {
                    const map<KeyFrame *, size_t> observations = pMP->GetObservations();
                    for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end();
                         it != itend; it++)
                        keyframeCounter[it->first]++;
                } else {
                    mCurrentFrame.mvpMapPoints[i] = NULL;
                }
            }
        }

//        for (int i = 0; i < mCurrentFrame.NL; i++) {
//            if (mCurrentFrame.mvpMapLines[i]) {
//                MapLine *pML = mCurrentFrame.mvpMapLines[i];
//                if (!pML->isBad()) {
//                    const map<KeyFrame *, size_t> observations = pML->GetObservations();
//                    for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end();
//                         it != itend; it++)
//                        keyframeCounter[it->first]++;
//                } else {
//                    mCurrentFrame.mvpMapLines[i] = NULL;
//                }
//            }
//        }
//
//        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
//            if (mCurrentFrame.mvpMapPlanes[i]) {
//                MapPlane *pMP = mCurrentFrame.mvpMapPlanes[i];
//                if (!pMP->isBad()) {
//                    const map<KeyFrame *, size_t> observations = pMP->GetObservations();
//                    for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end();
//                         it != itend; it++)
//                        keyframeCounter[it->first]++;
//                } else {
//                    mCurrentFrame.mvpMapPlanes[i] = NULL;
//                }
//            }
//        }

        if (keyframeCounter.empty())
            return;

        int max = 0;
        KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

        mvpLocalKeyFrames.clear();
        mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

// All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
        for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end();
             it != itEnd; it++) {
            KeyFrame *pKF = it->first;

            if (pKF->isBad())
                continue;

            if (it->second > max) {
                max = it->second;
                pKFmax = pKF;
            }

            mvpLocalKeyFrames.push_back(it->first);
            pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }


// Include also some not-already-included keyframes that are neighbors to already-included keyframes
        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            // Limit the number of keyframes
            if (mvpLocalKeyFrames.size() > 80)
                break;

            KeyFrame *pKF = *itKF;

            const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

            for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end();
                 itNeighKF != itEndNeighKF; itNeighKF++) {
                KeyFrame *pNeighKF = *itNeighKF;
                if (!pNeighKF->isBad()) {
                    if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                        mvpLocalKeyFrames.push_back(pNeighKF);
                        pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            const set<KeyFrame *> spChilds = pKF->GetChilds();
            for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++) {
                KeyFrame *pChildKF = *sit;
                if (!pChildKF->isBad()) {
                    if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                        mvpLocalKeyFrames.push_back(pChildKF);
                        pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            KeyFrame *pParent = pKF->GetParent();
            if (pParent) {
                if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.push_back(pParent);
                    pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }

        }

        if (pKFmax) {
            mpReferenceKF = pKFmax;
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }
    }

    void Tracking::ComputeNonPlaneAreaInCurrentFrame(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> &combinedPoints) {
//        PointCloud::Ptr combinedPoints (new PointCloud());
        PointCloud::Ptr combinedNoPlanePoints(new PointCloud());
        combinedNoPlanePoints = combinedPoints;
        Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( mCurrentFrame.mTcw );
//        cout << "frame no plane observation size" << pF.mvNoPlanePoints[id].size() << endl;
        pcl::transformPointCloud(mCurrentFrame.mvNoPlanePoints, *combinedNoPlanePoints, T.inverse().matrix());
        mpMap->AddNonPlaneArea(*combinedNoPlanePoints);
        for (int i = 0; i < mpMap->GetAllMapPlanes().size(); ++i) {
            *combinedPoints += *mpMap->GetAllMapPlanes()[i]->mvNoPlanePoints;
        }
    }

    /*
     * 对整个map进行补全
     * combinedPoints：map 中的所有非平面点
     * vpOuterPlanes：最外层平面
     * outerPlaneIdx：ID
     * */
    void Tracking::ComputeCameraCenterToPlaneAndCompletion(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> &combinedPoints,
                           std::vector<MapPlane*> &vpOuterPlanes, vector<int> &outerPlaneIdx, float threshold) {
        cout << "enter compute the camera center intersection " << endl;
        auto verThreshold = Config::Get<double>("Plane.MFVerticalThreshold");
        cv::Mat CameraCenter = mCurrentFrame.GetCameraCenter();
        cout << "outer plane idx size" << " " << outerPlaneIdx.size() << endl;

        for (auto& vMP : mpMap->GetAllMapPlanes()) {
            vMP->mvCompletePoints->clear();
        }
        cout << "get camera center" << CameraCenter << endl;
        cv::Mat DirectionVector;
        cv::Mat ParameterVector;
        pcl::PointXYZRGB minPoint1, maxPoint1;

        auto AllMapPlanes = mpMap->GetAllMapPlanes();
        cout << "before size" << " " << AllMapPlanes.size() << endl;
        if (!AllMapPlanes.empty())
        {
            for (int r = 0; r < AllMapPlanes.size(); ++r) {
                if (AllMapPlanes[r]->smallFlag)
                    *combinedPoints += *AllMapPlanes[r]->mvPlanePoints;
            }
        }
        if (!vpOuterPlanes.empty())
        {
            PointCloud::Ptr sumPlanePoints (new PointCloud());
            for (auto& vMP : AllMapPlanes) {
                *sumPlanePoints += *vMP->mvPlanePoints;
            }
            mpMap->EstimateBoundingBox(sumPlanePoints, minPoint1, maxPoint1);
        }
        for (auto& p : *combinedPoints) {
            int NumOfCrossPoint = 0;
            float base = sqrt(pow((p.x - CameraCenter.at<float>(0)), 2) + pow((p.y - CameraCenter.at<float>(1)), 2) +
                              pow((p.z - CameraCenter.at<float>(2)), 2));
            DirectionVector = (cv::Mat_<float>(3, 1) << (CameraCenter.at<float>(0) - p.x) / base,
                    (CameraCenter.at<float>(1) - p.y) / base,
                    (CameraCenter.at<float>(2) - p.z) / base);
            ParameterVector = (cv::Mat_<float>(3, 1) << (CameraCenter.at<float>(0)) - p.x,
                    (CameraCenter.at<float>(1) - p.y),
                    (CameraCenter.at<float>(2) - p.z));
            float minPlaneDistance = 10;
            int minID = -1;
            pcl::PointXYZRGB minPoint;
            if (!vpOuterPlanes.empty()) {
                for (int i = 0; i < vpOuterPlanes.size(); i++) {
                    cv::Mat plane1 = vpOuterPlanes[i]->GetWorldPos();
                    auto planePoint = vpOuterPlanes[i]->mvPlanePoints->points[5];
                    float angle12 = plane1.at<float>(0, 0) * DirectionVector.at<float>(0, 0) +
                                    plane1.at<float>(1, 0) * DirectionVector.at<float>(1, 0) +
                                    plane1.at<float>(2, 0) * DirectionVector.at<float>(2, 0);
                    if (angle12 <= verThreshold && angle12 >= -verThreshold) {
                        continue;
                    } else {
//                    float B = -(plane1.at<float>(0,0)*p.x + plane1.at<float>(1,0)*p.y +
//                            plane1.at<float>(2,0)*p.z + plane1.at<float>(3,0));
                        float B = -(plane1.at<float>(0, 0) * CameraCenter.at<float>(0) +
                                    plane1.at<float>(1, 0) * CameraCenter.at<float>(1) +
                                    plane1.at<float>(2, 0) * CameraCenter.at<float>(2) + plane1.at<float>(3, 0));
                        float C = (planePoint.x - CameraCenter.at<float>(0)) * plane1.at<float>(0, 0) +
                                  (planePoint.y - CameraCenter.at<float>(1)) * plane1.at<float>(1, 0) +
                                  (planePoint.z - CameraCenter.at<float>(2)) * plane1.at<float>(2, 0);
                        float t = B / angle12;
                        float x = CameraCenter.at<float>(0) + DirectionVector.at<float>(0, 0) * t;
                        float y = CameraCenter.at<float>(1) + DirectionVector.at<float>(1, 0) * t;
                        float z = CameraCenter.at<float>(2) + DirectionVector.at<float>(2, 0) * t;
                        float result =
                                plane1.at<float>(0, 0) * x + plane1.at<float>(1, 0) * y + plane1.at<float>(2, 0) * z +
                                plane1.at<float>(3, 0);
                        float minDistance = plane1.at<float>(0, 0) * CameraCenter.at<float>(0) +
                                            plane1.at<float>(1, 0) * CameraCenter.at<float>(1) +
                                            plane1.at<float>(2, 0) * CameraCenter.at<float>(2);
                        float ParallelResultX = (CameraCenter.at<float>(0) - p.x) / (p.x - x);
                        float ParallelResultY = (CameraCenter.at<float>(1) - p.y) / (p.y - y);
                        float ParallelResultZ = (CameraCenter.at<float>(2) - p.z) / (p.z - z);
                        if (ParallelResultX > 0 && ParallelResultY > 0 && ParallelResultZ > 0) {
                            vector<double> VecOfIx;
                            NumOfCrossPoint++;
                            float distance = sqrt(pow(DirectionVector.at<float>(0, 0) * t, 2) +
                                                  pow(DirectionVector.at<float>(1, 0) * t, 2) +
                                                  pow(DirectionVector.at<float>(2, 0) * t, 2));
                            if (distance < minPlaneDistance && x >= minPoint1.x &&
                            x <= maxPoint1.x && y >= minPoint1.y && y <= maxPoint1.y &&
                            z >= minPoint1.z && z <= maxPoint1.z) {
                                minPlaneDistance = distance;
                                pcl::PointXYZRGB p;
                                p.x = x;
                                p.y = y;
                                p.z = z;
                                p.r = 0.0;
                                p.g = 0.0;
                                p.b = 1.0;
                                minPoint = p;
                                minID = i;
                            }
                        }
                    }
                }
                if (minID >= 0)
                {
                    vpOuterPlanes[minID]->mvCompletePoints->points.emplace_back(minPoint);
                }
                for (int w = 0; w < vpOuterPlanes.size(); ++w) {
                    pcl::PointXYZRGB minPoint2, maxPoint2;
                    mpMap->EstimateBoundingBox(vpOuterPlanes[w]->mvPlanePoints, minPoint2, maxPoint2);
                    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
                    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
                    if (!vpOuterPlanes[w]->mvCompletePoints->empty())
                    {
                        for (size_t t; t < vpOuterPlanes[w]->mvCompletePoints->size(); t++) {
                            auto &pCom = vpOuterPlanes[w]->mvCompletePoints->points[t];
                            if (pCom.x > minPoint2.x && pCom.x < maxPoint2.x &&
                            pCom.y > minPoint2.y && pCom.y < maxPoint2.y &&
                            pCom.z > minPoint2.z && pCom.z < maxPoint2.z)
                            {
                                inliers->indices.emplace_back();
                            }
                        }
                    }
                    if (inliers->indices.size() < vpOuterPlanes[w]->mvCompletePoints->size())
                    {
                        extract.setInputCloud(vpOuterPlanes[w]->mvCompletePoints);
                        extract.setIndices(inliers);
                        extract.setNegative(true);
                        extract.filter(*vpOuterPlanes[w]->mvCompletePoints);
                    }

                }
            }
        }
    }


    bool Tracking::Relocalization() {
        cout << "Tracking:localization" << endl;
// Compute Bag of Words Vector
        mCurrentFrame.ComputeBoW();

// Relocalization is performed when tracking is lost
// Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
        vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

        cout << "Tracking,vpCandidateKFs" << vpCandidateKFs.size() << endl;
        if (vpCandidateKFs.empty())
            return false;

        const int nKFs = vpCandidateKFs.size();

// We perform first an ORB matching with each candidate
// If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.75, true);

        vector<PnPsolver *> vpPnPsolvers;
        vpPnPsolvers.resize(nKFs);

        vector<vector<MapPoint *> > vvpMapPointMatches;
        vvpMapPointMatches.resize(nKFs);

        vector<bool> vbDiscarded;
        vbDiscarded.resize(nKFs);

        int nCandidates = 0;

        for (int i = 0; i < nKFs; i++) {
            KeyFrame *pKF = vpCandidateKFs[i];
            if (pKF->isBad())
                vbDiscarded[i] = true;
            else {
                int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
                if (nmatches < 15) {
                    vbDiscarded[i] = true;
                    continue;
                } else {
                    PnPsolver *pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                    pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
                    vpPnPsolvers[i] = pSolver;
                    nCandidates++;
                }
            }
        }

// Alternatively perform some iterations of P4P RANSAC
// Until we found a camera pose supported by enough inliers
        bool bMatch = false;
        ORBmatcher matcher2(0.9, true);

        while (nCandidates > 0 && !bMatch) {
            for (int i = 0; i < nKFs; i++) {
                if (vbDiscarded[i])
                    continue;

                // Perform 5 Ransac Iterations
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;

                PnPsolver *pSolver = vpPnPsolvers[i];
                cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

                // If Ransac reachs max. iterations discard keyframe
                if (bNoMore) {
                    vbDiscarded[i] = true;
                    nCandidates--;
                }

                // If a Camera Pose is computed, optimize
                if (!Tcw.empty()) {
                    Tcw.copyTo(mCurrentFrame.mTcw);

                    set<MapPoint *> sFound;

                    const int np = vbInliers.size();

                    for (int j = 0; j < np; j++) {
                        if (vbInliers[j]) {
                            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                            sFound.insert(vvpMapPointMatches[i][j]);
                        } else
                            mCurrentFrame.mvpMapPoints[j] = NULL;
                    }

                    int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                    if (nGood < 10)
                        continue;

                    for (int io = 0; io < mCurrentFrame.N; io++)
                        if (mCurrentFrame.mvbOutlier[io])
                            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

                    // If few inliers, search by projection in a coarse window and optimize again
                    if (nGood < 50) {
                        int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10,
                                                                      100);

                        if (nadditional + nGood >= 50) {
                            nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                            // If many inliers but still not enough, search by projection again in a narrower window
                            // the camera has been already optimized with many points
                            if (nGood > 30 && nGood < 50) {
                                sFound.clear();
                                for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                    if (mCurrentFrame.mvpMapPoints[ip])
                                        sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                                nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3,
                                                                          64);

                                // Final optimization
                                if (nGood + nadditional >= 50) {
                                    nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                    for (int io = 0; io < mCurrentFrame.N; io++)
                                        if (mCurrentFrame.mvbOutlier[io])
                                            mCurrentFrame.mvpMapPoints[io] = NULL;
                                }
                            }
                        }
                    }


                    // If the pose is supported by enough inliers stop ransacs and continue
                    if (nGood >= 50) {
                        bMatch = true;
                        break;
                    }
                }
            }

            if (!bMatch) {

            }
        }

        if (!bMatch) {
            return false;
        } else {
            mnLastRelocFrameId = mCurrentFrame.mnId;
            return true;
        }

    }

    void Tracking::UpdateOuterPlanes(std::vector<MapPlane *> &OuterPlanes) {
        mpMap->mspOuterPlanes.clear();
        for (auto vOP : OuterPlanes) {
            mpMap->mspOuterPlanes.emplace_back(vOP);
        }
    }

    void Tracking::Reset() {
        mpViewer->RequestStop();

        cout << "System Reseting" << endl;
        while (!mpViewer->isStopped())
            usleep(3000);

// Reset Local Mapping
        cout << "Reseting Local Mapper...";
        mpLocalMapper->RequestReset();
        cout << " done" << endl;

// Reset Loop Closing
        cout << "Reseting Loop Closing...";
        mpLoopClosing->RequestReset();
        cout << " done" << endl;

// Clear BoW Database
        cout << "Reseting Database...";
        mpKeyFrameDB->clear();
        cout << " done" << endl;

// Clear Map (this erase MapPoints and KeyFrames)
        mpMap->clear();

        KeyFrame::nNextId = 0;
        Frame::nNextId = 0;
        mState = NO_IMAGES_YET;

        if (mpInitializer) {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
        }

        mlRelativeFramePoses.clear();
        mlpReferences.clear();
        mlFrameTimes.clear();
        mlbLost.clear();

        mpViewer->Release();
    }

    void Tracking::ChangeCalibration(const string &strSettingPath) {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0) {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        Frame::mbInitialComputations = true;
    }

    void Tracking::InformOnlyTracking(const bool &flag) {
        mbOnlyTracking = flag;
    }


} //namespace ORB_SLAM
