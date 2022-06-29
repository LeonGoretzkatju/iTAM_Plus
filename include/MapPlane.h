//
// Created by fishmarch on 19-5-24.
//

#ifndef ORB_SLAM2_MAPPLANE_H
#define ORB_SLAM2_MAPPLANE_H

#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"
#include "Converter.h"

#include <opencv2/core/core.hpp>
#include <mutex>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/exceptions.h>

#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/features/boundary.h>
#include <math.h>
#include <boost/make_shared.hpp>

#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>


#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/features/normal_3d.h>


#include <pcl/filters/covariance_sampling.h>
#include <pcl/filters/normal_space.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/io/ply_io.h>

#include <pcl/filters/statistical_outlier_removal.h>
#include "Layout.h"

namespace ORB_SLAM2 {
    class KeyFrame;
    class Frame;
    class Map;
    class MapPlane {
        typedef pcl::PointXYZRGB PointT;
        typedef pcl::PointCloud <PointT> PointCloud;
    public:
        MapPlane(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);

        void SetWorldPos(const cv::Mat &Pos);
        cv::Mat GetWorldPos();

        void flipNormal();
        void AddLayout(Layout *pLayout);
        void EraseLayout(Layout *pLayout);

        void Replace(MapPlane* pMP);
        void ReplaceVerticalObservations(MapPlane* pMP);
        void ReplaceParallelObservations(MapPlane* pMP);
        MapPlane* GetReplaced();

        void IncreaseVisible(int n=1);
        void IncreaseFound(int n=1);
        float GetFoundRatio();
        inline int GetFound(){
            return mnFound;
        }

        void SetBadFlag();
        bool isBad();

        KeyFrame* GetReferenceKeyFrame();

        void AddObservation(KeyFrame* pKF, int idx);
        void AddParObservation(KeyFrame* pKF, int idx);
        void AddVerObservation(KeyFrame* pKF, int idx);

        /*
         * if the plane finds a boundary with other planes, we need to associate it with the plane.
         * boundaryLine: endpoints, direction vector.
         * index: the id of the plane that shares boundary with this
         *
         * */
        void AddCrossLines(Eigen::Matrix<double, 6, 1> &vEndPoints, int &index);
        //std::tuple<long unsigned int, long unsigned int, Eigen::Matrix<double, 3, 1>>
        vector<std::tuple< Eigen::Matrix<double, 6, 1>, int> > GetCrossLines();
        void UpdateCrossLine(Eigen::Matrix<double, 6, 1> &boundaryLine, int iID);

        void EraseObservation(KeyFrame* pKF);
        void EraseVerObservation(KeyFrame* pKF);
        void EraseParObservation(KeyFrame* pKF);

        std::map<KeyFrame*, size_t> GetObservations();
        std::map<KeyFrame*, size_t> GetParObservations();
        std::map<KeyFrame*, size_t> GetVerObservations();
        int Observations();
        int GetIndexInKeyFrame(KeyFrame *pKF);
        int GetIndexInVerticalKeyFrame(KeyFrame *pKF);
        int GetIndexInParallelKeyFrame(KeyFrame *pKF);
        bool IsInKeyFrame(KeyFrame *pKF);
        bool IsVerticalInKeyFrame(KeyFrame *pKF);
        bool IsParallelInKeyFrame(KeyFrame *pKF);
        void UpdateCoefficientsAndPoints();
        void UpdateCoefficientsAndPoints(Frame& pF, int id);
        void UpdateCompletePoints(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> &mvCompletePoints, KeyFrame *frame);
        void UpdateCompletePointsFrame(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> &mvCompletePoints, Frame &frame);
        void UpdateCoefficientsAndPoints(KeyFrame *pKF, int id);
        void UpdateComputePlaneBoundary();
        void UpdateComputePlaneBoundary(Frame& pF, int id);

    public:
        bool smallFlag;
        long unsigned int mnId; ///< Global ID for MapPlane;
        static long unsigned int nNextId;
        long int mnFirstKFid;
        long int mnFirstFrame;
        int nObs;

        // save all
        vector<std::tuple< Eigen::Matrix<double, 6, 1>, int> > mvtEndPointsIndex;

        static std::mutex mGlobalMutex;

        long unsigned int mnBALocalForKF; //used in local BA

        long unsigned int mnFuseCandidateForKF;

        // Variables used by loop closing
        long unsigned int mnLoopPlaneForKF;
        long unsigned int mnLoopVerticalPlaneForKF;
        long unsigned int mnLoopParallelPlaneForKF;
        long unsigned int mnCorrectedByKF;
        long unsigned int mnCorrectedReference;
        cv::Mat mPosGBA;
        long unsigned int mnBAGlobalForKF;

        //used for visualization
        int mRed;
        int mGreen;
        int mBlue;

        PointCloud::Ptr mvPlanePoints;
        PointCloud::Ptr mvCompletePoints;
        PointCloud::Ptr mvNoPlanePoints;
        PointCloud::Ptr copyPlanePoints;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_boundary;
        //Tracking counters
        int mnVisible;
        int mnFound;

        PointCloud::Ptr mpVirtualPts3d;                                                               // 平面撒的点（w系）
        PointCloud::Ptr mpVirtualPts2d;                                                               // 平面撒的点（local系）
        pcl::PointCloud<pcl::Normal>::Ptr mpVirtualNormals;                                           // 平面撒点的法向
        std::vector<std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>> mvLines; // 1个平面与其他平面最多可以找到4条交线段

        // 针对不能找到4个layout节点的平面
        PointCloud::Ptr mpPts2d;                     // 平面的点（local系）
        pcl::PointCloud<pcl::Normal>::Ptr mpNormals; // 平面点的法向

        std::unordered_set<Layout *> msLayout;
        Eigen::Vector3f mPw1, mPw2, mPw3, mPw4; // w系下平面的4个Layout节点（在local系下依次与横轴正半轴夹角递增）

    protected:
        cv::Mat mWorldPos; ///< Position in absolute coordinates

        std::map<KeyFrame*, size_t> mObservations;
        std::map<KeyFrame*, size_t> mParObservations;
        std::map<KeyFrame*, size_t> mVerObservations;

        std::mutex mMutexPos;
        std::mutex mMutexFeatures;

        KeyFrame* mpRefKF;

        bool mbBad;
        MapPlane* mpReplaced;

        Map* mpMap;

        bool MaxPointDistanceFromPlane(cv::Mat &plane, PointCloud::Ptr pointCloud);

//        void SetColor();
    };
}
#endif //ORB_SLAM2_MAPPLANE_H
