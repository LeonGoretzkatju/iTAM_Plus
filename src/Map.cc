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

#include "Map.h"
#include<mutex>
#include <utility>
using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace ORB_SLAM2 {

    size_t PartialManhattanMapHash::operator() (const std::pair<MapPlane*, MapPlane*>& key) const {
        int id1, id2;
        if (key.first->mnId > key.second->mnId) {
            id1 = key.second->mnId;
            id2 = key.first->mnId;
        } else {
            id1 = key.first->mnId;
            id2 = key.second->mnId;
        }

        size_t hash = 0;
        hash += (71*hash + id1) % 5;
        hash += (71*hash + id2) % 5;
        return hash;
    }

    size_t PairPlaneMapHash::operator() (const std::pair<MapPlane*, MapPlane*>& key) const {
        int id1, id2;
        if (key.first->mnId > key.second->mnId) {
            id1 = key.second->mnId;
            id2 = key.first->mnId;
        } else {
            id1 = key.first->mnId;
            id2 = key.second->mnId;
        }

        size_t hash = 0;
        hash += (71*hash + id1) % 5;
        hash += (71*hash + id2) % 5;
        return hash;
    }

    bool PartialManhattanMapEqual::operator() (const std::pair<MapPlane*, MapPlane*>& a, const std::pair<MapPlane*, MapPlane*>& b) const {
        MapPlane* pMP11, *pMP12, *pMP21, *pMP22;
        if (a.first->mnId > a.second->mnId) {
            pMP11 = a.second;
            pMP12 = a.first;
        } else {
            pMP11 = a.first;
            pMP12 = a.second;
        }

        if (b.first->mnId > b.second->mnId) {
            pMP21 = b.second;
            pMP22 = b.first;
        } else {
            pMP21 = b.first;
            pMP22 = b.second;
        }

        std::pair<MapPlane*, MapPlane*> p1 = std::make_pair(pMP11, pMP12);
        std::pair<MapPlane*, MapPlane*> p2 = std::make_pair(pMP21, pMP22);

        return p1 == p2;
    }

    bool PairPlaneMapEqual::operator() (const std::pair<MapPlane*, MapPlane*>& a, const std::pair<MapPlane*, MapPlane*>& b) const {
        MapPlane* pMP11, *pMP12, *pMP21, *pMP22;
        if (a.first->mnId > a.second->mnId) {
            pMP11 = a.second;
            pMP12 = a.first;
        } else {
            pMP11 = a.first;
            pMP12 = a.second;
        }

        if (b.first->mnId > b.second->mnId) {
            pMP21 = b.second;
            pMP22 = b.first;
        } else {
            pMP21 = b.first;
            pMP22 = b.second;
        }

        std::pair<MapPlane*, MapPlane*> p1 = std::make_pair(pMP11, pMP12);
        std::pair<MapPlane*, MapPlane*> p2 = std::make_pair(pMP21, pMP22);

        return p1 == p2;
    }

    size_t TuplePlaneMapHash::operator() (const std::tuple<MapPlane*, MapPlane*, MapPlane*>& key) const {
        vector<int> ids;
        ids.push_back(get<0>(key)->mnId);
        ids.push_back(get<1>(key)->mnId);
        ids.push_back(get<2>(key)->mnId);
        sort(ids.begin(), ids.end());

        size_t hash = 0;
        hash += (71*hash + ids[0]) % 5;
        hash += (71*hash + ids[1]) % 5;
        hash += (71*hash + ids[2]) % 5;
        return hash;
    }

    bool TuplePlaneMapEqual::operator() (const std::tuple<MapPlane*, MapPlane*, MapPlane*>& a,
                                        const std::tuple<MapPlane*, MapPlane*, MapPlane*>& b) const {
        MapPlane* pMP11, *pMP12, *pMP13, *pMP21, *pMP22, *pMP23;

        pMP11 = get<0>(a);
        pMP12 = get<1>(a);
        pMP13 = get<2>(a);

        if (pMP11 > pMP12)
        {
            std::swap(pMP11, pMP12);
        }
        if (pMP12 > pMP13)
        {
            std::swap(pMP12, pMP13);
        }
        if (pMP11 > pMP12)
        {
            std::swap(pMP11, pMP12);
        }

        pMP21 = get<0>(b);
        pMP22 = get<1>(b);
        pMP23 = get<2>(b);

        if (pMP21 > pMP22)
        {
            std::swap(pMP21, pMP22);
        }
        if (pMP22 > pMP23)
        {
            std::swap(pMP22, pMP23);
        }
        if (pMP21 > pMP22)
        {
            std::swap(pMP21, pMP22);
        }

        std::tuple<MapPlane*, MapPlane*, MapPlane*> t1 = std::make_tuple(pMP11, pMP12, pMP13);
        std::tuple<MapPlane*, MapPlane*, MapPlane*> t2 = std::make_tuple(pMP21, pMP22, pMP23);

        return t1 == t2;
    }

    size_t ManhattanMapHash::operator() (const std::tuple<MapPlane*, MapPlane*, MapPlane*>& key) const {
        vector<int> ids;
        ids.push_back(get<0>(key)->mnId);
        ids.push_back(get<1>(key)->mnId);
        ids.push_back(get<2>(key)->mnId);
        sort(ids.begin(), ids.end());

        size_t hash = 0;
        hash += (71*hash + ids[0]) % 5;
        hash += (71*hash + ids[1]) % 5;
        hash += (71*hash + ids[2]) % 5;
        return hash;
    }

    bool ManhattanMapEqual::operator() (const std::tuple<MapPlane*, MapPlane*, MapPlane*>& a,
                                        const std::tuple<MapPlane*, MapPlane*, MapPlane*>& b) const {
        MapPlane* pMP11, *pMP12, *pMP13, *pMP21, *pMP22, *pMP23;

        pMP11 = get<0>(a);
        pMP12 = get<1>(a);
        pMP13 = get<2>(a);

        if (pMP11 > pMP12)
        {
            std::swap(pMP11, pMP12);
        }
        if (pMP12 > pMP13)
        {
            std::swap(pMP12, pMP13);
        }
        if (pMP11 > pMP12)
        {
            std::swap(pMP11, pMP12);
        }

        pMP21 = get<0>(b);
        pMP22 = get<1>(b);
        pMP23 = get<2>(b);

        if (pMP21 > pMP22)
        {
            std::swap(pMP21, pMP22);
        }
        if (pMP22 > pMP23)
        {
            std::swap(pMP22, pMP23);
        }
        if (pMP21 > pMP22)
        {
            std::swap(pMP21, pMP22);
        }

        std::tuple<MapPlane*, MapPlane*, MapPlane*> t1 = std::make_tuple(pMP11, pMP12, pMP13);
        std::tuple<MapPlane*, MapPlane*, MapPlane*> t2 = std::make_tuple(pMP21, pMP22, pMP23);

        return t1 == t2;
    }

    Map::Map() : mnMaxKFid(0), mnBigChangeIdx(0) {
    }

    void Map::AddKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexMap);
        mspKeyFrames.insert(pKF);
        if (pKF->mnId > mnMaxKFid)
            mnMaxKFid = pKF->mnId;
    }

    void Map::AddMapPoint(MapPoint *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPoints.insert(pMP);
    }

    void Map::EraseMapPoint(MapPoint *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPoints.erase(pMP);

        // TODO: This only erase the pointer.
        // Delete the MapPoint
    }

    void Map::EraseKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexMap);
        mspKeyFrames.erase(pKF);

        // TODO: This only erase the pointer.
        // Delete the MapPoint
    }

    void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs) {
        unique_lock<mutex> lock(mMutexMap);
        mvpReferenceMapPoints = vpMPs;
    }

    void Map::InformNewBigChange() {
        unique_lock<mutex> lock(mMutexMap);
        mnBigChangeIdx++;
    }

    int Map::GetLastBigChangeIdx() {
        unique_lock<mutex> lock(mMutexMap);
        return mnBigChangeIdx;
    }

    vector<KeyFrame *> Map::GetAllKeyFrames() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<KeyFrame *>(mspKeyFrames.begin(), mspKeyFrames.end());
    }

    vector<MapPoint *> Map::GetAllMapPoints() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapPoint *>(mspMapPoints.begin(), mspMapPoints.end());
    }

    long unsigned int Map::MapPointsInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspMapPoints.size();
    }

    long unsigned int Map::KeyFramesInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspKeyFrames.size();
    }

    vector<MapPoint *> Map::GetReferenceMapPoints() {
        unique_lock<mutex> lock(mMutexMap);
        return mvpReferenceMapPoints;
    }

    long unsigned int Map::GetMaxKFid() {
        unique_lock<mutex> lock(mMutexMap);
        return mnMaxKFid;
    }

    void Map::clear() {
        for (auto mspMapPoint : mspMapPoints)
            delete mspMapPoint;
        for (auto mspMapLine : mspMapLines)
            delete mspMapLine;
        for (auto mspMapPlane : mspMapPlanes)
            delete mspMapPlane;

        for (auto mspKeyFrame : mspKeyFrames)
            delete mspKeyFrame;

        mspMapPlanes.clear();
        mspMapPoints.clear();
        mspKeyFrames.clear();
        mspMapLines.clear();
        //mspBoundaryLines.clear();
        mspDirectionVector.clear();
        BoundaryPoints.clear();
        mnMaxKFid = 0;
        mvpReferenceMapPoints.clear();
        mvpReferenceMapLines.clear();
        mvpKeyFrameOrigins.clear();
    }

    void Map::AddMapLine(MapLine *pML) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapLines.insert(pML);
    }

    void Map::AddBoundaryLine(Eigen::Matrix<double ,6 , 1> &boundaryLine) {
        unique_lock<mutex> lock(mMutexMap);
        mspBoundaryLines.emplace_back(boundaryLine);
    }

    void Map::AddCrossPoint(cv::Mat& CrossPoint) {
        unique_lock<mutex> lock(mMutexMap);
        mspCrossPoints.emplace_back(CrossPoint);
    }

    void Map::AddLayout(list<Layout *> &lLayouts) {
        unique_lock<mutex> lock(mMutexMap);
        DrawLayout = lLayouts;
    }

    void Map::AddDirectionVector(Eigen::Matrix<double, 3, 1> &DirectionVector) {
        unique_lock<mutex> lock(mMutexMap);
        mspDirectionVector.emplace_back(DirectionVector);
    }

    void Map::EraseMapLine(MapLine *pML) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapLines.erase(pML);
    }

    /**
     * @brief 设置参考MapLines，将用于DrawMapLines函数画图
     * @param vpMLs Local MapLines
     */
    void Map::SetReferenceMapLines(const std::vector<MapLine *> &vpMLs) {
        unique_lock<mutex> lock(mMutexMap);
        mvpReferenceMapLines = vpMLs;
    }

    vector<MapLine *> Map::GetAllMapLines() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapLine *>(mspMapLines.begin(), mspMapLines.end());
    }

    vector<Eigen::Matrix<double ,6 , 1>> Map::GetAllPlaneIntersections() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<Eigen::Matrix<double ,6 , 1>>(mspBoundaryLines.begin(), mspBoundaryLines.end());
    }

    vector<cv::Mat> Map::GetAllCrossPointInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<cv::Mat>(mspCrossPoints.begin(), mspCrossPoints.end());
    }

    vector<MapLine *> Map::GetReferenceMapLines() {
        unique_lock<mutex> lock(mMutexMap);
        return mvpReferenceMapLines;
    }

    long unsigned int Map::MapLinesInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspMapLines.size();
    }

    void Map::AddMapPlane(MapPlane *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        pMP->mvPlanePoints.get()->points;
        pMP->mvNoPlanePoints.get()->points;
        mspMapPlanes.insert(pMP);
    }

    void Map::AddOuterPlane(vector<MapPlane *> pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspOuterPlanes = std::move(pMP);
    }

    void Map::AddBoundaryPoints(pcl::PointXYZRGB &p) {
        unique_lock<mutex> lock(mMutexMap);
        BoundaryPoints.emplace_back(p);
    }

    void Map::AddInlierLines(pcl::PointXYZRGB &p) {
        unique_lock<mutex> lock(mMutexMap);
        InlierLines.emplace_back(p);
    }

    bool SetSortZ(pcl::PointXYZRGB &p1, pcl::PointXYZRGB &p2) {
        if (p1.z != p2.z)
            return p1.z < p2.z;
        else
            if(p1.x != p2.x)
                return p1.x < p2.x;
            else
                if(p1.y != p2.y)
                    return p1.y < p2.y;
    }

    double Map::PointDistanceFromPlane(const cv::Mat &plane, PointCloud::Ptr pointCloud, double minSize) {
        double sum = 0;
        for (int i = 0; i < minSize; ++i){
//            cout <<"p x" << p.x << endl;
            double dis = abs(plane.at<float>(0, 0) * pointCloud->points[i].x +
                             plane.at<float>(1, 0) * pointCloud->points[i].y +
                             plane.at<float>(2, 0) * pointCloud->points[i].z +
                             plane.at<float>(3, 0));
//            cout << "compute 1 iteration" << endl;
            sum += dis;
        }
        return sum/minSize;
    }

    double Map::PointToPlaneDistance(const cv::Mat &plane, pcl::PointXYZRGB &point) {
        double dis = abs(plane.at<float>(0, 0) * point.x +
                         plane.at<float>(1, 0) * point.y +
                         plane.at<float>(2, 0) * point.z +
                         plane.at<float>(3, 0));
        return dis;
    }

    void Map::UpdateCompletePointsAllPlanes(const std::vector<MapPlane *> &vpMapPlanes, KeyFrame *CurrentFrame) {
        for (int i = 0; i < vpMapPlanes.size(); ++i) {
            if (!vpMapPlanes[i] || vpMapPlanes[i]->isBad()) {
                continue;
            }
            vpMapPlanes[i]->UpdateCompletePoints(vpMapPlanes[i]->mvCompletePoints, CurrentFrame);
        }
    }

    void Map::ExtractLayout(const vector<MapPlane *> &vpMapPlanes, list<Layout *> &lLayouts, cv::Mat &CameraCenter) {
        lLayouts.remove_if([](Layout *x) { return x->confidence != 1.0; });
        unordered_set<int> table;
        for (const auto &plane : vpMapPlanes)
            table.insert(plane->mnId);
        lLayouts.remove_if([&table](Layout *x) {
            if (table.find(x->mvPlaneId[0]) == table.end() || table.find(x->mvPlaneId[1]) == table.end() || table.find(x->mvPlaneId[2]) == table.end())
                return true;
            else
                return false;
        });
        if (vpMapPlanes.size() < 3)
            return;
        vector<int> outerPlaneIdx;

        // 最外层平面
        auto outerPlanes = ExtractOuterPlaneBounding(vpMapPlanes, outerPlaneIdx, CameraCenter);
        cout << "outerplanes idx size" << outerPlaneIdx.size() << endl;

        vector<vector<int>> parrl;
        for (size_t ii = 0; ii < outerPlaneIdx.size(); ii++)
        {
            for (size_t jj = 0; jj < outerPlaneIdx.size(); jj++)
            {
                for (size_t kk = 0; kk < outerPlaneIdx.size(); kk++)
                {
                    int i = outerPlaneIdx[ii], j = outerPlaneIdx[jj], k = outerPlaneIdx[kk];
                    if (i == j || i == k || j == k)
                        continue;

                    Eigen::Matrix3f A;
                    A << vpMapPlanes[i]->GetWorldPos().at<float>(0, 0), vpMapPlanes[i]->GetWorldPos().at<float>(1, 0), vpMapPlanes[i]->GetWorldPos().at<float>(2, 0),
                            vpMapPlanes[j]->GetWorldPos().at<float>(0, 0), vpMapPlanes[j]->GetWorldPos().at<float>(1, 0), vpMapPlanes[j]->GetWorldPos().at<float>(2, 0),
                            vpMapPlanes[k]->GetWorldPos().at<float>(0, 0), vpMapPlanes[k]->GetWorldPos().at<float>(1, 0), vpMapPlanes[k]->GetWorldPos().at<float>(2, 0);
                    Eigen::Vector3f x;
                    Eigen::Vector3f b;
                    b << -1 * vpMapPlanes[i]->GetWorldPos().at<float>(3, 0), -1 * vpMapPlanes[j]->GetWorldPos().at<float>(3, 0), -1 * vpMapPlanes[k]->GetWorldPos().at<float>(3, 0);

                    // 判断平面是否接近平行
                    if (abs(A.determinant()) < 5e-1)
                        continue;

                    x = A.colPivHouseholderQr().solve(b);
                    int id1 = vpMapPlanes[i]->mnId, id2 = vpMapPlanes[j]->mnId, id3 = vpMapPlanes[k]->mnId;
                    Layout *node = new Layout(x[0], x[1], x[2], id1, id2, id3);

                    node->confidence = 0.0;

                    // 判断是否应该将该layout节点设置为高可信度
                    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
                    vector<int> pointIdxRadiusSearch;
                    vector<float> pointRadiusSquaredDistance;
                    float radius = 0.1;

                    pcl::PointXYZRGB searchPoint(x[0], x[1], x[2]);
                    kdtree.setInputCloud(vpMapPlanes[i]->mvPlanePoints);
                    int n1 = kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
                    kdtree.setInputCloud(vpMapPlanes[j]->mvPlanePoints);
                    int n2 = kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
                    kdtree.setInputCloud(vpMapPlanes[k]->mvPlanePoints);
                    int n3 = kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
                    bool b1 = n1 > 0, b2 = n2 > 0, b3 = n3 > 0;
                    if (b1 && b2 && b3)
                        node->confidence = 1.0;
                    else if (b1 && b2 && vpMapPlanes[k]->mvPlanePoints->size() > 500)
                        node->confidence = 1.0;
                    else if (b1 && b3 && vpMapPlanes[j]->mvPlanePoints->size() > 500)
                        node->confidence = 1.0;
                    else if (b2 && b3 && vpMapPlanes[i]->mvPlanePoints->size() > 500)
                        node->confidence = 1.0;
                    else if (b1 || b2 || b3)
                    {
                        float radius = 0.8;
                        kdtree.setInputCloud(vpMapPlanes[i]->mvPlanePoints);
                        int n1 = kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
                        kdtree.setInputCloud(vpMapPlanes[j]->mvPlanePoints);
                        int n2 = kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
                        kdtree.setInputCloud(vpMapPlanes[k]->mvPlanePoints);
                        int n3 = kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
                        if (n1 > 10 && n2 > 10 && n3 > 10 && vpMapPlanes[i]->mvPlanePoints->size() > 500 && vpMapPlanes[j]->mvPlanePoints->size() > 500 && vpMapPlanes[k]->mvPlanePoints->size() > 500)
                            node->confidence = 1.0;
                    }

                    // lock.lock();
                    bool bAbortNode = false;
                    auto it = lLayouts.begin();
                    while (it != lLayouts.end())
                    {
                        Layout *p = *it;
                        auto curIt = it++;
                        if (sqrt((node->pos[0] - p->pos[0]) * (node->pos[0] - p->pos[0]) + (node->pos[1] - p->pos[1]) * (node->pos[1] - p->pos[1]) + (node->pos[2] - p->pos[2]) * (node->pos[2] - p->pos[2])) < 0.05)
                        {
                            // 当前节点node为高可信度 || 当前节点node和遍历到的节点p都不为高可信度，则直接将后找到的node替代p
                            if (node->confidence == 1.0 || (node->confidence != 1.0 && p->confidence != 1.0))
                            {
                                lLayouts.erase(curIt);
                                vpMapPlanes[i]->EraseLayout(*curIt);
                                vpMapPlanes[k]->EraseLayout(*curIt);
                                vpMapPlanes[j]->EraseLayout(*curIt);
                            }
                                // 当前节点node不为高可信度的前提下，如果p为高可信度，则应该放弃push_back当前节点
                            else if (node->confidence != 1.0 && p->confidence == 1.0)
                                bAbortNode = true;
                        }
                    }
                    // lock.unlock();
                    if (!bAbortNode)
                    {
                        // mpMap->AddLayout(node);
                        lLayouts.push_back(node);
                        vpMapPlanes[i]->AddLayout(node);
                        vpMapPlanes[k]->AddLayout(node);
                        vpMapPlanes[j]->AddLayout(node);
                    }
                }
            }
        }

    }

    void Map::triangleRecon(const PointCloud::Ptr &pCloud2d, const PointCloud::Ptr &pCloud3d,
                            const pcl::PointCloud<pcl::Normal>::Ptr &pNormals, pcl::PolygonMesh &triangles) {
        triangulateio in, out;
        // inputs
        in.numberofpoints = pCloud2d->size();
        in.pointlist = (float *)malloc(in.numberofpoints * 2 * sizeof(float));
        int32_t k = 0;
        for (auto &p : pCloud2d->points)
        {
            in.pointlist[k++] = p.x;
            in.pointlist[k++] = p.y;
        }
        in.numberofpointattributes = 0;
        in.pointattributelist = nullptr;
        in.pointmarkerlist = nullptr;
        in.numberofholes = 0;
        in.holelist = nullptr;
        in.numberofregions = 0;
        in.regionlist = nullptr;
        // outputs
        out.pointlist = nullptr;
        out.pointattributelist = nullptr;
        out.pointmarkerlist = nullptr;
        out.trianglelist = nullptr;
        out.triangleattributelist = nullptr;
        out.neighborlist = nullptr;
        out.segmentlist = nullptr;
        out.segmentmarkerlist = nullptr;
        out.edgelist = nullptr;
        out.edgemarkerlist = nullptr;

        // char parameters[] = "pnezcQ";
        char parameters[] = "zQ";
        if (in.numberofpoints >= 3)
        {
            triangulate(parameters, &in, &out, nullptr);

            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
            pcl::concatenateFields(*pCloud3d, *pNormals, *cloud_with_normals);
            pcl::toPCLPointCloud2(*cloud_with_normals, triangles.cloud);
            k = 0;
            for (int index = 0; index < out.numberoftriangles; ++index)
            {
                pcl::Vertices vtx;
                vtx.vertices.push_back(out.trianglelist[k]); // trianglelist为vertex的索引，一次索引3个就是组成一个三角面片
                k++;
                vtx.vertices.push_back(out.trianglelist[k]);
                k++;
                vtx.vertices.push_back(out.trianglelist[k]);
                k++;
                triangles.polygons.push_back(vtx);
            }
        }
        // free memory used for triangulation
        free(in.pointlist);
        free(out.pointlist);
        free(out.trianglelist);
    }

    void Map::generateLine(const Eigen::Vector3f &startPoint, const Eigen::Vector3f &endPoint,
                           vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> &line,
                           const float &interval) {
        Eigen::Vector3f direction = (endPoint - startPoint).normalized();
        Eigen::Vector3f step = interval * direction;
        Eigen::Vector3f pt = startPoint;
        // 生成的点pt足够接近终点？
        while (abs(pt.x() - endPoint.x()) > interval || abs(pt.y() - endPoint.y()) > interval)
        {
            line.push_back(pt);
            pt += step;
        }
    }

    vector<Eigen::Vector3f> Map::discreteCrossLine(Eigen::Vector3f &startPoint, Eigen::Vector3f &endPoint, int &PtsSize) {
        float base = sqrt(pow((endPoint[0]-startPoint[0]),2)+pow((endPoint[1]-startPoint[1]),2)+
                                  pow((endPoint[2]-startPoint[2]),2));
        vector<Eigen::Vector3f> pts3D;
        pts3D.reserve(PtsSize+1);
        float step = base/PtsSize;
        for (int i = 0; i <= PtsSize; ++i) {
            Eigen::Vector3f p;
            p = startPoint * (1 - i / PtsSize) + endPoint * (i / PtsSize);
            pts3D.emplace_back(p);
        }
        cout << "pts3D size" << pts3D.size() << endl;
        return pts3D;
    }

    template<class T>
    inline T uMax3( const T& a, const T& b, const T& c)
    {
        float m=a>b?a:b;
        return m>c?m:c;
    }

    void Map::reconstructNoPlane(const vector<MapPlane *> &vpMapPlanes, const float &resolution,
                                 const string &outPath) {
        int optimizedDepth;
        PointCloud::Ptr combinedPoints (new PointCloud());
        for (auto plane : vpMapPlanes) {
            *combinedPoints += *plane->mvNoPlanePoints;
        }
        Eigen::Vector4f min,max;
        pcl::getMinMax3D(*combinedPoints, min, max);
        float mapLength = uMax3(max[0]-min[0], max[1]-min[1], max[2]-min[2]);
        optimizedDepth = 12;
        for(int i=6; i<12; ++i)
        {
            if(mapLength/float(1<<i) < 0.03f)
            {
                optimizedDepth = i;
                break;
            }
        }
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudTranslated(new pcl::PointCloud<pcl::PointXYZRGB>());
        *cloudTranslated = *combinedPoints;
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree_for_points (new pcl::search::KdTree<pcl::PointXYZRGB>);
        kdtree_for_points->setInputCloud(cloudTranslated);
        pcl::NormalEstimation<pcl::PointXYZRGB,pcl::Normal> normEst;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        normEst.setInputCloud(combinedPoints);
        normEst.setSearchMethod(kdtree_for_points);
        normEst.setKSearch(20); //It was 20
        normEst.compute(*normals);//Normals are estimated using standard method.
        cout << "finish norm est" << endl;
        cout << "norms size" << normals->points.size() << endl;
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloudWithNormals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::concatenateFields(*combinedPoints, *normals, *cloudWithNormals);
        cout << "finish combine" << endl;
        pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh);
        pcl::Poisson<pcl::PointXYZRGBNormal> poisson;
        poisson.setDepth(optimizedDepth);
        poisson.setInputCloud(cloudWithNormals);
        poisson.reconstruct(*mesh);
//        auto open3dPlane = std::shared_ptr<open3d::geometry::PointCloud>();
//        for (auto nopt : *combinedPoints) {
//            Eigen::Vector3f storeno;
//            storeno << nopt.x , nopt.y, nopt.z;
//            open3dPlane->points_.emplace_back(storeno);
//        }
//        auto boundingbox = open3dPlane->GetAxisAlignedBoundingBox();
//        auto poisson_mesh = open3d::geometry::TriangleMesh::CreateFromPointCloudPoisson(*open3dPlane,
//                                                                8,0,1.1, false);
//        auto p_mesh = std::get<0>(poisson_mesh);
//        auto clear_mesh = p_mesh->Crop(boundingbox);
//        open3d::io::WriteTriangleMeshToPLY("/home/nuc/clearno.ply",*clear_mesh, true, true, true, true, true, true);
//            poisson.setDepth(9);//9
//            poisson.setInputCloud(cloudWithNormals);
//            poisson.setPointWeight(4);//4
//            poisson.setDegree(2);
//            poisson.setSamplesPerNode(1.5);//1.5
//            poisson.setScale(1.1);//1.1
//            poisson.setIsoDivide(8);//8
//            poisson.setConfidence(true);
//            poisson.setOutputPolygons(true);
//            poisson.setManifold(true);
//            poisson.setSolverDivide(8);//8
//            poisson.reconstruct(*mesh);
        cout << "finish possion" << endl;
//            vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
//            pcl::PolygonMesh mesh_pcl;
//            pcl::VTKUtils::convertToVTK(*mesh,polydata);
//            pcl::VTKUtils::convertToPCL(polydata,mesh_pcl);
//            pcl::io::savePolygonFilePLY(outPath + "/" + to_string(plane->mnId) + "_no_plane_mesh.ply", mesh_pcl);
        pcl::io::savePLYFile(outPath + "/" + to_string(111) + "_no_plane_mesh.ply", *mesh);
    }

    void Map::reconstructPlane(const vector<MapPlane *> &vpMapPlanes,
                               const float &resolution, const string &outPath) {
        int ptsSize = 100;
        for (auto plane : vpMapPlanes)
        {
            float minX = 0, minY = 0; // 平面2d点的最小值
            // 建立平面坐标系uv
            Eigen::Vector3f normal;
            normal << plane->GetWorldPos().at<float>(0, 0), plane->GetWorldPos().at<float>(1, 0), plane->GetWorldPos().at<float>(2, 0);
            Eigen::Vector3f u;
            int size = plane->mvPlanePoints->size();
            u << plane->mvPlanePoints->points[0].x - plane->mvPlanePoints->points[size - 1].x,
                    plane->mvPlanePoints->points[0].y - plane->mvPlanePoints->points[size - 1].y,
                    plane->mvPlanePoints->points[0].z - plane->mvPlanePoints->points[size - 1].z;
            Eigen::Vector3f v = normal.cross(u);
            u.normalize();
            v.normalize();
            auto CrossLine3Dsets = plane->GetCrossLines();
            PointCloud::Ptr discreteSets (new PointCloud());
            if (!CrossLine3Dsets.empty())
            {
                for (int m = 0; m < CrossLine3Dsets.size(); ++m) {
                    auto Line = std::get<0>(CrossLine3Dsets[m]);
                    Eigen::Vector3f startPoint, endPoint;
                    startPoint << Line(0), Line(1), Line(2);
                    endPoint << Line(3), Line(4), Line(5);
                    auto pts3D = discreteCrossLine(startPoint,endPoint,ptsSize);
                    for (int n = 0; n < pts3D.size(); ++n) {
                        PointT pp;
                        pp.x = pts3D[n](0);
                        pp.y = pts3D[n](1);
                        pp.z = pts3D[n](2);
                        pp.r = plane->mRed;
                        pp.g = plane->mGreen;
                        pp.b = plane->mBlue;
                        discreteSets->points.push_back(PointT(pp));
                        DrawDiscretePoints.push_back(PointT(pp));
                    }
                }
            }
            // 将3d点投影到uv得到2d点
            size_t i = 0;
            for (const auto &pt : (*plane->mvPlanePoints))
            {
                Eigen::Vector3f pt3d(pt.x, pt.y, pt.z);
                Eigen::Vector3f pt2d(u.dot(pt3d), v.dot(pt3d), 0);

                minX = min(minX, pt2d.x());
                minY = min(minY, pt2d.y());
                plane->mvPlanePoints->points[i].r = plane->mRed;
                plane->mvPlanePoints->points[i].g = plane->mGreen;
                plane->mvPlanePoints->points[i].b = plane->mBlue;
                pcl::PointXYZRGB tempPoint;
                tempPoint.x = pt2d.x();
                tempPoint.y = pt2d.y();
                tempPoint.z = 0;
                tempPoint.r = plane->mRed;
                tempPoint.g = plane->mGreen;
                tempPoint.b = plane->mBlue;
                //plane->mpPts2d->push_back(tempPoint);
                plane->mpPts2d->points.push_back(PointT(tempPoint));
                plane->mpNormals->points.push_back(pcl::Normal(plane->GetWorldPos().at<float>(0, 0), plane->GetWorldPos().at<float>(1, 0), plane->GetWorldPos().at<float>(2, 0)));
                i++;
            }
            size_t j = 0;
            for (const auto &ptt : *discreteSets)
            {
                Eigen::Vector3f pt3d(ptt.x, ptt.y, ptt.z);
                Eigen::Vector3f pt2d(u.dot(pt3d), v.dot(pt3d), 0);

                minX = min(minX, pt2d.x());
                minY = min(minY, pt2d.y());
                discreteSets->points[j].r = plane->mRed;
                discreteSets->points[j].g = plane->mGreen;
                discreteSets->points[j].b = plane->mBlue;
                pcl::PointXYZRGB tempPoint6;
                tempPoint6.x = pt2d.x();
                tempPoint6.y = pt2d.y();
                tempPoint6.z = 0;
                tempPoint6.r = plane->mRed;
                tempPoint6.g = plane->mGreen;
                tempPoint6.b = plane->mBlue;
                //plane->mpPts2d->push_back(tempPoint);
                plane->mpPts2d->points.push_back(PointT(tempPoint6));
                plane->mpNormals->points.push_back(pcl::Normal(plane->GetWorldPos().at<float>(0, 0), plane->GetWorldPos().at<float>(1, 0), plane->GetWorldPos().at<float>(2, 0)));
                j++;
            }
            size_t k = 0;
            for (const auto &pttt : *plane->mvCompletePoints)
            {
                Eigen::Vector3f ptC3d(pttt.x, pttt.y, pttt.z);
                Eigen::Vector3f ptC2d(u.dot(ptC3d), v.dot(ptC3d), 0);

                minX = min(minX, ptC2d.x());
                minY = min(minY, ptC2d.y());
                plane->mvCompletePoints->points[k].r = plane->mRed;
                plane->mvCompletePoints->points[k].g = plane->mGreen;
                plane->mvCompletePoints->points[k].b = plane->mBlue;
                pcl::PointXYZRGB tempPoint66;
                tempPoint66.x = ptC2d.x();
                tempPoint66.y = ptC2d.y();
                tempPoint66.z = 0;
                tempPoint66.r = plane->mRed;
                tempPoint66.g = plane->mGreen;
                tempPoint66.b = plane->mBlue;
                //plane->mpPts2d->push_back(tempPoint);
                plane->mpPts2d->points.push_back(PointT(tempPoint66));
                plane->mpNormals->points.push_back(pcl::Normal(plane->GetWorldPos().at<float>(0, 0), plane->GetWorldPos().at<float>(1, 0), plane->GetWorldPos().at<float>(2, 0)));
                k++;
            }
            PointCloud::Ptr allPoints (new PointCloud());
            *allPoints = *plane->mvPlanePoints + *discreteSets;
            *allPoints = *allPoints + *plane->mvCompletePoints;
            cout << "finish all points" << endl;
//                cout << "all size" << allPoints->points.size() << endl;
            plane->mpPts2d->height = 1;
            plane->mpPts2d->width = plane->mpPts2d->points.size();
            plane->mpPts2d->resize(plane->mpPts2d->height * plane->mpPts2d->width);

            plane->mpNormals->height = 1;
            plane->mpNormals->width = plane->mpNormals->points.size();
            plane->mpNormals->resize(plane->mpNormals->height * plane->mpNormals->width);

            // 调整平面2d点，避免出现负数（有负数可能会在triangulate时segmentation fault）
            for (size_t i = 0; i < plane->mpPts2d->size(); i++)
            {
                plane->mpPts2d->points[i].x += fabs(minX);
                plane->mpPts2d->points[i].y += fabs(minY);
            }

            // 使用triangle库进行构网
            pcl::PolygonMesh triangles;
            cout << "mvplanepoints" << plane->mvPlanePoints->points.size() << endl;
            cout << "mpPts2d" << plane->mpPts2d->size() << endl;
            cout << "mpNormals size" << plane->mpNormals->size() << endl;
            triangleRecon(plane->mpPts2d, allPoints, plane->mpNormals, triangles);
            pcl::io::savePLYFile(outPath + "/" + to_string(plane->mnId) + "_plane_mesh.ply", triangles);
        }
    }

    void Map::EstimateBoundingBox(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> &cloud,
                                       pcl::PointXYZRGB &minPoint, pcl::PointXYZRGB &maxPoint) {

        //PointT min_pt, max_pt;						// 沿参考坐标系坐标轴的边界值
        pcl::getMinMax3D(*cloud, minPoint, maxPoint);
    }

    std::vector<MapPlane *> Map::ExtractOuterPlaneBounding(const std::vector<MapPlane *> &vpMapPlanes,
                                                           std::vector<int> &outerPlaneIdx, cv::Mat &CameraCenter) {
//        unique_lock<mutex> lock(mMutexMap);
        vector<MapPlane *> outerPlane;
//        cout << "minpoint xyz" << minPoint1.x << " " << minPoint1.y << " " << minPoint1.z << endl;
//        cout << "maxpoint xyz" << maxPoint1.x << " " << maxPoint1.y << " " << maxPoint1.z << endl;
        //整体思路就是 计算平面上的点投影到其他平面上的点的总数，如果说这个总数与这个平面点总数之比 大于 0.6 那么这个平面就是所谓的遮挡平面
        // 或者说是小平面 每一个mapplane有一个smallflag标志位 如果是小平面的话 那么这个标志位就是true
        DrawTupleCube.clear();
        for (int i = 0; i < vpMapPlanes.size(); ++i) {
            if (!vpMapPlanes[i] || vpMapPlanes[i]->isBad()){
                continue;
            }
            float w = 0.0;
            for (int j = 0; j < vpMapPlanes.size(); ++j) {
                if (!vpMapPlanes[j] || vpMapPlanes[j]->isBad()){
                    continue;
                }
                if (vpMapPlanes[i] == vpMapPlanes[j]){
                    continue;
                }
                else
                {
                    cv::Mat plane1 = vpMapPlanes[j]->GetWorldPos();
                    pcl::PointXYZRGB minPoint1, maxPoint1;
                    EstimateBoundingBox(vpMapPlanes[j]->mvPlanePoints, minPoint1, maxPoint1);
                    auto tupleCube = std::make_tuple(minPoint1,maxPoint1);
                    AddTupleCube(tupleCube);
                    for (int k = 0; k < vpMapPlanes[i]->mvPlanePoints->size(); ++k) {
                        auto &p = vpMapPlanes[i]->mvPlanePoints->points[k];
                        float base = sqrt(
                                pow((p.x - CameraCenter.at<float>(0)), 2) + pow((p.y - CameraCenter.at<float>(1)), 2) +
                                pow((p.z - CameraCenter.at<float>(2)), 2));
                        cv::Mat DirectionVector = (cv::Mat_<float>(3, 1) << (p.x - CameraCenter.at<float>(0)) / base,
                                (p.y - CameraCenter.at<float>(1)) / base,
                                (p.z - CameraCenter.at<float>(2)) / base);
                        float angle12 = plane1.at<float>(0, 0) * DirectionVector.at<float>(0, 0) +
                                        plane1.at<float>(1, 0) * DirectionVector.at<float>(1, 0) +
                                        plane1.at<float>(2, 0) * DirectionVector.at<float>(2, 0);
                        float B = -(plane1.at<float>(0, 0) * CameraCenter.at<float>(0) + plane1.at<float>(1, 0) * CameraCenter.at<float>(1) +
                                    plane1.at<float>(2, 0) * CameraCenter.at<float>(2) + plane1.at<float>(3, 0));
                        float t = B / angle12;
                        float x = CameraCenter.at<float>(0) + DirectionVector.at<float>(0, 0) * t;
                        float y = CameraCenter.at<float>(1) + DirectionVector.at<float>(1, 0) * t;
                        float z = CameraCenter.at<float>(2) + DirectionVector.at<float>(2, 0) * t;
                        float ParallelResultX = (x - p.x) / (p.x - CameraCenter.at<float>(0));
                        float ParallelResultY = (y - p.y) / (p.y - CameraCenter.at<float>(1));
                        float ParallelResultZ = (z - p.z) / (p.z - CameraCenter.at<float>(2));
                        if (ParallelResultZ > 0 && ParallelResultX > 0 && ParallelResultY > 0)
                        {
                            if (x >= minPoint1.x && x <= maxPoint1.x &&
                                y >= minPoint1.y && y <= maxPoint1.y && z <= maxPoint1.z && z >= minPoint1.z)
                            {
                                w++;
                            }
                        }
                    }
                }
            }
            float cresult = w / vpMapPlanes[i]->mvPlanePoints->size();
            if (cresult >= 0.6)
            {
                vpMapPlanes[i]->smallFlag = true;
            }
            else
            {
                vpMapPlanes[i]->smallFlag = false;
            }
        }
        for (int m = 0; m < vpMapPlanes.size(); ++m) {
            if (!vpMapPlanes[m]->smallFlag)
            {
                outerPlane.emplace_back(vpMapPlanes[m]);
                outerPlaneIdx.push_back(m);
            }
            cout << "every small flag" << "  " << vpMapPlanes[m]->smallFlag << endl;
        }
        return outerPlane;
    }

    std::vector<MapPlane *> Map::ExtractOuterPlane(const std::vector<MapPlane *> &vpMapPlanes, std::vector<int> &outerPlaneIdx) {
//        vector<int> outerPlaneIdx;
        std::vector<MapPlane *> outerPlanes;
        unordered_set<int> sQueried;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        Eigen::Vector4f centroid;
        for (size_t i = 0; i < vpMapPlanes.size(); i++)
            *cloud += *vpMapPlanes[i]->mvPlanePoints;
        pcl::compute3DCentroid(*cloud, centroid);
        for(const auto &plane:vpMapPlanes)
        {
            // 指向屋内的方向
            Eigen::Vector3f innerDirection(centroid.x() - plane->mvPlanePoints->points[0].x, centroid.y() - plane->mvPlanePoints->points[0].y, centroid.z() - plane->mvPlanePoints->points[0].z);
            // 指向屋内的方向应该与法向量方向相同，否则法向量应该取反
            Eigen::Vector3f normal;
            normal[0] = plane->GetWorldPos().at<float>(0, 0);
            normal[1] = plane->GetWorldPos().at<float>(1, 0);
            normal[2] = plane->GetWorldPos().at<float>(2, 0);
            if (normal.dot(innerDirection) < 0)
                plane->flipNormal();
        }

        for (size_t i = 0; i < vpMapPlanes.size(); i++)
        {
            if (sQueried.find(i) != sQueried.end())
                continue;

            if (vpMapPlanes[i]->mvPlanePoints->size() < 100) // 限制平面最小点数
                continue;

            Eigen::Vector3f normalI;
            normalI[0] = vpMapPlanes[i]->GetWorldPos().at<float>(0, 0);
            normalI[1] = vpMapPlanes[i]->GetWorldPos().at<float>(1, 0);
            normalI[2] = vpMapPlanes[i]->GetWorldPos().at<float>(2, 0);

            float D = vpMapPlanes[i]->GetWorldPos().at<float>(3, 0);
            // 重心到平面i的距离
            float disI = fabs(normalI[0] * centroid[0] + normalI[1] * centroid[1] + normalI[2] * centroid[2] + D) / sqrt(normalI[0] * normalI[0] + normalI[1] * normalI[1] + normalI[2] * normalI[2]);
            sQueried.insert(i);

            float maxDis = disI;
            int maxDisIdx = i;

            for (size_t j = i + 1; j < vpMapPlanes.size(); j++)
            {
                if (sQueried.find(j) != sQueried.end())
                    continue;
                Eigen::Vector3f normalJ;
                normalJ[0] = vpMapPlanes[j]->GetWorldPos().at<float>(0, 0);
                normalJ[1] = vpMapPlanes[j]->GetWorldPos().at<float>(1, 0);
                normalJ[2] = vpMapPlanes[j]->GetWorldPos().at<float>(2, 0);

                float D = vpMapPlanes[j]->GetWorldPos().at<float>(3, 0);
                // 重心到平面j的距离
                float disJ = fabs(normalJ[0] * centroid[0] + normalJ[1] * centroid[1] + normalJ[2] * centroid[2] + D) / sqrt(normalJ[0] * normalJ[0] + normalJ[1] * normalJ[1] + normalJ[2] * normalJ[2]);

                float cosVal = normalI.dot(normalJ) / (normalI.norm() * normalJ.norm());
                float angle = acos(cosVal) * 180 / M_PI;
                if (angle < 60)
                {
                    sQueried.insert(j);
                    if (disJ > maxDis)
                    {
                        maxDis = disJ;
                        maxDisIdx = j;
                    }
                }
            }
            outerPlaneIdx.push_back(maxDisIdx);
            outerPlanes.emplace_back(vpMapPlanes[maxDisIdx]);
        }
        return outerPlanes;
    }

    void Map::UpdateCompletePointsFrame(const std::vector<MapPlane*> &vpMapPlanes,std::vector<MapPlane*> &vpOuterPlanes,
                        boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> &NoPlaneArea, Frame& CurrentFrame)
    {
//        unique_lock<mutex> lock(mMutexMap);
        auto verThreshold = Config::Get<double>("Plane.MFVerticalThreshold");
        if (!vpOuterPlanes.empty() && (vpMapPlanes.size() != vpOuterPlanes.size()))
        {
            for (int i = 0; i < vpOuterPlanes.size(); ++i) {
                if (!vpOuterPlanes[i] || vpOuterPlanes[i]->isBad()) {
                    continue;
                }
                // Tracking中,每个plane instance中被补充的点
                if (!vpOuterPlanes[i]->mvCompletePoints->points.empty()) {
                    // 反投影
                    vpOuterPlanes[i]->UpdateCompletePointsFrame(vpOuterPlanes[i]->mvCompletePoints, CurrentFrame);
                }
                else
                    continue;
            }
        }
        else if (!vpOuterPlanes.empty() && vpOuterPlanes.size() == vpMapPlanes.size())
        {
            for (auto& vMP : vpOuterPlanes) {
                vMP->UpdateCompletePointsFrame(vMP->mvCompletePoints, CurrentFrame);
            }
        }
        else if (vpOuterPlanes.empty() && !vpMapPlanes.empty())
        {
            for (int i = 0; i < vpMapPlanes.size(); ++i) {
                if (!vpMapPlanes[i] || vpMapPlanes[i]->isBad()) {
                    continue;
                }
                if (!vpMapPlanes[i]->mvCompletePoints->points.empty()) {
                    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
                    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
                    vpMapPlanes[i]->UpdateCompletePointsFrame(vpMapPlanes[i]->mvCompletePoints, CurrentFrame);
                }
            }
        }
        else
        {
            return;
        }
    }

    void Map::ComputeCrossPoint(const std::vector<MapPlane *> &vpMapPlanes, double threshold, double threshold1) {
        auto verThreshold = Config::Get<double>("Plane.MFVerticalThreshold");
        for (int i = 0; i < vpMapPlanes.size(); ++i) {
            if (!vpMapPlanes[i] || vpMapPlanes[i]->isBad()) {
                continue;
            }
            for (int j = i + 1; j < vpMapPlanes.size(); ++j) {
                int minSize = min(vpMapPlanes[i]->mvPlanePoints->points.size(),vpMapPlanes[j]->mvPlanePoints->points.size());
                cv::Mat plane1 = vpMapPlanes[i]->GetWorldPos();
                cv::Mat plane2 = vpMapPlanes[j]->GetWorldPos();
                if (!vpMapPlanes[j] || vpMapPlanes[j]->isBad()) {
                    continue;
                }
                float angle12 = plane1.at<float>(0,0)*plane2.at<float>(0,0) + plane1.at<float>(1,0)*plane2.at<float>(1,0)+
                              plane1.at<float>(2,0)*plane2.at<float>(2,0);
                double dis = PointDistanceFromPlane(plane1,vpMapPlanes[j]->mvPlanePoints, minSize);

                if ( (angle12 <= verThreshold && angle12 >= -verThreshold) || dis < threshold)
                {
                    for (int k = j+1; k < vpMapPlanes.size(); ++k) {
                        cv::Mat plane3 = vpMapPlanes[k]->GetWorldPos();
                        if (!vpMapPlanes[k] || vpMapPlanes[k]->isBad()) {
                            continue;
                        }
                        float angle13 = plane1.at<float>(0) * plane3.at<float>(0) +
                                        plane1.at<float>(1) * plane3.at<float>(1) +
                                        plane1.at<float>(2) * plane3.at<float>(2);

                        float angle23 = plane2.at<float>(0) * plane3.at<float>(0) +
                                        plane2.at<float>(1) * plane3.at<float>(1) +
                                        plane2.at<float>(2) * plane3.at<float>(2);
                        if (angle13 > verThreshold || angle13 < -verThreshold || angle23 > verThreshold || angle23 < -verThreshold) {
                            continue;
                        }
                        Eigen::Matrix<float, 3, 3> Xd, Yd, Zd, D;
                        D << plane1.at<float>(0), plane1.at<float>(1), plane1.at<float>(2),
                             plane2.at<float>(0), plane2.at<float>(1), plane2.at<float>(2),
                             plane3.at<float>(0), plane3.at<float>(1), plane3.at<float>(2);
                        auto resultD = D.determinant();
                        Xd << -plane1.at<float>(3), plane1.at<float>(1), plane1.at<float>(2),
                              -plane2.at<float>(3), plane2.at<float>(1), plane2.at<float>(2),
                              -plane3.at<float>(3), plane3.at<float>(1), plane3.at<float>(2);
                        auto resultXd = Xd.determinant();
                        Yd << plane1.at<float>(0), -plane1.at<float>(3), plane1.at<float>(2),
                              plane2.at<float>(0), -plane2.at<float>(3), plane2.at<float>(2),
                              plane3.at<float>(0), -plane3.at<float>(3), plane3.at<float>(2);
                        auto resultYd = Yd.determinant();
                        Zd << plane1.at<float>(0), plane1.at<float>(1), -plane1.at<float>(3),
                              plane2.at<float>(0), plane2.at<float>(1), -plane2.at<float>(3),
                              plane3.at<float>(0), plane3.at<float>(1), -plane3.at<float>(3);
                        auto resultZd = Zd.determinant();
                        if (resultD != 0)
                        {
                            float x = resultXd / resultD;
                            float y = resultYd / resultD;
                            float z = resultZd / resultD;
                            cv::Mat CrossPoint = (cv::Mat_<float>(3, 1) << x, y, z);
//                            cout << "交点的坐标" << "       " << CrossPoint << endl;
                            AddCrossPoint(CrossPoint);
                        }
                    }
                }
            }
        }
    }

    void Map::ComputeCrossLine(const std::vector<MapPlane*> &vpMapPlanes, double threshold, double threshold1) {

        // all planes in the map
        BoundaryPoints.clear();
        mspBoundaryLines.clear();

        for(int i = 0; i < vpMapPlanes.size(); i++) {

            for (int j = i+1; j < vpMapPlanes.size(); j++) {
                PointCloud::Ptr boundary (new PointCloud());

                // 两个平面的边缘点
                PointCloud::Ptr boundary_i (new PointCloud());
                PointCloud::Ptr boundary_j (new PointCloud());

                // 两自两个平面的四个点
                PointCloud::Ptr boundary_ij (new PointCloud());

                int minSize = min(vpMapPlanes[i]->mvPlanePoints->points.size(),vpMapPlanes[j]->mvPlanePoints->points.size());
                cv::Mat p1 = vpMapPlanes[i]->GetWorldPos(); // normalized
                cv::Mat p2 = vpMapPlanes[j]->GetWorldPos(); // normalized already
                float angle = p1.at<float>(0,0)*p2.at<float>(0,0) + p1.at<float>(1,0)*p2.at<float>(1,0)+
                              p1.at<float>(2,0)*p2.at<float>(2,0);

                double dis = PointDistanceFromPlane(p1,vpMapPlanes[j]->mvPlanePoints, minSize);

                if (angle < 0.28716 || dis < threshold) {
                    threshold = dis;
                    if (threshold < 0.5)
                        threshold = 0.5;
                    // step1:
                    // boundary points of the jth plane
                    for (auto p : vpMapPlanes[j]->mvPlanePoints->points) {
                        if (PointToPlaneDistance(p1, p) < threshold1)
                        {
                            threshold1 = PointToPlaneDistance(p1, p);
                            if (threshold1 < 0.05)
                                threshold1 = 0.05;
                            //boundary->points.emplace_back(p);
                            boundary_j->points.emplace_back(p);
                            //AddBoundaryPoints(p);
                        }
                    }

                    if(boundary_j->points.size() < 4)
                        continue;

                    std::sort(boundary_j->points.begin(), boundary_j->points.end(), SetSortZ);
                    pcl::PointXYZRGB point1_j = boundary_j->points[0];
                    pcl::PointXYZRGB point2_j = boundary_j->points[boundary_j->points.size() - 1];

                    //endpoints for the plane
                    vector<pcl::PointXYZRGB> endpoints_j;
                    endpoints_j.push_back(point1_j);
                    endpoints_j.push_back(point2_j);

                    boundary_ij->points.emplace_back(point1_j);
                    boundary_ij->points.emplace_back(point2_j);

                    // step 2
                    // boundary_i：第ｉ个平面上的点集中 距离j平面最近的点．也就是说,平面ｉ上贴近平面j的边界点
                    for (auto pp : vpMapPlanes[i]->mvPlanePoints->points) {

                        if (PointToPlaneDistance(p2, pp) < threshold1)
                        {
                            threshold1 = PointToPlaneDistance(p2, pp);
                            if (threshold1 < 0.1)
                                threshold1 = 0.1;
                            //boundary->points.emplace_back(pp);
                            boundary_i->points.emplace_back(pp);
                            //AddBoundaryPoints(pp);
                        }
                    }

                    // find the boundary
                    if(boundary_i->points.size() < 4)
                        continue;
                    std::sort(boundary_i->points.begin(), boundary_i->points.end(), SetSortZ);
                    pcl::PointXYZRGB point1_i = boundary_i->points[0];
                    pcl::PointXYZRGB point2_i = boundary_i->points[boundary_i->points.size() - 1];

                    //endpoints for the plane
                    vector<pcl::PointXYZRGB> endpoints_i;
                    endpoints_i.push_back(point1_i);
                    endpoints_i.push_back(point2_i);


                    boundary_ij->points.emplace_back(point1_i);
                    boundary_ij->points.emplace_back(point2_i);


                    // 四个端点都有
                    if(boundary_ij->points.size() == 4) //
                    {
                        std::sort(boundary_ij->points.begin(), boundary_ij->points.end(), SetSortZ);
                        // intersection of two boundaries ;
                        // 交集
                        pcl::PointXYZRGB point1 = boundary_ij->points[1];
                        pcl::PointXYZRGB point2 = boundary_ij->points[2];

                        // projection points
                        pcl::PointXYZRGB ProjectLeft1, ProjectLeft2, ProjectRight1, ProjectRight2;
                        pcl::PointXYZRGB UpCrossPoint, DownCrossPoint;
                        float tLeft1 = (vpMapPlanes[i]->GetWorldPos()).at<float>(0, 0) * point1.x +
                                       (vpMapPlanes[i]->GetWorldPos()).at<float>(1, 0) * point1.y +
                                       (vpMapPlanes[i]->GetWorldPos()).at<float>(2, 0) * point1.z +
                                       (vpMapPlanes[i]->GetWorldPos()).at<float>(3, 0);
                        float tLeft2 = (vpMapPlanes[i]->GetWorldPos()).at<float>(0, 0) * point2.x +
                                       (vpMapPlanes[i]->GetWorldPos()).at<float>(1, 0) * point2.y +
                                       (vpMapPlanes[i]->GetWorldPos()).at<float>(2, 0) * point2.z +
                                       (vpMapPlanes[i]->GetWorldPos()).at<float>(3, 0);
                        float tRight1 = (vpMapPlanes[j]->GetWorldPos()).at<float>(0, 0) * point1.x +
                                        (vpMapPlanes[j]->GetWorldPos()).at<float>(1, 0) * point1.y +
                                        (vpMapPlanes[j]->GetWorldPos()).at<float>(2, 0) * point1.z +
                                        (vpMapPlanes[j]->GetWorldPos()).at<float>(3, 0);
                        float tRight2 = (vpMapPlanes[j]->GetWorldPos()).at<float>(0, 0) * point2.x +
                                        (vpMapPlanes[j]->GetWorldPos()).at<float>(1, 0) * point2.y +
                                        (vpMapPlanes[j]->GetWorldPos()).at<float>(2, 0) * point2.z +
                                        (vpMapPlanes[j]->GetWorldPos()).at<float>(3, 0);
                        ProjectLeft1.x = point1.x - (vpMapPlanes[i]->GetWorldPos()).at<float>(0, 0) * tLeft1;
                        ProjectLeft1.y = point1.y - (vpMapPlanes[i]->GetWorldPos()).at<float>(1, 0) * tLeft1;
                        ProjectLeft1.z = point1.z - (vpMapPlanes[i]->GetWorldPos()).at<float>(2, 0) * tLeft1;
                        ProjectRight1.x = point1.x - (vpMapPlanes[j]->GetWorldPos()).at<float>(0, 0) * tRight1;
                        ProjectRight1.y = point1.y - (vpMapPlanes[j]->GetWorldPos()).at<float>(1, 0) * tRight1;
                        ProjectRight1.z = point1.z - (vpMapPlanes[j]->GetWorldPos()).at<float>(2, 0) * tRight1;
                        ProjectLeft2.x = point2.x - (vpMapPlanes[i]->GetWorldPos()).at<float>(0, 0) * tLeft2;
                        ProjectLeft2.y = point2.y - (vpMapPlanes[i]->GetWorldPos()).at<float>(1, 0) * tLeft2;
                        ProjectLeft2.z = point2.z - (vpMapPlanes[i]->GetWorldPos()).at<float>(2, 0) * tLeft2;
                        ProjectRight2.x = point2.x - (vpMapPlanes[j]->GetWorldPos()).at<float>(0, 0) * tRight2;
                        ProjectRight2.y = point2.y - (vpMapPlanes[j]->GetWorldPos()).at<float>(1, 0) * tRight2;
                        ProjectRight2.z = point2.z - (vpMapPlanes[j]->GetWorldPos()).at<float>(2, 0) * tRight2;



                        float startPo1x =0, startPo1y =0, startPo1z = 0;
                        float startPo2x =0, startPo2y =0, startPo2z = 0;
                        if (abs(angle)<0.08)  //两个平面垂直
                        {
                            cout<<endl<< "there are perpendicular planes"<<endl;
                            //
//                            cout<<ProjectLeft1.x<<endl<<point1.x<<endl<<ProjectLeft1.y<<endl<<point1.y<<endl<<ProjectLeft1.z<<point1.z<<endl;
//                            cout<<ProjectLeft1.x<<endl<<point2.x<<endl<<ProjectLeft1.y<<endl<<point2.y<<endl<<ProjectLeft1.z<<point2.z<<endl;

                            if(ComputeDistance(ProjectLeft1,point1)<0.05)
                            {
                                // ProjectRight1 is the point
                                cout<<"1"<<endl;
                                startPo1x = ProjectRight1.x;
                                startPo1y = ProjectRight1.y;
                                startPo1z = ProjectRight1.z;
                            } else{
                                cout<<"2:"<<ProjectLeft1.x-point1.x<<endl;
                                startPo1x = ProjectLeft1.x;
                                startPo1y = ProjectLeft1.y;
                                startPo1z = ProjectLeft1.z;
                            }

                            if(ComputeDistance(ProjectLeft2,point2)<0.05)
                            {
                                // ProjectRight1 is the point
                                cout<<"3"<<endl;
                                startPo2x = ProjectRight2.x;
                                startPo2y = ProjectRight2.y;
                                startPo2z = ProjectRight2.z;
                            } else{
                                cout<<"4: "<<ProjectLeft2.y-point2.y<<endl;
                                startPo2x = ProjectLeft2.x;
                                startPo2y = ProjectLeft2.y;
                                startPo2z = ProjectLeft2.z;
                            }

                        }
                        else{
                            cout<<endl<<"compute new points"<<endl;

                            if(ComputeDistance(ProjectLeft1,ProjectRight1)<0.05)
                            {
                                startPo1x = (ProjectLeft1.x+ProjectRight1.x)/2;
                                startPo1y = (ProjectLeft1.y+ProjectRight1.y)/2;
                                startPo1z = (ProjectLeft1.z+ProjectRight1.z)/2;
                            } else{
                                cv::Mat pN1 = vpMapPlanes[i]->GetWorldPos();
                                cv::Mat pN2 = vpMapPlanes[j]->GetWorldPos();

                                //compute endpoints
                                // ProjectRight1 is the project point.
                                pcl::PointXYZRGB startPo1;
                                cout<<"..."<<ProjectLeft1.x-point1.x<<", "<<ProjectLeft1.y-point1.y<<endl;
                                if(ProjectLeft1.x-point1.x==ProjectLeft1.y-point1.y==ProjectLeft1.z-point1.z)
                                {
                                    cout<<"yes ..."<<endl;
                                    startPo1 = ComputeEndpoint4(pN1,pN2,ProjectLeft1,ProjectRight1);
                                } else
                                    startPo1 = ComputeEndpoint4(pN1,pN2,ProjectRight1,ProjectLeft1);

                                startPo1x = startPo1.x;
                                startPo1y = startPo1.y;
                                startPo1z = startPo1.z;
                            }

                            if(ComputeDistance(ProjectLeft2,ProjectRight2)<0.05)
                            {
                                startPo2x = (ProjectLeft2.x+ProjectRight2.x)/2;
                                startPo2y = (ProjectLeft2.y+ProjectRight2.y)/2;
                                startPo2z = (ProjectLeft2.z+ProjectRight2.z)/2;
                            } else{
                                cout<<endl<<"compute new points"<<endl;
                                cv::Mat pN1 = vpMapPlanes[i]->GetWorldPos();
                                cv::Mat pN2 = vpMapPlanes[j]->GetWorldPos();
                                //compute endpoints
                                pcl::PointXYZRGB startPo2;
                                if(ProjectLeft2.x-point2.x==ProjectLeft2.y-point2.y==ProjectLeft2.z-point2.z)
                                    startPo2 = ComputeEndpoint4(pN1,pN2,ProjectLeft2,ProjectRight2);
                                else
                                    startPo2 = ComputeEndpoint4(pN1,pN2,ProjectRight2,ProjectLeft2);

                                startPo2x = startPo2.x;
                                startPo2y = startPo2.y;
                                startPo2z = startPo2.z;
                            }
                        }

                        Eigen::Matrix<double, 6, 1> boundaryLine;     // two endpoints of the line
                        Eigen::Matrix<double, 3, 1> DirectionVector;  //


                        boundaryLine << startPo1x, startPo1y, startPo1z, startPo2x, startPo2y, startPo2z;
                        double base = sqrt(pow((startPo2x - startPo1x), 2) + pow((startPo2y - startPo1y), 2) + pow((startPo2z - startPo1z), 2));

                        DirectionVector << (startPo2x -startPo1x)/base, (startPo2y - startPo1y)/base, (startPo2x - startPo1z)/base;


                        // 访问平面现在的交线情况
                        bool existdLine = false;
                        //Eigen::Matrix<double, 6, 1> boundaryLine1;
                        vector<std::tuple< Eigen::Matrix<double, 6, 1>, int>> vt_i = vpMapPlanes[i]->GetCrossLines();

                        for(size_t m =0; m<vt_i.size();m++)
                        {
                            Eigen::Matrix<double, 6, 1> boundaryLine1 = std::get<0>(vt_i[m]);
                            auto id = std::get<1>(vt_i[m]);

                            AddBoundaryLine(boundaryLine1);

                            //cout<<"id:"<<endl<<endl<<id<<","<<vpMapPlanes[j]->mnId<<endl;
                            if(vpMapPlanes[j]->mnId==id)
                            {
                                //cout<<"yes"<<endl;
                                // existed cross line between i and j
                                // 已经存在他们之间的交线
                                existdLine = true;
                                // update endpoints
                                PointCloud::Ptr boundary_update (new PointCloud());
                                pcl::PointXYZRGB endPoint1, endPoint2;
                                endPoint1.x = boundaryLine1(0,0); endPoint1.y = boundaryLine1(1,0); endPoint1.z = boundaryLine1(2,0);
                                boundary_update->points.emplace_back(endPoint1);
                                endPoint2.x = boundaryLine1(3,0); endPoint2.y = boundaryLine1(4,0); endPoint2.z = boundaryLine1(5,0);
                                boundary_update->points.emplace_back(endPoint2);
                                for (int n = 0; n<boundary_i->points.size();n++)
                                {
                                    //cout<<"boundary_update:"<<boundary_update->points.size()<<endl;
                                    boundary_update->points.emplace_back(boundary_i->points[n]);
                                }
                                std::sort(boundary_update->points.begin(), boundary_update->points.end(), SetSortZ);
                                // new endpoints
                                pcl::PointXYZRGB point1_update = boundary_update->points[0];
                                pcl::PointXYZRGB point2_update = boundary_update->points[boundary_update->points.size() - 1];

                                Eigen::Matrix<double, 6, 1> boundaryLine_update;     // two endpoints of the line
                                boundaryLine_update << point1_update.x, point1_update.y, point1_update.z, point2_update.x, point2_update.y, point2_update.z;
                                // reset endpoints
                                vpMapPlanes[i]->UpdateCrossLine(boundaryLine_update,m);

                            }



                        }

                        vector<std::tuple< Eigen::Matrix<double, 6, 1>, int>> vt_j = vpMapPlanes[j]->GetCrossLines();
                        for(size_t m =0; m<vt_j.size();m++)
                        {
                            Eigen::Matrix<double, 6, 1> boundaryLine1 = std::get<0>(vt_j[m]);
                            auto id = std::get<1>(vt_j[m]);

                            AddBoundaryLine(boundaryLine1);


                            if(vpMapPlanes[j]->mnId==id)
                            {
                                // existed cross line between i and j
                                // 已经存在他们之间的交线
                                existdLine = true;
                                // update endpoints

                                PointCloud::Ptr boundary_update (new PointCloud());
                                pcl::PointXYZRGB endPoint1, endPoint2;
                                endPoint1.x = boundaryLine1(0,0); endPoint1.y = boundaryLine1(1,0); endPoint1.z = boundaryLine1(2,0);
                                boundary_update->points.emplace_back(endPoint1);
                                endPoint2.x = boundaryLine1(3,0); endPoint2.y = boundaryLine1(4,0); endPoint2.z = boundaryLine1(5,0);
                                boundary_update->points.emplace_back(endPoint2);
                                for (int n = 0; n<boundary_j->points.size();n++)
                                {
                                    //cout<<"boundary_update:"<<boundary_update->points.size()<<endl;
                                    boundary_update->points.emplace_back(boundary_j->points[n]);
                                }
                                std::sort(boundary_update->points.begin(), boundary_update->points.end(), SetSortZ);
                                // new endpoints
                                pcl::PointXYZRGB point1_update = boundary_update->points[0];
                                pcl::PointXYZRGB point2_update = boundary_update->points[boundary_update->points.size() - 1];

                                Eigen::Matrix<double, 6, 1> boundaryLine_update;     // two endpoints of the line
                                boundaryLine_update << point1_update.x, point1_update.y, point1_update.z, point2_update.x, point2_update.y, point2_update.z;
                                // reset endpoints
                                vpMapPlanes[j]->UpdateCrossLine(boundaryLine_update,m);
                            }


                        }

                        if(!existdLine) {
                            vpMapPlanes[i]->AddCrossLines(boundaryLine, j);
                            vpMapPlanes[j]->AddCrossLines(boundaryLine, i);
                        }

                        // visulization
                        {
                            //AddBoundaryLine(boundaryLine);

                            for (int n = 0; n < boundary_i->points.size(); ++n)
                            {
                                AddBoundaryPoints(boundary_i->points[n]);
                            }
                            for (int n = 0; n < boundary_j->points.size(); ++n)
                            {
                                AddBoundaryPoints(boundary_j->points[n]);
                            }
                        }

//                        cout << "finish boundary line" << endl;
                    }
                }
            }
        }
//        cout << "finish this step" << endl;
    }

    bool Map::CheckParallel(Eigen::Matrix<double, 3, 1> &DirectionVector1, Eigen::Matrix<double, 3, 1> &DirectionVector2)
    {
        double angle = abs(DirectionVector1(0,0)*DirectionVector2(0,0)+DirectionVector1(1,0)*DirectionVector2(1,0)+DirectionVector1(2,0)*DirectionVector2(2,0));
        bool parallel = false;
        if(angle<0.1)
        {
            parallel = true;
        }
        return parallel;
    }

    double Map::ComputeDistance(pcl::PointXYZRGB &p1, pcl::PointXYZRGB&p2)
    {
        return sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2)+pow(p1.z-p2.z,2));
    }

    pcl::PointXYZRGB Map::ComputeEndpoint(cv::Mat &pN1, cv::Mat &pN2,pcl::PointXYZRGB &ProjectLeft1,pcl::PointXYZRGB &ProjectRight1)
    {
        /*  projectleft1  8
                        *                8  8
                        *                8     8
                        *  projectright1 88888888 unknown endpoint1 X (x,y,z)^T
                        *
                        *
                        * AX = b
                        * n_1*X = -d
                        * n_2*X = -d
                        * [x_a-x_b, y_a-y_b,z_a-z_b] * X = x_a*x_b+y_a*y_b+z_a*z_b-x_b^2-y_b^2-z_b^2
                        * */
        cout<<"pN1"<<pN1<<endl;

        cv::Mat temp_pN1 = (cv::Mat_<float>(3, 1) << pN1.at<float>(0), pN1.at<float>(1), pN1.at<float>(2));
        cv::Mat temp_pN2 = (cv::Mat_<float>(3, 1) << pN2.at<float>(0), pN2.at<float>(1), pN2.at<float>(2));

        cv::Mat CrossLine = temp_pN1.cross(temp_pN2);
        cv::Mat A, bup, bdown;

        // Ax = b
        A = cv::Mat::eye(cv::Size(3, 3), CV_32F);

        A.at<float>(0, 0) = pN1.at<float>(0);
        A.at<float>(1, 0) = pN2.at<float>(0);
        A.at<float>(2, 0) = ProjectLeft1.x-ProjectRight1.x;
        A.at<float>(0, 1) = pN1.at<float>(1);
        A.at<float>(1, 1) = pN2.at<float>(1);
        A.at<float>(2, 1) = ProjectLeft1.y-ProjectRight1.y;
        A.at<float>(0, 2) = pN1.at<float>(2);
        A.at<float>(1, 2) = pN2.at<float>(2);
        A.at<float>(2, 2) = ProjectLeft1.z-ProjectRight1.z;


        float b1 = -pN1.at<float>(3);
        float b2 = -pN2.at<float>(3);
        float b3 = ProjectLeft1.x*ProjectRight1.x+ProjectLeft1.y*ProjectRight1.y+ProjectLeft1.z*ProjectRight1.z-ProjectRight1.x*ProjectRight1.x-ProjectRight1.y*ProjectRight1.y-ProjectRight1.z*ProjectRight1.z;//_b^2-z_b^2
        bup = (cv::Mat_<float>(3, 1) << b1, b2, b3);
        cout<<"A:"<<A<<endl<<"bup:"<<bup<<endl;

        /**
         * [ a11 a12 a13
         *   a21 a22 a23    x = b
         *   a31 a32 a33]
         *
         * **/
        float a11 = pN1.at<float>(0);
        float a21 = pN2.at<float>(0);
        float a31 = ProjectLeft1.x-ProjectRight1.x;
        float a12 = pN1.at<float>(1);
        float a22 = pN2.at<float>(1);
        float a32 = ProjectLeft1.y-ProjectRight1.y;
        float a13 = pN1.at<float>(2);
        float a23 = pN2.at<float>(2);
        float a33 = ProjectLeft1.z-ProjectRight1.z;

        /**
        * [ a11 a12 a13
        *   0 e f   x = b
        *   0 g h]
        *
        * **/
        float e = a22/a21*a11-a12;
        float f = a23/a21*a11-a13;
        b2 = b2/a21*a11 - b1;
        float g = a32/a31*a11-a12;
        float h = a33/a31*a11-a13;
        b3 = b3/a31*a11 - b1;

        /**
        * [ a11 a12 a13
        *   0   e   f       x = b
        *   0   0   j]
        *
        * **/
        float j = h/g*e-f;
        b3 = b3/g*e - b2;
        cout<<"j:"<<j<<endl;
        pcl::PointXYZRGB UpCrossPoint;
        if(abs(j)>0.0001)
        {
            UpCrossPoint.z = b3/j;
            UpCrossPoint.y = (b2-f*UpCrossPoint.z)/e;
            UpCrossPoint.x = (b1-a13*UpCrossPoint.z-a12*UpCrossPoint.y)/a11;
        } else{
            UpCrossPoint.z = 0;
            UpCrossPoint.y = 0;//(b2-f*UpCrossPoint.z)/e;
            UpCrossPoint.x = 0;//(b1-a13*UpCrossPoint.z-a12*UpCrossPoint.y)/a11;
        }



        cout<<"A:"<<A<<endl<<A.inv()<<endl;
        cout<<"cross point:"<<UpCrossPoint.x<<","<<UpCrossPoint.y<<","<<UpCrossPoint.z<<endl;
        cout<<"A:"<<A<<endl<<"bup:"<<bup<<endl;
        cout<<"ProjectLeft1"<<ProjectLeft1<<endl<<ProjectRight1<<endl;
        cout<<A.at<float>(0, 0)*UpCrossPoint.x+A.at<float>(0, 1)*UpCrossPoint.y+A.at<float>(0, 2)*UpCrossPoint.z<<endl;
        cout<<A.at<float>(1, 0)*UpCrossPoint.x+A.at<float>(1, 1)*UpCrossPoint.y+A.at<float>(1, 2)*UpCrossPoint.z<<endl;
        cout<<A.at<float>(2, 0)*UpCrossPoint.x+A.at<float>(2, 1)*UpCrossPoint.y+A.at<float>(2, 2)*UpCrossPoint.z<<endl;
        cout<<"check reprojection point"<<endl;
        cout<<A.at<float>(0, 0)*ProjectLeft1.x+A.at<float>(0, 1)*ProjectLeft1.y+A.at<float>(0, 2)*ProjectLeft1.z<<endl;
        cout<<A.at<float>(1, 0)*ProjectRight1.x+A.at<float>(1, 1)*ProjectRight1.y+A.at<float>(1, 2)*ProjectRight1.z<<endl;

        return UpCrossPoint;
    }

    pcl::PointXYZRGB Map::ComputeEndpoint2(cv::Mat &pN1, cv::Mat &pN2,pcl::PointXYZRGB &ProjectLeft1,pcl::PointXYZRGB &ProjectRight1)
    {
        /*  projectleft1                 8
                        *                8  8
                        *                8     8
                        *  projectright1 88888888 unknown endpoint1 X (x,y,z)^T
                        *
                        *
                        * AX = b
                        * n_1*X = -d
                        * n_2*X = -d
                        * [x_a-x_b, y_a-y_b,z_a-z_b] * X = x_a*x_b+y_a*y_b+z_a*z_b-x_b^2-y_b^2-z_b^2
                        * */
        cout<<"pN1"<<pN1<<endl;

        cv::Mat temp_pN1 = (cv::Mat_<float>(3, 1) << pN1.at<float>(0), pN1.at<float>(1), pN1.at<float>(2));
        cv::Mat temp_pN2 = (cv::Mat_<float>(3, 1) << pN2.at<float>(0), pN2.at<float>(1), pN2.at<float>(2));

        cv::Mat CrossLine = temp_pN1.cross(temp_pN2);
        cv::Mat A, bup, bdown;

        // Ax = b
        A = cv::Mat::eye(cv::Size(3, 3), CV_32F);

        A.at<float>(0, 0) = pN1.at<float>(0);
        A.at<float>(1, 0) = pN2.at<float>(0);
        A.at<float>(2, 0) = ProjectLeft1.x-ProjectRight1.x;
        A.at<float>(0, 1) = pN1.at<float>(1);
        A.at<float>(1, 1) = pN2.at<float>(1);
        A.at<float>(2, 1) = ProjectLeft1.y-ProjectRight1.y;
        A.at<float>(0, 2) = pN1.at<float>(2);
        A.at<float>(1, 2) = pN2.at<float>(2);
        A.at<float>(2, 2) = ProjectLeft1.z-ProjectRight1.z;


        float b1 = -pN1.at<float>(3);
        float b2 = -pN2.at<float>(3);
        float b3 = ProjectLeft1.x*ProjectRight1.x+ProjectLeft1.y*ProjectRight1.y+ProjectLeft1.z*ProjectRight1.z-ProjectRight1.x*ProjectRight1.x-ProjectRight1.y*ProjectRight1.y-ProjectRight1.z*ProjectRight1.z;//_b^2-z_b^2
        bup = (cv::Mat_<float>(3, 1) << b1, b2, b3);

        /**
         * [ a11 a12 a13
         *   a21 a22 a23    x = b
         *   a31 a32 a33]
         *
         * **/
        float a = pN1.at<float>(0);
        float b = pN2.at<float>(0);
        float c = ProjectLeft1.x-ProjectRight1.x;
        float d = pN1.at<float>(1);
        float e = pN2.at<float>(1);
        float f = ProjectLeft1.y-ProjectRight1.y;
        float g = pN1.at<float>(2);
        float h = pN2.at<float>(2);
        float i = ProjectLeft1.z-ProjectRight1.z;

        /**
        * [ 1 a12/a11 a13/a11 b1/a11
        *   0 e f  b2
        *   0 g h  b3]
        *
        *  [a b c d
         *  e f d e ]
        *
        * **/
        float x = (f-c/a*d)/(e-b/a*d);
        float y = (b2-b1/a*d)/(e-b/a*d);
        float m = (i-c/a*g)/(h-c/a*g);
        float n = (f-c/a*d)/(h-c/a*g);

        pcl::PointXYZRGB UpCrossPoint;

        UpCrossPoint.z = (n-y)/(m-x);
        UpCrossPoint.y = y-UpCrossPoint.z*x;//(b2-f*UpCrossPoint.z)/e;
        UpCrossPoint.x = b1/a-c/a*UpCrossPoint.z-b/a*UpCrossPoint.y;


        cout<<"cross point:"<<UpCrossPoint.x<<","<<UpCrossPoint.y<<","<<UpCrossPoint.z<<endl;
        cout<<"A:"<<A<<endl<<"bup:"<<bup<<endl;
        cout<<"ProjectLeft1"<<ProjectLeft1<<endl<<ProjectRight1<<endl;
        cout<<A.at<float>(0, 0)*UpCrossPoint.x+A.at<float>(0, 1)*UpCrossPoint.y+A.at<float>(0, 2)*UpCrossPoint.z<<endl;
        cout<<A.at<float>(1, 0)*UpCrossPoint.x+A.at<float>(1, 1)*UpCrossPoint.y+A.at<float>(1, 2)*UpCrossPoint.z<<endl;
        cout<<A.at<float>(2, 0)*UpCrossPoint.x+A.at<float>(2, 1)*UpCrossPoint.y+A.at<float>(2, 2)*UpCrossPoint.z<<endl;
        cout<<"check reprojection point"<<endl;
        cout<<A.at<float>(0, 0)*ProjectLeft1.x+A.at<float>(0, 1)*ProjectLeft1.y+A.at<float>(0, 2)*ProjectLeft1.z<<endl;
        cout<<A.at<float>(1, 0)*ProjectRight1.x+A.at<float>(1, 1)*ProjectRight1.y+A.at<float>(1, 2)*ProjectRight1.z<<endl;

        return UpCrossPoint;
    }
    pcl::PointXYZRGB Map::ComputeEndpoint3(cv::Mat &pN1, cv::Mat &pN2,pcl::PointXYZRGB &ProjectLeft1,pcl::PointXYZRGB &ProjectRight1)
    {
        /*  projectleft1                 8
                        *                8  8
                        *                8     8
                        *  projectright1 88888888 unknown endpoint1 X (x,y,z)^T
                        *
                        *
                        * AX = b
                        * n_1*X = -d
                        * n_2*X = -d
                        * [x_a-x_b, y_a-y_b,z_a-z_b] * X = x_a*x_b+y_a*y_b+z_a*z_b-x_b^2-y_b^2-z_b^2
                        * */
        cout<<"pN1"<<pN1<<endl;

        cv::Mat temp_pN1 = (cv::Mat_<float>(3, 1) << pN1.at<float>(0), pN1.at<float>(1), pN1.at<float>(2));
        cv::Mat temp_pN2 = (cv::Mat_<float>(3, 1) << pN2.at<float>(0), pN2.at<float>(1), pN2.at<float>(2));

        cv::Mat CrossLine = temp_pN1.cross(temp_pN2);
        cv::Mat A, bup, bdown;

        // Ax = b
        A = cv::Mat::eye(cv::Size(3, 3), CV_32F);

        A.at<float>(0, 0) = pN1.at<float>(0);
        A.at<float>(1, 0) = pN2.at<float>(0);
        A.at<float>(2, 0) = ProjectLeft1.x-ProjectRight1.x;
        A.at<float>(0, 1) = pN1.at<float>(1);
        A.at<float>(1, 1) = pN2.at<float>(1);
        A.at<float>(2, 1) = ProjectLeft1.y-ProjectRight1.y;
        A.at<float>(0, 2) = pN1.at<float>(2);
        A.at<float>(1, 2) = pN2.at<float>(2);
        A.at<float>(2, 2) = ProjectLeft1.z-ProjectRight1.z;


        float b1 = -pN1.at<float>(3);
        float b2 = -pN2.at<float>(3);
        float b3 = ProjectLeft1.x*ProjectRight1.x+ProjectLeft1.y*ProjectRight1.y+ProjectLeft1.z*ProjectRight1.z-ProjectRight1.x*ProjectRight1.x-ProjectRight1.y*ProjectRight1.y-ProjectRight1.z*ProjectRight1.z;//_b^2-z_b^2
        bup = (cv::Mat_<float>(3, 1) << b1, b2, b3);

        /**
         * [ a b c
         *   d e f    x = b
         *   g h i]
         *
         * **/
        float a = pN1.at<float>(0);
        float d = pN2.at<float>(0);
        float g = ProjectLeft1.x-ProjectRight1.x;
        float b = pN1.at<float>(1);
        float e = pN2.at<float>(1);
        float h = ProjectLeft1.y-ProjectRight1.y;
        float c = pN1.at<float>(2);
        float f = pN2.at<float>(2);
        float i = ProjectLeft1.z-ProjectRight1.z;

        /**
        * [ 1 x y z
        *   0 1 m  n
        *   0 0 p q]
        *
        * **/

        float x = b/a ;//(f-c/a*d)/(e-b/a*d);
        float y = c/a; //(b2-b1/a*d)/(e-b/a*d);
        float z = b1/a;

        float m = (a*f-c*d)/(a*e-b*d);//(i-c/a*g)/(h-c/a*g);
        float n = (b2*a-b1*d)/(a*e-b*d);

        float p = (i*a-c*g)/(h*a-g*b)-m;
        float q = (b3*a-b1*g)/(h*a-g*b)-n;


        pcl::PointXYZRGB UpCrossPoint;

        UpCrossPoint.z = q/p;
        UpCrossPoint.y = n- UpCrossPoint.z*m; //y-UpCrossPoint.z*x;//(b2-f*UpCrossPoint.z)/e;
        UpCrossPoint.x = z - UpCrossPoint.y*y-UpCrossPoint.x*x;//b1/a-c/a*UpCrossPoint.z-b/a*UpCrossPoint.y;


        cout<<"cross point:"<<UpCrossPoint.x<<","<<UpCrossPoint.y<<","<<UpCrossPoint.z<<endl;
        cout<<"A:"<<A<<endl<<"bup:"<<bup<<endl;
        cout<<"ProjectLeft1"<<ProjectLeft1<<endl<<ProjectRight1<<endl;
        cout<<A.at<float>(0, 0)*UpCrossPoint.x+A.at<float>(0, 1)*UpCrossPoint.y+A.at<float>(0, 2)*UpCrossPoint.z<<endl;
        cout<<A.at<float>(1, 0)*UpCrossPoint.x+A.at<float>(1, 1)*UpCrossPoint.y+A.at<float>(1, 2)*UpCrossPoint.z<<endl;
        cout<<A.at<float>(2, 0)*UpCrossPoint.x+A.at<float>(2, 1)*UpCrossPoint.y+A.at<float>(2, 2)*UpCrossPoint.z<<endl;
        cout<<"check reprojection point"<<endl;
        cout<<A.at<float>(0, 0)*ProjectLeft1.x+A.at<float>(0, 1)*ProjectLeft1.y+A.at<float>(0, 2)*ProjectLeft1.z<<endl;
        cout<<A.at<float>(1, 0)*ProjectRight1.x+A.at<float>(1, 1)*ProjectRight1.y+A.at<float>(1, 2)*ProjectRight1.z<<endl;

        return UpCrossPoint;
    }


    pcl::PointXYZRGB Map::ComputeEndpoint4(cv::Mat &pN1, cv::Mat &pN2,pcl::PointXYZRGB &ProjectLeft1,pcl::PointXYZRGB &ProjectRight1)
    {
        /*  projectleft1                 8
                        *                8  8
                        *                8     8
                        *  projectright1 88888888 unknown endpoint1 X (x,y,z)^T
                        *
                        *
                        * AX = b
                        * n_1*X = -d
                        * n_2*X = -d
                        * [x_a-x_b, y_a-y_b,z_a-z_b] * X = x_a*x_b+y_a*y_b+z_a*z_b-x_b^2-y_b^2-z_b^2
                        * */
        cout<<"pN1"<<endl<<"ComputeEndpoint4"<<pN1<<endl;

        cv::Mat temp_pN1 = (cv::Mat_<float>(3, 1) << pN1.at<float>(0), pN1.at<float>(1), pN1.at<float>(2));
        cv::Mat temp_pN2 = (cv::Mat_<float>(3, 1) << pN2.at<float>(0), pN2.at<float>(1), pN2.at<float>(2));

        cv::Mat CrossLine = temp_pN1.cross(temp_pN2);
        cout<<"CrossLine:"<<CrossLine<<endl;
        cv::Mat A, bup, bdown;

        // Ax = b
        A = cv::Mat::eye(cv::Size(3, 3), CV_32F);

        A.at<float>(0, 0) = pN1.at<float>(0);
        A.at<float>(1, 0) = pN2.at<float>(0);
        A.at<float>(2, 0) = CrossLine.at<float>(0,0);
        A.at<float>(0, 1) = pN1.at<float>(1);
        A.at<float>(1, 1) = pN2.at<float>(1);
        A.at<float>(2, 1) = CrossLine.at<float>(1,0);
        A.at<float>(0, 2) = pN1.at<float>(2);
        A.at<float>(1, 2) = pN2.at<float>(2);
        A.at<float>(2, 2) = CrossLine.at<float>(2,0);


        float b1 = -pN1.at<float>(3);
        float b2 = -pN2.at<float>(3);
        float b3 = ProjectLeft1.x*CrossLine.at<float>(0,0)+ProjectLeft1.y*CrossLine.at<float>(1,0)+ProjectLeft1.z*CrossLine.at<float>(2,0);//_b^2-z_b^2
        bup = (cv::Mat_<float>(3, 1) << b1, b2, b3);

        /**
         * [ a b c
         *   d e f    x = b
         *   g h i]
         *
         * **/
        float a = pN1.at<float>(0);
        float d = pN2.at<float>(0);
        float g = CrossLine.at<float>(0,0);
        float b = pN1.at<float>(1);
        float e = pN2.at<float>(1);
        float h = CrossLine.at<float>(1,0);
        float c = pN1.at<float>(2);
        float f = pN2.at<float>(2);
        float i = CrossLine.at<float>(2,0);

        /**
        * [ 1 x y z
        *   0 1 m  n
        *   0 0 p q]
        *
        * **/

        float x = b/a ;//(f-c/a*d)/(e-b/a*d);
        float y = c/a; //(b2-b1/a*d)/(e-b/a*d);
        float z = b1/a;

        float m = (a*f-c*d)/(a*e-b*d);//(i-c/a*g)/(h-c/a*g);
        float n = (b2*a-b1*d)/(a*e-b*d);

        float p = (i*a-c*g)/(h*a-g*b)-m;
        float q = (b3*a-b1*g)/(h*a-g*b)-n;


        pcl::PointXYZRGB UpCrossPoint;

        UpCrossPoint.z = q/p;
        UpCrossPoint.y = n- UpCrossPoint.z*m; //y-UpCrossPoint.z*x;//(b2-f*UpCrossPoint.z)/e;
        UpCrossPoint.x = z - UpCrossPoint.y*y-UpCrossPoint.x*x;//b1/a-c/a*UpCrossPoint.z-b/a*UpCrossPoint.y;


        cout<<"cross point:"<<UpCrossPoint.x<<","<<UpCrossPoint.y<<","<<UpCrossPoint.z<<endl;
        cout<<"A:"<<A<<endl<<"bup:"<<bup<<endl;
        cout<<"ProjectLeft1"<<ProjectLeft1<<endl<<ProjectRight1<<endl;
        cout<<A.at<float>(0, 0)*UpCrossPoint.x+A.at<float>(0, 1)*UpCrossPoint.y+A.at<float>(0, 2)*UpCrossPoint.z<<endl;
        cout<<A.at<float>(1, 0)*UpCrossPoint.x+A.at<float>(1, 1)*UpCrossPoint.y+A.at<float>(1, 2)*UpCrossPoint.z<<endl;
        cout<<A.at<float>(2, 0)*UpCrossPoint.x+A.at<float>(2, 1)*UpCrossPoint.y+A.at<float>(2, 2)*UpCrossPoint.z<<endl;
        cout<<"check reprojection point"<<endl;
        cout<<A.at<float>(0, 0)*ProjectLeft1.x+A.at<float>(0, 1)*ProjectLeft1.y+A.at<float>(0, 2)*ProjectLeft1.z<<endl;
        cout<<A.at<float>(1, 0)*ProjectRight1.x+A.at<float>(1, 1)*ProjectRight1.y+A.at<float>(1, 2)*ProjectRight1.z<<endl;

        return UpCrossPoint;
    }



    void Map::AddMapPlaneBoundary(MapPlane* pMP){
        unique_lock<mutex> lock(mMutexMap);
        pMP->cloud_boundary.get()->points;
        mspMapPlanesBoundaries.insert(pMP);
    }

    void Map::AddCube(pcl::PointXYZRGB &point) {
        unique_lock<mutex> lock(mMutexMap);
        DrawCube.emplace_back(point);
    }

    void Map::AddTupleCube(std::tuple<pcl::PointXYZRGB, pcl::PointXYZRGB> &tupleCube) {
        unique_lock<mutex> lock(mMutexMap);
        DrawTupleCube.emplace_back(tupleCube);
    }

    vector<pcl::PointXYZRGB> Map::GetAllCubes() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<pcl::PointXYZRGB>(DrawCube.begin(), DrawCube.end());
    }

    vector<std::tuple<pcl::PointXYZRGB, pcl::PointXYZRGB>> Map::GetAllTupleCubes() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<std::tuple<pcl::PointXYZRGB, pcl::PointXYZRGB>>(DrawTupleCube.begin(),DrawTupleCube.end());
    }

    void Map::EraseMapPlane(MapPlane *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPlanes.erase(pMP);
    }

    vector<MapPlane *> Map::GetAllMapPlanes() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapPlane *>(mspMapPlanes.begin(), mspMapPlanes.end());
    }

    vector<MapPlane *> Map::GetAllMapPlaneBoundary() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapPlane *>(mspMapPlanesBoundaries.begin(), mspMapPlanesBoundaries.end());
    }

    vector<pcl::PointXYZRGB> Map::GetAllInlierLines() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<pcl::PointXYZRGB>(InlierLines.begin(), InlierLines.end());
    }


    vector<cv::Mat> Map::GetAllCrossLines() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<cv::Mat>(CrossLineDraw.begin(),CrossLineDraw.end());
    }

    vector<pcl::PointXYZRGB> Map::GetAllBoundaryPoints() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<pcl::PointXYZRGB>(BoundaryPoints.begin(), BoundaryPoints.end());
    }

    vector<cv::Mat> Map::GetAllCrossPoints() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<cv::Mat>(CrossPointDraw.begin(),CrossPointDraw.end());
    }

    long unsigned int Map::MapPlanesInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspMapPlanes.size();
    }

    void Map::FlagMatchedPlanePoints(ORB_SLAM2::Frame &pF, const float &dTh) {
//match plane points based on the distance between the point and the plane
        unique_lock<mutex> lock(mMutexMap);
        int nMatches = 0;

        for (int i = 0; i < pF.mnPlaneNum; ++i) {

            cv::Mat pM = pF.ComputePlaneWorldCoeff(i);

            if (pF.mvpMapPlanes[i]) {
                for (auto mapPoint : mspMapPoints) {
                    cv::Mat pW = mapPoint->GetWorldPos();

                    double dis = abs(pM.at<float>(0, 0) * pW.at<float>(0, 0) +
                                     pM.at<float>(1, 0) * pW.at<float>(1, 0) +
                                     pM.at<float>(2, 0) * pW.at<float>(2, 0) +
                                     pM.at<float>(3, 0));

                    if (dis < 0.5) {
                        mapPoint->SetAssociatedWithPlaneFlag(true);
                        nMatches++;
                    }
                }
            }
        }

//        cout << "Point matches: " << nMatches << endl;
    }

    void Map::AddTuplePlaneObservation(MapPlane *pMP1, MapPlane *pMP2, MapPlane *pMP3, KeyFrame* pKF) {
        unique_lock<mutex> lock(mMutexMap);

        TuplePlane planes = std::make_tuple(pMP1, pMP2, pMP3);
        if (mmpTuplePlanesObservations.count(planes) != 0)
            return;
//        cout << "Insert Manhattan3 pMP1: " << pMP1->mnId << endl;
//        cout << "Insert Manhattan3 pMP2: " << pMP2->mnId << endl;
//        cout << "Insert Manhattan3 pMP3: " << pMP3->mnId << endl;
        pKF->SetNotErase();
        mmpTuplePlanesObservations[planes] = pKF;
    }

    KeyFrame* Map::GetTuplePlaneObservation(MapPlane *pMP1, MapPlane *pMP2, MapPlane *pMP3) {
        unique_lock<mutex> lock(mMutexMap);
        TuplePlane planes = std::make_tuple(pMP1, pMP2, pMP3);
        if (mmpTuplePlanesObservations.count(planes)) {
            return mmpTuplePlanesObservations[planes];
        } else {
            return static_cast<KeyFrame*>(nullptr);
        }
    }

    void Map::AddManhattanObservation(MapPlane *pMP1, MapPlane *pMP2, MapPlane *pMP3, KeyFrame* pKF) {
        unique_lock<mutex> lock(mMutexMap);

        Manhattan manhattan = std::make_tuple(pMP1, pMP2, pMP3);
        if (mmpManhattanObservations.count(manhattan) != 0)
            return;
//        cout << "Insert Manhattan3 pMP1: " << pMP1->mnId << endl;
//        cout << "Insert Manhattan3 pMP2: " << pMP2->mnId << endl;
//        cout << "Insert Manhattan3 pMP3: " << pMP3->mnId << endl;
        pKF->SetNotErase();
        mmpManhattanObservations[manhattan] = pKF;
    }

    KeyFrame* Map::GetManhattanObservation(MapPlane *pMP1, MapPlane *pMP2, MapPlane *pMP3) {
        unique_lock<mutex> lock(mMutexMap);
        Manhattan manhattan = std::make_tuple(pMP1, pMP2, pMP3);
        if (mmpManhattanObservations.count(manhattan)) {
            return mmpManhattanObservations[manhattan];
        } else {
            return static_cast<KeyFrame*>(nullptr);
        }
    }

    void Map::AddCrossLineToMap(MapPlane *pMP1, MapPlane *pMP2, cv::Mat CrossLine) {
        unique_lock<mutex> lock(mMutexMap);
        CrossLineSet = std::make_tuple(pMP1->mnId,pMP2->mnId,CrossLine);
        CrossLineDraw.emplace_back(CrossLine);
        CrossLineSets.emplace_back(CrossLineSet);
    }

    void Map::AddCrossPointToMap(MapPlane *pMP1, MapPlane *pMP2, MapPlane *pMP3, cv::Mat CrossPoint) {
        unique_lock<mutex> lock(mMutexMap);
        CrossPointSet = std::make_tuple(pMP1->mnId,pMP2->mnId,pMP3->mnId,CrossPoint);
        CrossPointDraw.emplace_back(CrossPoint);
        CrossPointSets.emplace_back(CrossPointSet);
    }

    void Map::AddNonPlaneArea(pcl::PointCloud<pcl::PointXYZRGB> &NonPlaneArea) {
        unique_lock<mutex> lock(mMutexMap);
        DrawNonPlaneArea = NonPlaneArea;
    }

    void Map::AddPairPlanesObservation(MapPlane *pMP1, MapPlane *pMP2, KeyFrame* pKF) {
        unique_lock<mutex> lock(mMutexMap);

        PairPlane plane = std::make_pair(pMP1, pMP2);
//        cout << "Insert Manhattan2 pMP1: " << pMP1->mnId << endl;
//        cout << "Insert Manhattan2 pMP2: " << pMP2->mnId << endl;
        if (mmpPairPlanesObservations.count(plane) != 0)
            return;
        pKF->SetNotErase();
        mmpPairPlanesObservations[plane] = pKF;
    }

    KeyFrame* Map::GetCrossLineObservation(MapPlane *pMP1, MapPlane *pMP2) {
        unique_lock<mutex> lock(mMutexMap);
        PairPlane plane = std::make_pair(pMP1, pMP2);
        if (mmpPairPlanesObservations.count(plane)) {
            return mmpPairPlanesObservations[plane];
        } else {
            return static_cast<KeyFrame*>(nullptr);
        }
    }

    Map::PairPlanes Map::GetAllPairPlaneObservation() {
        return mmpPairPlanesObservations;
    }

    Map::Manhattans Map::GetAllManhattanObservations() {
        return mmpManhattanObservations;
    }

    Map::TuplePlanes Map::GetAllTuplePlaneObservations() {
        return mmpTuplePlanesObservations;
    }

    void Map::AddPartialManhattanObservation(MapPlane *pMP1, MapPlane *pMP2, KeyFrame* pKF) {
        unique_lock<mutex> lock(mMutexMap);

        PartialManhattan manhattan = std::make_pair(pMP1, pMP2);
//        cout << "Insert Manhattan2 pMP1: " << pMP1->mnId << endl;
//        cout << "Insert Manhattan2 pMP2: " << pMP2->mnId << endl;
        if (mmpPartialManhattanObservations.count(manhattan) != 0)
            return;
        pKF->SetNotErase();
        mmpPartialManhattanObservations[manhattan] = pKF;
    }

    KeyFrame* Map::GetPartialManhattanObservation(MapPlane *pMP1, MapPlane *pMP2) {
        unique_lock<mutex> lock(mMutexMap);
        PartialManhattan manhattan = std::make_pair(pMP1, pMP2);
        if (mmpPartialManhattanObservations.count(manhattan)) {
            return mmpPartialManhattanObservations[manhattan];
        } else {
            return static_cast<KeyFrame*>(nullptr);
        }
    }

    Map::PartialManhattans Map::GetAllPartialManhattanObservations() {
        return mmpPartialManhattanObservations;
    }

} //namespace ORB_SLAM
