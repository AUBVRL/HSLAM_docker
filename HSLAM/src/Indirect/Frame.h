#pragma once
#include "util/settings.h"
#include "util/NumType.h"
#include <boost/thread.hpp>

namespace HSLAM
{
    class CalibHessian;
    class FeatureDetector;
    class FrameShell;
    class MapPoint;
    struct FrameHessian;
    class Map;

    template <typename Type> class IndexThreadReduce;

    class Frame : public std::enable_shared_from_this<Frame>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        

        Frame(float* Img, std::shared_ptr<FeatureDetector> detector, CalibHessian *_HCalib, FrameHessian *_fh, FrameShell *_fs, std::shared_ptr<Map> _gMap);
        ~Frame();
      
        void ReduceToEssential();
        bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

        std::vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r) const;
        void ComputeBoVW();
        void assignFeaturesToGrid();
        
        inline std::shared_ptr<MapPoint> getMapPoint(int idx)
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            return mvpMapPoints[idx];
        }

        inline std::vector<std::shared_ptr<MapPoint>> getMapPointsV() //equivalent to getmappointmatches
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            return mvpMapPoints;
        }

        std::set<std::shared_ptr<MapPoint>> getMapPointsS(); 
        
        inline void EraseMapPointMatch(const size_t &idx)
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            mvpMapPoints[idx].reset();
        }

        void EraseMapPointMatch(std::shared_ptr<MapPoint> pMP);

        inline bool isBad()
        {
            boost::lock_guard<boost::mutex> l(mMutexConnections);
            return mbBad;
        }
        
        void setBadFlag();

        // Enable/Disable bad flag changes
        inline void SetNotErase()
        {
            boost::unique_lock<boost::mutex> l(mMutexConnections);
            mbNotErase = true;
        }

        inline void SetErase()
        {

            boost::unique_lock<boost::mutex> l(mMutexConnections);
            if (mspLoopEdges.empty())
                mbNotErase = false;

            l.unlock();
            if (mbToBeErased)
                setBadFlag();
        }

        inline void ReplaceMapPointMatch(const size_t &idx, std::shared_ptr<MapPoint> pMP)
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            mvpMapPoints[idx] = pMP;
        }

        static bool weightComp(int a, int b)
        {
            return a > b;
        }

        void addMapPoint(std::shared_ptr<MapPoint>& Mp);
        void addMapPointMatch(std::shared_ptr<MapPoint> Mp, size_t index);
        bool isInFrustum(std::shared_ptr<MapPoint> pMP, float viewingCosLimit);
        
        std::shared_ptr<Frame> getPtr();
        
        void AddConnection(std::shared_ptr<Frame> pKF, const int &weight);
        void EraseConnection(std::shared_ptr<Frame> pKF);

        void UpdateBestCovisibles();
        std::set<std::shared_ptr<Frame>, std::owner_less<std::shared_ptr<Frame>>> GetConnectedKeyFrames();
        std::vector<std::shared_ptr<Frame>> GetVectorCovisibleKeyFrames();
        std::vector<std::shared_ptr<Frame>> GetBestCovisibilityKeyFrames(const int &N);
        std::vector<std::shared_ptr<Frame>> GetCovisiblesByWeight(const int &w);
        int GetWeight(std::shared_ptr<Frame> pKF);
        void UpdateConnections();


        // Spanning tree functions
        void AddChild(std::shared_ptr<Frame> pKF);
        void EraseChild(std::shared_ptr<Frame> pKF);
        void ChangeParent(std::shared_ptr<Frame> pKF);
        std::set<std::shared_ptr<Frame>> GetChilds();
        std::shared_ptr<Frame> GetParent();
        bool hasChild(std::shared_ptr<Frame> pKF);

        // Loop Edges
        void AddLoopEdge(std::shared_ptr<Frame> pKF);
        std::set<std::shared_ptr<Frame>> GetLoopEdges();

        cv::Mat Image;
        cv::Mat Occupancy;
        std::vector<std::vector<unsigned short int>> mGrid;
        std::vector<cv::KeyPoint> mvKeys;

        std::vector<std::shared_ptr<MapPoint>> mvpMapPoints; //used to store all matches + generated mapPoints
        
        
        std::vector<std::shared_ptr<MapPoint>> tMapPoints; //used to store current frame matches (can be reset after a keyframe is created)
        std::vector<bool> mvbOutlier; //can also be reset after keyframe is created
        long unsigned int mnTrackReferenceForFrame;
        std::weak_ptr<Frame> mpReferenceKF;

        cv::Mat Descriptors;
        //BoW
        DBoW3::BowVector mBowVec;
        DBoW3::FeatureVector mFeatVec;

        int nFeatures;
        bool isReduced;
        bool NeedConnRefresh;


        CalibHessian *HCalib;
        FrameHessian *fh;
        FrameShell *fs;

        long unsigned int mnLoopQuery;
        int mnLoopWords;
        float mLoopScore;

        size_t mnFuseTargetForKF;

        enum kfstate {active=0, marginalized};
        kfstate kfState;
        kfstate getState();
        void setState(kfstate state);

    private:
        boost::mutex _mtx;
        boost::mutex mMutexConnections;
        boost::mutex BoVWmutex; //computeBoVW may be called from various threads! lock it

        std::map<std::shared_ptr<Frame>, int> mConnectedKeyFrameWeights;
        std::vector<std::shared_ptr<Frame>> mvpOrderedConnectedKeyFrames;
        std::vector<int> mvOrderedWeights;

        std::shared_ptr<Frame> mpParent;
        std::set<std::shared_ptr<Frame>> mspChildrens;
        std::set<std::shared_ptr<Frame>> mspLoopEdges;
        bool mbFirstConnection;
        bool mbNotErase;
        bool mbBad;
        bool mbToBeErased;
        std::shared_ptr<Map> globalMap;

        
    };

} // namespace HSLAM
