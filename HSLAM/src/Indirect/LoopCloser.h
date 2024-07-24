#pragma once
#ifndef _LoopCloser__H_
#define _LoopCloser__H_

#include "util/NumType.h"
#include "util/settings.h"
#include <boost/thread.hpp>
#include <list>


using namespace std;


namespace HSLAM {
    class FullSystem;
    class Frame;
    class MapPoint;
    class Map;
    class CalibHessian;
    class Matcher;


    class LoopCloser {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        LoopCloser(FullSystem *fullSystem);

        ~LoopCloser() {
        }

        void InsertKeyFrame(std::shared_ptr<Frame> &frame, int maxMpId);
        bool DetectLoop();
        bool computeSim3();
        void CorrectLoop();
        void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);
        void Run();
        void SetFinish(bool finish = true);
        void copyActiveMapData(std::vector<std::shared_ptr<Frame>> & _KFs ,std::vector<std::shared_ptr<MapPoint>> & _MPs);

    private:
        // data
        FullSystem *fullSystem;
        std::weak_ptr<Map> globalMap;  // global map
        std::weak_ptr<Matcher> wpmatcher;
     
        std::shared_ptr<Frame> currentKF = nullptr;
        std::shared_ptr<Frame> candidateKF = nullptr;

        bool finished = false;
        bool needFinish = false;
        boost::thread mainLoop;


        // loop kf queue
        std::deque<std::tuple<std::shared_ptr<Frame>, std::vector<std::shared_ptr<Frame>>, std::vector<std::shared_ptr<MapPoint>> ,size_t, size_t> > KFqueue;  //frame, current active Frame, current active Mps, maxKfId, maxMpId
        boost::mutex mutexKFQueue;


        // parameters

        std::vector<ConsistentGroup> mvConsistentGroups;
        std::vector<std::shared_ptr<Frame>> mvpEnoughConsistentCandidates;
        std::vector<std::shared_ptr<Frame>> mvpCurrentConnectedKFs;


        long unsigned int mLastLoopKFid=0;

        Sim3 mScw;
        std::vector<std::shared_ptr<MapPoint>> mvpCurrentMatchedPoints;
        std::vector<std::shared_ptr<MapPoint>> mvpLoopMapPoints;

        bool mbStopGBA;

        size_t currMaxMp;
        size_t currMaxKF;
        size_t minActId;

        std::vector<std::shared_ptr<MapPoint>> ActivePoints;
        std::vector<std::shared_ptr<Frame>> ActiveFrames;


    };
}

#endif // LDSO_LOOP_CLOSING_H_
