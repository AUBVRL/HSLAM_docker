#include "MapPoint.h"
#include "Indirect/Frame.h"
#include "FullSystem/HessianBlocks.h"
#include "Indirect/Matcher.h"
#include "Indirect/Map.h"
#include "util/FrameShell.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

namespace HSLAM
{
    using namespace std;
    size_t MapPoint::idCounter = 0;

    MapPoint::MapPoint(PointHessian *_ph, std::shared_ptr<Map> _globalMap)
    {
        assert(_ph->my_type > 4);
        globalMap = _globalMap;
        ph = _ph;
        sourceFrame = ph->host->shell->frame;
        index = ph->my_type - 5;
        nObs = 0;
        mnVisible = 1;
        mnFound = 1;
        mbBad = false;
        mnFuseCandidateForKF = 0;

        mnLoopPointForKF = 0;
        mnCorrectedByKF = 0;
        mnCorrectedReference = 0;

        pt = Vec2i(ph->u, ph->v);
        
        auto calib = sourceFrame->HCalib;

        sourceFrame->Descriptors.row(index).copyTo(mDescriptor);
        mnLastFrameSeen = 0;
        mnTrackReferenceForFrame = -1;
        status = mpDirStatus::active;
        idepth = ph->idepth;
        idepthH = ph->idepth_hessian;
        
        worldPose = sourceFrame->fs->getPose().cast<float>() * (Vec3f((pt[0] * calib->fxli() + calib->cxli()), (pt[1] * calib->fyli() + calib->cyli()), 1.0f) * (1.0f/idepth));

        // normal vector pointing towards the first frame
        Vec3f Owi = sourceFrame->fs->getCameraCenter().cast<float>();
        Vec3f normali = worldPose - Owi;
        mNormalVector = normali / normali.norm();
        // OIdepth=pointhessian->idepth_zero_scaled;
        // OWeight = sqrt(1e-3/(pointhessian->efPoint->HdiF+1e-12));

        boost::lock_guard<boost::mutex> l(_mtx);  //just in case some other thread wants to create a mapPoint
        id = idCounter;
        idCounter++;
        
    }

    
    shared_ptr<MapPoint> MapPoint::getPtr()
    {
         return shared_from_this();
    }


    void MapPoint::updateGlobalPose()
    {
        boost::lock_guard<boost::mutex> l(_mtx);
        // idepth = ph->idepth;
        // idepthH = ph->idepth_hessian;
        auto calib = sourceFrame->HCalib;
        worldPose = sourceFrame->fs->getPoseOptiInv().cast<float>() * (Vec3f((pt[0] * calib->fxli() + calib->cxli()), (pt[1] * calib->fyli() + calib->cyli()), 1.0f) * (1.0f/idepth));
        // worldPose = sourceFrame->fs->getPose().cast<float>() * (Vec3f( ((pt[0] - calib->cxl() )/ calib->fxl() ), ((pt[1]-calib->cyl()) / calib->fyl()), 1.0f) * (sourceFrame->fs->getPoseOptiInv().scale()/idepth) ); //getPoseOptiInv

    }

    void MapPoint::updateDepth()
    {
        assert(ph != 0);
        boost::lock_guard<boost::mutex> l(_mtx);
        idepth = ph->idepth;
        if(ph->efPoint)
            idepthH = ph->idepth_hessian; //efPoint->HdiF; //ph->idepth_hessian
    }

    void MapPoint::updateDepthfromInd(float _idepth)
    {
        boost::lock_guard<boost::mutex> l(_mtx);
        idepth = _idepth;
    }

    Vec3f MapPoint::getWorldPose()
    {
        boost::lock_guard<boost::mutex> l(_mtx);
        return worldPose;
    }

    // Vec3f MapPoint::getWorldPosewPose(SE3 &pose)
    // {
    //     boost::lock_guard<boost::mutex> l(_mtx);

    //     if(idepth < 0)
    //         return Vec3f(0.0f, 0.0f, 0.0f);

    //     // float depth = 1.0f / idepth;
    //     auto calib = sourceFrame->HCalib;
    //     auto pt = sourceFrame->mvKeys[index].pt;

    //     // float x = (pt.x * calib->fxli() + calib->cxli()) * depth;
    //     // float y = (pt.y * calib->fyli() + calib->cyli()) * depth;
    //     // float z = depth;
    //     return pose.cast<float>() * (Vec3f((pt.x * calib->fxli() + calib->cxli()), (pt.y * calib->fyli() + calib->cyli()), 1.0f) * (1.0f/idepth));
    //     // SE3 Pose = sourceFrame->fs->getPose();
    // }

    // Vec3f MapPoint::getWorldPose()
    // {
    //      boost::lock_guard<boost::mutex> l(_mtx);
         
    //     if (idepth < 0)
    //         return Vec3f(0.0f, 0.0f, 0.0f);

    //     auto pose = sourceFrame->fs->getPose();
    //     float depth = 1.0f / idepth;
    //     auto calib = sourceFrame->HCalib;
    //     auto pt = sourceFrame->mvKeys[index].pt;

    //     float x = (pt.x * calib->fxli() + calib->cxli()) * depth;
    //     float y = (pt.y * calib->fyli() + calib->cyli()) * depth;
    //     float z = depth;
    //     return pose.cast<float>() * Vec3f(x, y, z);
    // }

    void MapPoint::ComputeDistinctiveDescriptors(bool isQuick)
    {
        if(isQuick && getNObservations() > 7)
            return;
            
        // Retrieve all observed descriptors
        vector<cv::Mat> vDescriptors;
        
        map<shared_ptr<Frame>, size_t> observations = GetObservations();

        if (observations.empty())
            return;

        vDescriptors.reserve(observations.size());

        for (map<shared_ptr<Frame>, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            shared_ptr<Frame> pKF = mit->first;
            if (!pKF->isBad())
                vDescriptors.push_back(mit->first->Descriptors.row(mit->second));
        }

        if (vDescriptors.empty())
            return;

        // Compute distances between them
        const size_t N = vDescriptors.size();

        float Distances[N][N];
        for (size_t i = 0; i < N; i++)
        {
            Distances[i][i] = 0;
            for (size_t j = i + 1; j < N; j++)
            {
                int distij = Matcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
                Distances[i][j] = distij;
                Distances[j][i] = distij;
            }
        }

        // Take the descriptor with least median distance to the rest
        int BestMedian = INT_MAX;
        int BestIdx = 0;
        for (size_t i = 0; i < N; i++)
        {
            vector<int> vDists(Distances[i], Distances[i] + N);
            sort(vDists.begin(), vDists.end());
            int median = vDists[0.5 * (N - 1)];

            if (median < BestMedian)
            {
                BestMedian = median;
                BestIdx = i;
            }
        }

        setDescriptor(vDescriptors[BestIdx]);

    }

    void MapPoint::UpdateNormalAndDepth()
    {
        map<shared_ptr<Frame>, size_t> observations;
        Vec3f Pos;
        {
            boost::lock_guard<boost::mutex> l(_mtx); // pose lock and diff lock?

            if (mbBad)
                return;
            observations = mObservations;
            // pRefKF = mpRefKF;
            Pos = worldPose;
        }

        if (observations.empty())
            return;

        Vec3f normal = Vec3f::Zero();
        int n = 0;
        for (map<shared_ptr<Frame>, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            shared_ptr<Frame> pKF = mit->first;
            Vec3f Owi = pKF->fs->getCameraCenter().cast<float>();
            Vec3f normali = worldPose - Owi;
            normal = normal + normali / normali.norm();
            n++;
        }

        // cv::Mat PC = Pos - pRefKF->GetCameraCenter();
        // const float dist = cv::norm(PC);
        // const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
        // const float levelScaleFactor = pRefKF->mvScaleFactors[level];
        // const int nLevels = pRefKF->mnScaleLevels;

        {
            boost::lock_guard<boost::mutex> l(_mtx); //pose lock
            // mfMaxDistance = dist*levelScaleFactor;
            // mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];
            mNormalVector = normal / n;
        }

        return;
    }

    void MapPoint::AddObservation(std::shared_ptr<Frame> &pKF, size_t idx)
    {
        boost::unique_lock<boost::mutex> l(_mtx); //diff lock?
        if (mObservations.count(pKF))
            return;
        mObservations[pKF] = idx;
        nObs = nObs + 1;
    }

    void MapPoint::EraseObservation(std::shared_ptr<Frame> &pKF)
    {
        bool bBad = false;
        {
            boost::lock_guard<boost::mutex> l(_mtx);
            if (sourceFrame != pKF) //cannot remove observation if its the source frame (it contains the depth data)
            {
                if (mObservations.count(pKF))
                {
                    int idx = mObservations[pKF];
                    nObs--;

                    mObservations.erase(pKF);

                    // If only 2 observations or less, discard point
                    if (nObs <= 2)
                        bBad = true;
                }
            }
        }

        if (bBad)
            SetBadFlag();
    }

    void MapPoint::SetBadFlag()
    {
        map<shared_ptr<Frame>, size_t> obs;
        {
            boost::lock_guard<boost::mutex> l(_mtx);            
            mbBad = true;
            obs = mObservations;
            mObservations.clear();
        }
        for (map<shared_ptr<Frame> , size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++) //will remove all point observations
        {
            shared_ptr<Frame> pKF = mit->first;
            pKF->EraseMapPointMatch(mit->second); 
        }

        globalMap.lock()->EraseMapPoint(getPtr());
    }

    void MapPoint::Replace(shared_ptr<MapPoint> pMP)
    {
        if (pMP->id == this->id)
            return;

        int nvisible, nfound;
        map<shared_ptr<Frame> , size_t> obs;
        {
            boost::lock_guard<boost::mutex> l(_mtx); //pose and diff lock?
            obs = mObservations;
            mObservations.clear();
            mbBad = true;
            nvisible = mnVisible;
            nfound = mnFound;
            mpReplaced = pMP;
        }

        for (map<shared_ptr<Frame>, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
        {
            // Replace measurement in keyframe
            shared_ptr<Frame> pKF = mit->first;

            if (!pMP->isInKeyframe(pKF))
            {
                pKF->ReplaceMapPointMatch(mit->second, pMP);
                pMP->AddObservation(pKF, mit->second);
            }
            else
            {
                pKF->EraseMapPointMatch(mit->second);
            }
        }
        pMP->increaseFound(nfound);
        pMP->increaseVisible(nvisible);
        pMP->ComputeDistinctiveDescriptors(false);

        globalMap.lock()->EraseMapPoint(getPtr());
    }

} // namespace HSLAM