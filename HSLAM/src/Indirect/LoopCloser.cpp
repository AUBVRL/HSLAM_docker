#include "Indirect/LoopCloser.h"
#include "Indirect/MapPoint.h"
#include "Indirect/Frame.h"
#include "Indirect/Map.h"
#include "Indirect/Matcher.h"
#include "Indirect/Optimizer.h"
#include "Indirect/Sim3Solver.h"

#include "FullSystem/FullSystem.h"
#include <iostream>


namespace HSLAM {

    LoopCloser::LoopCloser(FullSystem *fullsystem) : fullSystem(fullsystem)
    {
        globalMap = fullSystem->globalMap;
        wpmatcher = fullSystem->matcher;
        mainLoop = boost::thread(&LoopCloser::Run, this);
        currMaxMp = 0;
        currMaxKF = 0;
        minActId = 0;
    }

    void LoopCloser::InsertKeyFrame(shared_ptr<Frame> &frame, int maxMpId)
    {
        std::vector<std::shared_ptr<MapPoint>> curActMP;
        std::vector<std::shared_ptr<Frame>> curActKF;

        boost::unique_lock<boost::mutex> lock(mutexKFQueue);
        frame->SetNotErase();
        copyActiveMapData(curActKF, curActMP);
        KFqueue.push_back(std::make_tuple(frame, curActKF, curActMP, frame->fs->KfId, maxMpId));
    }

    void LoopCloser::Run() {
        finished = false;
        //printf("LoopCloser RUN,  needFinish, is :%d \n", needFinish ); //debugNA
        while (1) {

            if (needFinish) {break; }

            {
                // get the oldest one
                boost::unique_lock<boost::mutex> lock(mutexKFQueue);
                if (KFqueue.empty()) {
                    lock.unlock();
                    usleep(5000);
                    continue;
                }
                
                if (KFqueue.size() > 5)
                { //can happen if optimization took too long!! in this case just add the accumulated kfs to the database and move on
                    
                    for (auto it : KFqueue)
                    {
                        auto frame = std::get<0>(it);
                        frame->ComputeBoVW();
                        globalMap.lock()->KfDB->add(frame);
                        frame->SetErase(); //allow mapper to erase it if deemed it not useful already!
                    }
                        
                    KFqueue.clear();
                    continue;
                }

                //copy a snapshot of active frames and mapPoints when the candidate was inserted!! use the max KFIds to keep out all new data from the loop closure process (newer MapPoints or Kfs should be helf fixed to preseve gauge!)
                std::tie(currentKF, ActiveFrames, ActivePoints, currMaxKF, currMaxMp) = KFqueue.front();
                
                currentKF->ComputeBoVW();
                KFqueue.pop_front(); 

            }

            bool loopDetected = DetectLoop();
            if (loopDetected)
            { 
                printf("Loop detected"); //debugNA
                if (computeSim3())
                {
                    static Timer loopCorrTime("loopCorr");
                    loopCorrTime.startTime();
                    auto gMap = globalMap.lock();
                    if (gMap->isIdle()) //prevent from doing a loop closure correction when another is taking place!
                    {
                        gMap->setBusy(true);
                        CorrectLoop();
                        gMap->setBusy(false);
                    }
                    loopCorrTime.endTime(true);
                }
            }

            usleep(5000);
        }

        finished = true;
        //printf("RUN or MainLoop is finished with finished : %i \n", finished);
    }

    void LoopCloser::copyActiveMapData(std::vector<std::shared_ptr<Frame>> & _KFs ,std::vector<std::shared_ptr<MapPoint>> & _MPs)
    {
        // boost::unique_lock<boost::mutex> lock(fullSystem->mapMutex);
        auto gMap = globalMap.lock();
        minActId = UINT_MAX;
        for (int i = 0, iend = fullSystem->frameHessians.size(); i < iend; ++i)
        {
            std::shared_ptr<Frame> actKF = fullSystem->frameHessians[i]->shell->frame;
            _KFs.push_back(actKF);

            std::vector<std::shared_ptr<MapPoint>> kfPts = actKF->getMapPointsV();
            for (int j = 0, jend = kfPts.size(); j < jend; ++j)
            {
                if (!kfPts[j])
                    continue;
                if(kfPts[j]->isBad() || kfPts[j]->getDirStatus() != MapPoint::active)
                    continue;
                _MPs.push_back(kfPts[j]);
            }

            if(actKF->fs->KfId < minActId)
                minActId = actKF->fs->KfId;
        }
    }


    bool LoopCloser::DetectLoop()
    {   

        auto gMap = globalMap.lock();
        //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
        if (currentKF->fs->KfId < mLastLoopKFid + kfGap)
        {
           // printf("Map contains less than 10 KF or less than 10 KF have passed from last loop detection\n");
            gMap->KfDB->add(currentKF);
            currentKF->SetErase();
            return false;
        }

        // Compute reference BoW similarity score
        // This is the lowest score to a connected keyframe in the covisibility graph
        // We will impose loop candidates to have a higher similarity than this
        const std::vector<std::shared_ptr<Frame>> vpConnectedKeyFrames = currentKF->GetVectorCovisibleKeyFrames();
        const DBoW3::BowVector &CurrentBowVec = currentKF->mBowVec;
        float minScore = 1;
        for (size_t i = 0; i < vpConnectedKeyFrames.size(); i++)
        {
            std::shared_ptr<Frame> pKF = vpConnectedKeyFrames[i];
            if (pKF->isBad())
                continue;
            const DBoW3::BowVector &BowVec = pKF->mBowVec;

            float score = Vocab.score(CurrentBowVec, BowVec);

            if (score < minScore)
                minScore = score;
        }

        // Query the database imposing the minimum score
        std::vector<std::shared_ptr<Frame>> vpCandidateKFs = gMap->KfDB->DetectLoopCandidates(currentKF, minScore);
        // If there are no loop candidates, just add new keyframe and return false
        if (vpCandidateKFs.empty())
        {   
            //printf("Detect Loop Candidates Empty, will return FALSE\n"); //debug NA

            gMap->KfDB->add(currentKF);
            mvConsistentGroups.clear();
            currentKF->SetErase();
            return false;
        }

        // For each loop candidate check consistency with previous loop candidates
        // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
        // A group is consistent with a previous group if they share at least a keyframe
        // We must detect a consistent loop in several consecutive keyframes to accept it
        mvpEnoughConsistentCandidates.clear();

        std::vector<ConsistentGroup> vCurrentConsistentGroups;
        std::vector<bool> vbConsistentGroup(mvConsistentGroups.size(), false);
        for (size_t i = 0, iend = vpCandidateKFs.size(); i < iend; i++)
        {
            std::shared_ptr<Frame> pCandidateKF = vpCandidateKFs[i];

            std::set<std::shared_ptr<Frame>, std::owner_less<std::shared_ptr<Frame>>> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
            spCandidateGroup.insert(pCandidateKF);

            bool bEnoughConsistent = false;
            bool bConsistentForSomeGroup = false;
            for (size_t iG = 0, iendG = mvConsistentGroups.size(); iG < iendG; iG++)
            {
                std::set<std::shared_ptr<Frame>,std::owner_less<std::shared_ptr<Frame>>> sPreviousGroup = mvConsistentGroups[iG].first;

                bool bConsistent = false;
                for (std::set<std::shared_ptr<Frame>,std::owner_less<std::shared_ptr<Frame>>>::iterator sit = spCandidateGroup.begin(), send = spCandidateGroup.end(); sit != send; sit++)
                {
                    if (sPreviousGroup.count(*sit))
                    {
                        bConsistent = true;
                        bConsistentForSomeGroup = true;
                        break;
                    }
                }

                if (bConsistent)
                {
                    int nPreviousConsistency = mvConsistentGroups[iG].second;
                    int nCurrentConsistency = nPreviousConsistency + 1;
                    if (!vbConsistentGroup[iG])
                    {
                        ConsistentGroup cg = std::make_pair(spCandidateGroup, nCurrentConsistency);
                        vCurrentConsistentGroups.push_back(cg);
                        vbConsistentGroup[iG] = true; //this avoid to include the same group more than once
                    }
                    if (nCurrentConsistency >= mnCovisibilityConsistencyTh && !bEnoughConsistent)
                    {
                        mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                        bEnoughConsistent = true; //this avoid to insert the same candidate more than once
                    }
                }
            }

            // If the group is not consistent with any previous group insert with consistency counter set to zero
            if (!bConsistentForSomeGroup)
            {
                ConsistentGroup cg = std::make_pair(spCandidateGroup, 0);
                vCurrentConsistentGroups.push_back(cg);
            }
        }

        // Update Covisibility Consistent Groups
        mvConsistentGroups = vCurrentConsistentGroups;

        // Add Current Keyframe to database
        gMap->KfDB->add(currentKF);

        if (mvpEnoughConsistentCandidates.empty())
        {
            currentKF->SetErase();
            return false;
        }
        else
        {
            return true;
        }

        currentKF->SetErase();
        return false;
    }

    bool LoopCloser::computeSim3()
    {
        const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

        auto matcher = wpmatcher.lock();

        std::vector<std::shared_ptr<Sim3Solver>> vpSim3Solvers;
        vpSim3Solvers.resize(nInitialCandidates);

        std::vector<std::vector<std::shared_ptr<MapPoint> >> vvpMapPointMatches;
        vvpMapPointMatches.resize(nInitialCandidates);

        std::vector<bool> vbDiscarded;
        vbDiscarded.resize(nInitialCandidates);

        int nCandidates = 0; //candidates with enough matches

        for (int i = 0; i < nInitialCandidates; i++)
        {
            std::shared_ptr<Frame> pKF = mvpEnoughConsistentCandidates[i];

            // avoid that local mapping erase it while it is being processed in this thread
            pKF->SetNotErase();

            if (pKF->isBad())
            {
                vbDiscarded[i] = true;
                continue;
            }

            std::vector<std::shared_ptr<MapPoint>> matches;
            int nmatches = matcher->SearchByBow(pKF, currentKF , 0.75, true, vvpMapPointMatches[i]); //0.75 def currentKF, pKF
            if (nmatches < 20)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                std::shared_ptr<Sim3Solver> pSolver = std::make_shared<Sim3Solver>(pKF, currentKF, vvpMapPointMatches[i], false); // def currentKF, pKF
                pSolver->SetRansacParameters(0.99, 20, 300);
                vpSim3Solvers[i] = pSolver;
            }

            nCandidates++;
        }

        bool bMatch = false;
        
        // Perform alternatively RANSAC iterations for each candidate
        // until one is succesful or all fail
        while (nCandidates > 0 && !bMatch)
        {
            for (int i = 0; i < nInitialCandidates; i++)
            {
                if (vbDiscarded[i])
                    continue;

                std::shared_ptr<Frame> pKF = mvpEnoughConsistentCandidates[i];

                // Perform 5 Ransac Iterations
                std::vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;

                std::shared_ptr<Sim3Solver> pSolver = vpSim3Solvers[i];
                cv::Mat Scm = pSolver->iterate(5, bNoMore, vbInliers, nInliers);
                
                // If Ransac reachs max. iterations discard keyframe
                if (bNoMore)
                {
                    vbDiscarded[i] = true;
                    nCandidates--;
                }

                // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
                if (!Scm.empty())
                {
                    std::vector<std::shared_ptr<MapPoint>> vpMapPointMatches(vvpMapPointMatches[i].size(), nullptr);
                    for (size_t j = 0, jend = vbInliers.size(); j < jend; j++)
                    {
                        if (vbInliers[j])
                            vpMapPointMatches[j] = vvpMapPointMatches[i][j];
                    }

                    Mat33f R = pSolver->GetEstimatedRotation();
                    Vec3f t = pSolver->GetEstimatedTranslation();
                    const float s = pSolver->GetEstimatedScale();
                    if(s < 0 )
                    {
                        vbDiscarded[i] = true;
                        nCandidates--;
                        continue;
                    }

                    int numatches = matcher->SearchBySim3(pKF, currentKF, vpMapPointMatches, s, R, t, 7.5); //def: currentKf, pKF
                    // cv::Mat Output;
                    // cv::hconcat(pKF->Image, currentKF->Image, Output);
                    // cv::cvtColor(Output, Output, CV_GRAY2RGB);
                    // static int count = 0;
                    // count = 0;
                    // for (int j = 0, jend = pKF->nFeatures; j < jend; ++j)
                    // {
                    //     if (vpMapPointMatches[j])
                    //     {
                    //         cv::Point2f Pt1 = pKF->mvKeys[j].pt;
                    //         cv::Point2f Pt2 = currentKF->mvKeys[vpMapPointMatches[j]->getIndexInKF(currentKF)].pt + cv::Point2f(640, 0);
                    //         cv::circle(Output, Pt1, 1, cv::Scalar(0, 255, 0), -1);
                    //         cv::circle(Output, Pt2, 1, cv::Scalar(0, 255, 0), -1);
                    //         cv::line(Output, Pt1, Pt2, cv::Scalar(255, 0, 0));
                    //         count++;
                    //     }
                    // }

                    // cv::namedWindow("matches", cv::WINDOW_KEEPRATIO);
                    // cv::imshow("matches", Output);

                    // cv::waitKey(1);
                    // std::cout << "candid: " << i << " inliers: " << numatches << " count " << count << std::endl;

                    Sim3 gScm = Sim3(SE3(R.cast<double>(), t.cast<double>()).matrix());
                    gScm.setScale(s);

                    const int nInliers = OptimizeSim3(pKF, currentKF, vpMapPointMatches, gScm, 10, false); //def: currentKf, pKF
                    // If optimization is succesful stop ransacs and continue
                    if (nInliers >= 30) //20
                    {
                        bMatch = true;
                        candidateKF = pKF;
                        Sim3 gSmw = currentKF->fs->getPoseOpti(); //Sim3(currentKF->fs->getPoseInverse().matrix()); //pKF
                        mScw = gScm * gSmw;
                       
                        mvpCurrentMatchedPoints = vpMapPointMatches;
                        break;
                    }
                }
            }
        }

        if (!bMatch)
        {
            for (int i = 0; i < nInitialCandidates; i++)
                mvpEnoughConsistentCandidates[i]->SetErase();
            currentKF->SetErase();
            return false;
        }

        // Retrieve MapPoints seen in Loop Keyframe and neighbors
        std::vector<std::shared_ptr<Frame>> vpLoopConnectedKFs = currentKF->GetVectorCovisibleKeyFrames(); //candidateKF
        vpLoopConnectedKFs.push_back(currentKF); //candidateKF
        mvpLoopMapPoints.clear();
        for (std::vector<std::shared_ptr<Frame>>::iterator vit = vpLoopConnectedKFs.begin(); vit != vpLoopConnectedKFs.end(); vit++)
        {
            std::shared_ptr<Frame> pKF = *vit;
            std::vector<std::shared_ptr<MapPoint>> vpMapPoints = pKF->getMapPointsV();
            for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
            {
                std::shared_ptr<MapPoint> pMP = vpMapPoints[i];
                if (pMP)
                {
                    if (!pMP->isBad() && pMP->mnLoopPointForKF != currentKF->fs->KfId)
                    {
                        mvpLoopMapPoints.push_back(pMP);
                        pMP->mnLoopPointForKF = currentKF->fs->KfId;
                    }
                }
            }
        }

        // Find more matches projecting with the computed Sim3
        int curmatches = matcher->SearchBySim3Projection(candidateKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints, 10); // currentKF
        // If enough matches accept Loop
        int nTotalMatches = 0;
        for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); i++)
        {
            if (mvpCurrentMatchedPoints[i])
                nTotalMatches++;
        }

        if (nTotalMatches >= 20) //20
        {
            for (int i = 0; i < nInitialCandidates; i++)
                if (mvpEnoughConsistentCandidates[i] != candidateKF)
                    mvpEnoughConsistentCandidates[i]->SetErase();
            return true;
        }
        else
        {
            for (int i = 0; i < nInitialCandidates; i++)
                mvpEnoughConsistentCandidates[i]->SetErase();
            currentKF->SetErase();
            return false;
        }
    }

    void LoopCloser::SetFinish(bool finish)
    {
        needFinish = finish;
        std::shared_ptr<Map> gMap = globalMap.lock();
        std::cout << "wait for loop closing thread to finish" << std::endl;
        
        while(!globalMap.lock()->isIdle())
        {
            usleep(10000);
        }
        mainLoop.join();
        usleep(5000);

        KFqueue.clear();
        mLastLoopKFid = 0;
     //  printf("LoopCloser set to finish, needFinish is %i \n", needFinish);      
    }

    void LoopCloser::CorrectLoop()
    {
        printf("Loop detected, to be corrected!");//debugNA

        // boost::unique_lock<boost::mutex> lck(fullSystem->mapMutex);  //
        auto gMap = globalMap.lock();
        std::vector<std::shared_ptr<Frame>> allKFrames;
        std::vector<std::shared_ptr<MapPoint>> allMapPoints;

        {
            // boost::unique_lock<boost::mutex> lck(fullSystem->mapMutex);
            globalMap.lock()->GetAllKeyFrames(allKFrames);
            globalMap.lock()->GetAllMapPoints(allMapPoints);
        }

        // Ensure current keyframe is updated
        currentKF->UpdateConnections();
        candidateKF->UpdateConnections();

        // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
        mvpCurrentConnectedKFs = candidateKF->GetVectorCovisibleKeyFrames(); //currentKF
        mvpCurrentConnectedKFs.push_back(candidateKF);                       //currentKF

        auto LatestConnected = currentKF->GetConnectedKeyFrames(); //currentKF

        KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
        CorrectedSim3[candidateKF] = mScw; //currentKF, mg2oScw

        auto TempTwc = candidateKF->fs->getPoseOptiInv();

        std::set<std::shared_ptr<Frame>> TempFixed; //these frames are old but found few matches to the latest KF- fix them here but don't fix them in poseGraph!

        for (std::vector<std::shared_ptr<Frame>>::iterator vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end(); vit != vend; vit++)
        {
            std::shared_ptr<Frame> pKFi = *vit;

            if (pKFi->fs->KfId > currMaxKF) //make sure keyframes added after loop detection are not modified!!
                continue;

            if (pKFi->fs->KfId >= minActId) //if the connected keyframe to the loop candidate was recently added don't update its pose!! this is like a bubble that protects the recently added keyframes from being changed!
                continue;

            Sim3 TiwTemp = pKFi->fs->getPoseOpti();
            
            if (pKFi != candidateKF) //currentKF - below not changed..
            {
                Sim3 g2oSic = TiwTemp * TempTwc; //SE3 Tic = Tiw * Twc

                Sim3 g2oCorrectedSiw = g2oSic * mScw;
                //Pose corrected with the Sim3 of the loop closure
                // if (!LatestConnected.count(pKFi))
                // if(LatestConnected.count(pKFi) == 0 )
                    CorrectedSim3[pKFi] = g2oCorrectedSiw;
                // else
                // {
                    // CorrectedSim3[pKFi] = TiwTemp;
                    // TempFixed.insert(pKFi);
                // }
            }
            //Pose without correction
            NonCorrectedSim3[pKFi] = TiwTemp;
        }

        // Correct all MapPoints obsrved by candidate keyframe and neighbors, so that they align with the other side of the loop
        for (KeyFrameAndPose::iterator mit = CorrectedSim3.begin(), mend = CorrectedSim3.end(); mit != mend; mit++)
        {
            std::shared_ptr<Frame> pKFi = mit->first;
            Sim3 g2oCorrectedSiw = mit->second;
            pKFi->fs->setPoseOpti(g2oCorrectedSiw);
            Sim3 g2oSiw = NonCorrectedSim3[pKFi];

            std::vector<std::shared_ptr<MapPoint>> vpMPsi = pKFi->getMapPointsV();
            for (size_t iMP = 0, endMPi = vpMPsi.size(); iMP < endMPi; iMP++)
            {
                std::shared_ptr<MapPoint> pMPi = vpMPsi[iMP];
                if (!pMPi)
                    continue;

                if (pMPi->id > currMaxMp || pMPi->sourceFrame->fs->KfId > currMaxKF) //prevent new data from being modified
                    continue;

                if (pMPi->isBad())
                    continue;
                if (pMPi->mnCorrectedByKF == pKFi->fs->KfId) 
                    continue;

                if (pMPi->getDirStatus() == MapPoint::active) //prevent active data from being modified!
                    continue;
                if (pMPi->sourceFrame->getState() == Frame::active)
                    continue;

                pMPi->updateGlobalPose();
                pMPi->mnCorrectedByKF = pKFi->fs->KfId;
                pMPi->mnCorrectedReference = currentKF->fs->KfId;
                pMPi->UpdateNormalAndDepth();
            }

            // Make sure connections are updated
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); i++)
        {
            if (mvpCurrentMatchedPoints[i])
            {
                std::shared_ptr<MapPoint> pLoopMP = candidateKF->getMapPoint(i); //mvpCurrentMatchedPoints[i];
                std::shared_ptr<MapPoint> pCurMP = mvpCurrentMatchedPoints[i];   //currentKF->getMapPoint(i);
                if (pLoopMP)                                                     //pCurMP  //Replace old mapPoints with new ones! should experiment with the other way around!
                    pLoopMP->Replace(pCurMP);                                    //pCurMP->Replace(pLoopMP);
                else                                                             //if old kf does not contain match, add it!
                {
                    candidateKF->addMapPointMatch(pCurMP, i);     //mpCurrentKF, pLoopMP
                    pCurMP->AddObservation(candidateKF, i);       // pLoopMP,  mpCurrentKF
                    pCurMP->ComputeDistinctiveDescriptors(false); //pLoopMP
                }
            }
        }

        // Project MapPoints observed in the neighborhood of the loop keyframe
        // into the current keyframe and neighbors using corrected poses.
        // Fuse duplications.
        SearchAndFuse(CorrectedSim3);

        // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
        std::map<std::shared_ptr<Frame>, std::set<std::shared_ptr<Frame>, std::owner_less<std::shared_ptr<Frame>>>, std::owner_less<std::shared_ptr<Frame>>>  LoopConnections;

        for (std::vector<std::shared_ptr<Frame>>::iterator vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end(); vit != vend; vit++)
        {
            std::shared_ptr<Frame> pKFi = *vit;
            if(pKFi->fs->KfId > currMaxKF)
                continue;
            std::vector<std::shared_ptr<Frame>> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

            // Update connections. Detect new links.
            pKFi->UpdateConnections();
            LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();
            for (std::vector<std::shared_ptr<Frame>>::iterator vit_prev = vpPreviousNeighbors.begin(), vend_prev = vpPreviousNeighbors.end(); vit_prev != vend_prev; vit_prev++)
            {
                LoopConnections[pKFi].erase(*vit_prev);
            }
            for (std::vector<std::shared_ptr<Frame>>::iterator vit2 = mvpCurrentConnectedKFs.begin(), vend2 = mvpCurrentConnectedKFs.end(); vit2 != vend2; vit2++)
            {
                LoopConnections[pKFi].erase(*vit2);
            }
        }

        // Optimize graph 
        //this allows future data captured since the candidate detection to also fix the gauge freedom and prevent unwanted errors!
        OptimizeEssentialGraph(fullSystem->allKeyFramesHistory, allMapPoints, TempFixed, currentKF ,candidateKF , NonCorrectedSim3, CorrectedSim3, LoopConnections, 
                                fullSystem->ef->connectivityMap, currMaxKF, minActId, currMaxMp, false); //mpMatchedKF, mpCurrentKF,
        
        for (auto it : fullSystem->allKeyFramesHistory)
            it->setRefresh(true);
        gMap->InformNewBigChange();

        // Add loop edge
        candidateKF->AddLoopEdge(currentKF);
        currentKF->AddLoopEdge(candidateKF);

        // Launch a new thread to perform Global Bundle Adjustment
        // mbRunningGBA = true;
        // mbFinishedGBA = false;
        mbStopGBA = false;
        // mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment, this, mpCurrentKF->mnId);
        
        // boost::unique_lock<boost::mutex> lock(fullSystem->trackMutex);
        
        // BundleAdjustment(allKFrames, allMapPoints, ActiveFrames, ActivePoints, 10, &mbStopGBA, true, true, maxKfId, currMaxKF, currMaxMp);
        // BundleAdjustment(allKFrames, allMapPoints, 10, &mbStopGBA, true, true, currMaxKF, minActId, currMaxMp);
        // // Loop closed. Release Local Mapping.
        // mpLocalMapper->Release();

        for (auto it : fullSystem->allKeyFramesHistory)
            it->setRefresh(true);

        mLastLoopKFid = currentKF->fs->KfId;
        cout << "Loop correction complete!" << endl;
    }

    void LoopCloser::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
    {
        // ORBmatcher matcher(0.8);
        auto matcher = wpmatcher.lock();
        for (KeyFrameAndPose::const_iterator mit = CorrectedPosesMap.begin(), mend = CorrectedPosesMap.end(); mit != mend; mit++)
        {
            std::shared_ptr<Frame> pKF = mit->first;

            Sim3 Scw = mit->second;
            // cv::Mat cvScw = Converter::toCvMat(g2oScw);

            std::vector<std::shared_ptr<MapPoint>> vpReplacePoints(mvpLoopMapPoints.size(), nullptr);
            matcher->Fuse(pKF, Scw, mvpLoopMapPoints, 4, vpReplacePoints);

            // Get Map Mutex
            // unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
            const int nLP = mvpLoopMapPoints.size();
            for (int i = 0; i < nLP; i++)
            {
                std::shared_ptr<MapPoint> pRep = vpReplacePoints[i];
                if (pRep)
                {
                    pRep->Replace(mvpLoopMapPoints[i]);
                }
            }
        }
    }

}
