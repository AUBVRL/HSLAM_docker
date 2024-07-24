#include "Indirect/Map.h"

#include "Indirect/MapPoint.h"
#include "Indirect/Frame.h"
#include "util/globalFuncs.h"
#include "util/FrameShell.h"

namespace HSLAM
{
    using namespace std;

    Map::Map()
    {
        mnMaxKFid = 0;
        mnMaxMPid = 0;
        mnBigChangeIdx = 0;
        KfDB = std::make_shared<KeyFrameDatabase>();
    }

    Map::~Map()
    {
        clear();
    }

    void Map::AddKeyFrame(shared_ptr<Frame> pKF)
    {
        boost::lock_guard<boost::mutex> l(mMutexMap);
        mspKeyFrames.insert(pKF);
        if (pKF->fs->KfId > mnMaxKFid)
            mnMaxKFid = pKF->fs->KfId;
    }

    void Map::AddMapPoint(shared_ptr<MapPoint> pMP)
    {
        boost::lock_guard<boost::mutex> l(mMutexMap);
        mspMapPoints.insert(pMP);
        if ( pMP->id > mnMaxMPid)
            mnMaxMPid = pMP->id;
    }

    void Map::EraseMapPoint(shared_ptr<MapPoint> pMP)
    {
        boost::lock_guard<boost::mutex> l(mMutexMap);
        mspMapPoints.erase(pMP);
    }

    void Map::EraseKeyFrame(shared_ptr<Frame> pKF)
    {
        boost::lock_guard<boost::mutex> l(mMutexMap);
        mspKeyFrames.erase(pKF);
    }

    void Map::SetReferenceMapPoints(const vector<shared_ptr<MapPoint>> &vpMPs)
    {
        boost::lock_guard<boost::mutex> l(mMutexMap);
        mvpReferenceMapPoints = vpMPs;
    }

    void Map::InformNewBigChange()
    {
        boost::lock_guard<boost::mutex> l(mMutexMap);
        mnBigChangeIdx++;
    }

    int Map::GetLastBigChangeIdx()
    {
        boost::lock_guard<boost::mutex> l(mMutexMap);
        return mnBigChangeIdx;
    }

    void Map::GetAllKeyFrames(vector<shared_ptr<Frame>>& _Out)
    {
        releaseVec(_Out);
        boost::lock_guard<boost::mutex> l(mMutexMap);
        _Out.reserve(mspKeyFrames.size());
        _Out.assign(mspKeyFrames.begin(), mspKeyFrames.end());
    }

    void Map::GetAllMapPoints(vector<shared_ptr<MapPoint>>& _Out)
    {
        releaseVec(_Out);
        boost::lock_guard<boost::mutex> l(mMutexMap);
        _Out.reserve(mspMapPoints.size());
        _Out.assign(mspMapPoints.begin(), mspMapPoints.end());
    }

    long unsigned int Map::MapPointsInMap()
    {
        boost::lock_guard<boost::mutex> l(mMutexMap);
        return mspMapPoints.size();
    }

    long unsigned int Map::KeyFramesInMap()
    {
        boost::lock_guard<boost::mutex> l(mMutexMap);
        return mspKeyFrames.size();
    }

    vector<shared_ptr<MapPoint>> Map::GetReferenceMapPoints()
    {
        boost::lock_guard<boost::mutex> l(mMutexMap);
        return mvpReferenceMapPoints;
    }

    size_t Map::GetMaxMPid()
    {
        boost::lock_guard<boost::mutex> l(mMutexMap);
        return mnMaxMPid;
    }


    void Map::clear()
    {
        mspMapPoints.clear();

        mspMapPoints.clear();
        mspKeyFrames.clear();
        mnMaxKFid = 0;
        mvpReferenceMapPoints.clear();
        mvpKeyFrameOrigins.clear();
        KfDB->clear();
    }


    //KEYFRAMEDATABSE
    KeyFrameDatabase::KeyFrameDatabase()
    {
        mvInvertedFile.resize(Vocab.size());
    }

    void KeyFrameDatabase::add(shared_ptr<Frame> pKF)
    {
        boost::lock_guard<boost::mutex> l(mMutex);
  
        for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end(); vit != vend; vit++)
            mvInvertedFile[vit->first].push_back(pKF);
    }

    void KeyFrameDatabase::erase(shared_ptr<Frame> pKF)
    {
        boost::lock_guard<boost::mutex> l(mMutex);

        // Erase elements in the Inverse File for the entry
        for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end(); vit != vend; vit++)
        {
            // List of keyframes that share the word
            list<shared_ptr<Frame>> &lKFs = mvInvertedFile[vit->first];

            for (list<shared_ptr<Frame>>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
            {
                if (pKF == *lit)
                {
                    lKFs.erase(lit);
                    break;
                }
            }
        }
    }

    void KeyFrameDatabase::clear()
    {
        mvInvertedFile.clear();
        mvInvertedFile.resize(Vocab.size());
    }

    vector<shared_ptr<Frame>> KeyFrameDatabase::DetectLoopCandidates(shared_ptr<Frame> pKF, float minScore)
    {
        set<shared_ptr<Frame>, std::owner_less<std::shared_ptr<Frame>>> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
        // auto vConnectedKeyFrames = pKF->GetCovisiblesByWeight(30);
        // std::set<shared_ptr<Frame>, std::owner_less<std::shared_ptr<Frame>>> spConnectedKeyFrames(vConnectedKeyFrames.begin(), vConnectedKeyFrames.end()); 

        list<shared_ptr<Frame>> lKFsSharingWords;

        // Search all keyframes that share a word with current keyframes
        // Discard keyframes connected to the query keyframe
        {
            boost::lock_guard<boost::mutex> l(mMutex);

            for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end(); vit != vend; vit++)
            {
                list<shared_ptr<Frame>> &lKFs = mvInvertedFile[vit->first];

                for (list<shared_ptr<Frame>>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
                {
                    shared_ptr<Frame> pKFi = *lit;
                    if (pKFi->mnLoopQuery != pKF->fs->KfId)
                    {
                        pKFi->mnLoopWords = 0;
                        if (!spConnectedKeyFrames.count(pKFi)) // if (pKF->fs->KfId > (pKFi->fs->KfId + minKfIdDist_LoopCandidate))
                        // if (pKF->fs->KfId > (pKFi->fs->KfId + minKfIdDist_LoopCandidate))
                        {
                            pKFi->mnLoopQuery = pKF->fs->KfId;
                            lKFsSharingWords.push_back(pKFi);
                        }
                    }
                    pKFi->mnLoopWords++;
                }
            }
        }

        if (lKFsSharingWords.empty())
            return vector<shared_ptr<Frame>>();

        list<pair<float, shared_ptr<Frame>>> lScoreAndMatch;

        // Only compare against those keyframes that share enough words
        int maxCommonWords = 0;
        for (list<shared_ptr<Frame>>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
        {
            if ((*lit)->mnLoopWords > maxCommonWords)
                maxCommonWords = (*lit)->mnLoopWords;
        }

        int minCommonWords = maxCommonWords * 0.8f;

        int nscores = 0;

        // Compute similarity score. Retain the matches whose score is higher than minScore
        for (list<shared_ptr<Frame>>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
        {
            shared_ptr<Frame> pKFi = *lit;

            if (pKFi->mnLoopWords > minCommonWords)
            {
                nscores++;

                float si = Vocab.score(pKF->mBowVec, pKFi->mBowVec); //mpVoc->score

                pKFi->mLoopScore = si;
                if (si >= minScore)
                    lScoreAndMatch.push_back(make_pair(si, pKFi));
            }
        }

        if (lScoreAndMatch.empty())
            return vector<shared_ptr<Frame>>();

        list<pair<float, shared_ptr<Frame>>> lAccScoreAndMatch;
        float bestAccScore = minScore;

        // Lets now accumulate score by covisibility
        for (list<pair<float, shared_ptr<Frame>>>::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend; it++)
        {
            shared_ptr<Frame> pKFi = it->second;
            vector<shared_ptr<Frame>> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

            float bestScore = it->first;
            float accScore = it->first;
            shared_ptr<Frame> pBestKF = pKFi;
            for (vector<shared_ptr<Frame>>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++)
            {
                shared_ptr<Frame> pKF2 = *vit;
                if (pKF2->mnLoopQuery == pKF->fs->KfId && pKF2->mnLoopWords > minCommonWords)
                {
                    accScore += pKF2->mLoopScore;
                    if (pKF2->mLoopScore > bestScore)
                    {
                        pBestKF = pKF2;
                        bestScore = pKF2->mLoopScore;
                    }
                }
            }

            lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
            if (accScore > bestAccScore)
                bestAccScore = accScore;
        }

        // Return all those keyframes with a score higher than 0.75*bestScore
        float minScoreToRetain = 0.75f * bestAccScore;

        set<shared_ptr<Frame>> spAlreadyAddedKF;
        vector<shared_ptr<Frame>> vpLoopCandidates;
        vpLoopCandidates.reserve(lAccScoreAndMatch.size());

        for (list<pair<float, shared_ptr<Frame>>>::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end(); it != itend; it++)
        {
            if (it->first > minScoreToRetain)
            {
                shared_ptr<Frame> pKFi = it->second;
                if (!spAlreadyAddedKF.count(pKFi))
                {
                    vpLoopCandidates.push_back(pKFi);
                    spAlreadyAddedKF.insert(pKFi);
                }
            }
        }

        return vpLoopCandidates;
    }

    
} // namespace HSLAM