#include "Matcher.h"
#include "Indirect/Frame.h"
#include "util/FrameShell.h"
#include "Indirect/MapPoint.h"
#include "DBoW3/DBoW3.h"
#include "FullSystem/HessianBlocks.h"

#include <cmath>
namespace HSLAM
{
    using namespace std;

    const int Matcher::TH_HIGH = 100;
    const int Matcher::TH_LOW = 50;
    const int Matcher::HISTO_LENGTH = 30;

    int Matcher::SearchByBoWTracking(shared_ptr<Frame> pKF, shared_ptr<Frame> F, float nnRatio, bool mbCheckOrientation, std::vector<std::shared_ptr<MapPoint>> &vpMapPointMatches)
    {
        pKF->ComputeBoVW();// won't cost a thing if already computed.
        F->ComputeBoVW();
        const vector<std::shared_ptr<MapPoint>> vpMapPointsKF = pKF->getMapPointsV();

        vpMapPointMatches = std::vector<std::shared_ptr<MapPoint>>(F->nFeatures, nullptr);

        const DBoW3::FeatureVector &vFeatVecKF = pKF->mFeatVec;

        int nmatches = 0;

        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f / HISTO_LENGTH;

        // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
        DBoW3::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
        DBoW3::FeatureVector::const_iterator Fit = F->mFeatVec.begin();
        DBoW3::FeatureVector::const_iterator KFend = vFeatVecKF.end();
        DBoW3::FeatureVector::const_iterator Fend = F->mFeatVec.end();

        while (KFit != KFend && Fit != Fend)
        {
            if (KFit->first == Fit->first)
            {
                const vector<unsigned int> vIndicesKF = KFit->second;
                const vector<unsigned int> vIndicesF = Fit->second;

                for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++)
                {
                    const unsigned int realIdxKF = vIndicesKF[iKF];

                    std::shared_ptr<MapPoint> pMP = vpMapPointsKF[realIdxKF];

                    if (!pMP)
                        continue;

                    if (pMP->isBad())
                        continue;

                    const cv::Mat &dKF = pKF->Descriptors.row(realIdxKF);

                    int bestDist1 = 256;
                    int bestIdxF = -1;
                    int bestDist2 = 256;

                    for (size_t iF = 0; iF < vIndicesF.size(); iF++)
                    {
                        const unsigned int realIdxF = vIndicesF[iF];

                        if (vpMapPointMatches[realIdxF])
                            continue;

                        const cv::Mat &dF = F->Descriptors.row(realIdxF);

                        const int dist = DescriptorDistance(dKF, dF);

                        if (dist < bestDist1)
                        {
                            bestDist2 = bestDist1;
                            bestDist1 = dist;
                            bestIdxF = realIdxF;
                        }
                        else if (dist < bestDist2)
                        {
                            bestDist2 = dist;
                        }
                    }

                    if (bestDist1 <= TH_LOW)
                    {
                        if (static_cast<float>(bestDist1) < nnRatio * static_cast<float>(bestDist2))
                        {
                            vpMapPointMatches[bestIdxF] = pMP;

                            const cv::KeyPoint &kp = pKF->mvKeys[realIdxKF];

                            if (mbCheckOrientation)
                            {
                                float rot = kp.angle - F->mvKeys[bestIdxF].angle;
                                if (rot < 0.0)
                                    rot += 360.0f;
                                int bin = round(rot * factor);
                                if (bin == HISTO_LENGTH)
                                    bin = 0;
                                assert(bin >= 0 && bin < HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdxF);
                            }
                            nmatches++;
                        }
                    }
                }

                KFit++;
                Fit++;
            }
            else if (KFit->first < Fit->first)
            {
                KFit = vFeatVecKF.lower_bound(Fit->first);
            }
            else
            {
                Fit = F->mFeatVec.lower_bound(KFit->first);
            }
        }

        if (mbCheckOrientation)
        {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; i++)
            {
                if (i == ind1 || i == ind2 || i == ind3)
                    continue;
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    vpMapPointMatches[rotHist[i][j]] = nullptr;
                    nmatches--;
                }
            }
        }

        return nmatches;
    }

    int Matcher::SearchByBow(std::shared_ptr<Frame> frame1, std::shared_ptr<Frame> frame2, float nnRatio, bool mbCheckOrientation, std::vector<std::shared_ptr<MapPoint>> &matches)
    {

        const vector<cv::KeyPoint> &vKeysUn1 = frame1->mvKeys;
        const std::vector<std::shared_ptr<MapPoint>> vpMapPoints1 = frame1->getMapPointsV();
        const cv::Mat &Descriptors1 = frame1->Descriptors;

        const std::vector<cv::KeyPoint> &vKeysUn2 = frame2->mvKeys;
        const std::vector<std::shared_ptr<MapPoint>> vpMapPoints2 = frame2->getMapPointsV();
        const cv::Mat &Descriptors2 = frame2->Descriptors;

        matches = std::vector<std::shared_ptr<MapPoint>>(vpMapPoints1.size(), nullptr); //Match
        vector<bool> vbMatched2(vpMapPoints2.size(), false);

        std::vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);

        const float factor = 1.0f / HISTO_LENGTH;

        int nmatches = 0;

        DBoW3::FeatureVector::const_iterator f1it = frame1->mFeatVec.begin();
        DBoW3::FeatureVector::const_iterator f2it = frame2->mFeatVec.begin();
        DBoW3::FeatureVector::const_iterator f1end = frame1->mFeatVec.end();
        DBoW3::FeatureVector::const_iterator f2end = frame2->mFeatVec.end();

        while (f1it != f1end && f2it != f2end)
        {
            if (f1it->first == f2it->first)
            {
                for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];

                    std::shared_ptr<MapPoint> pMP1 = vpMapPoints1[idx1];
                    if (!pMP1)
                        continue;
                    if (pMP1->isBad())
                        continue;

                    const cv::Mat &d1 = Descriptors1.row(idx1);

                    int bestDist1 = 256;
                    int bestIdx2 = -1;
                    int bestDist2 = 256;

                    for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++)
                    {
                        const size_t idx2 = f2it->second[i2];

                        std::shared_ptr<MapPoint> pMP2 = vpMapPoints2[idx2];

                        if (vbMatched2[idx2] || !pMP2)
                            continue;

                        if (pMP2->isBad())
                            continue;

                        const cv::Mat &d2 = Descriptors2.row(idx2);

                        int dist = DescriptorDistance(d1, d2);

                        if (dist < bestDist1)
                        {
                            bestDist2 = bestDist1;
                            bestDist1 = dist;
                            bestIdx2 = idx2;
                        }
                        else if (dist < bestDist2)
                        {
                            bestDist2 = dist;
                        }
                    }

                    if (bestDist1 < TH_LOW)
                    {
                        if (static_cast<float>(bestDist1) < nnRatio * static_cast<float>(bestDist2))
                        {
                            // Match m;
                            // m.index1 = idx1;
                            // m.index2 = bestIdx2;
                            // m.dist = bestDist1;
                            // matches[idx1] = m;
                            matches[idx1]= vpMapPoints2[bestIdx2];
                            vbMatched2[bestIdx2] = true;

                            if (mbCheckOrientation)
                            {
                                float rot = vKeysUn1[idx1].angle - vKeysUn2[bestIdx2].angle;
                                if (rot < 0.0)
                                    rot += 360.0f;
                                int bin = round(rot * factor);
                                if (bin == HISTO_LENGTH)
                                    bin = 0;
                                assert(bin >= 0 && bin < HISTO_LENGTH);
                                rotHist[bin].push_back(idx1);
                            }
                            nmatches++;
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if (f1it->first < f2it->first)
            {
                f1it = frame1->mFeatVec.lower_bound(f2it->first);
            }
            else
            {
                f2it = frame2->mFeatVec.lower_bound(f1it->first);
            }
        }

        if (mbCheckOrientation)
        {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; i++)
            {
                if (i == ind1 || i == ind2 || i == ind3)
                    continue;
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    matches[rotHist[i][j]] = nullptr;
                    nmatches--;
                }
            }
        }

        return nmatches;
    }

    int Matcher::SearchLocalMapByProjection(shared_ptr<Frame> F, vector<shared_ptr<MapPoint>> &vpMapPoints, float th, float nnratio) //this is run from tracking thread -> access tMapPoints only!!
    {
        int nmatches = 0;

        const bool bFactor = th != 1.0;

        for (size_t iMP = 0, iend = vpMapPoints.size(); iMP < iend; ++iMP)
        {
            shared_ptr<MapPoint> pMP = vpMapPoints[iMP];
            
            if (!pMP->mbTrackInView)
                continue;

            if (pMP->isBad())
                continue;

            // const int &nPredictedLevel = pMP->mnTrackScaleLevel;

            // The size of the window will depend on the viewing direction
            float r = RadiusByViewingCos(pMP->mTrackViewCos);

            if (bFactor)
                r *= th;
            
            const vector<size_t> vIndices = F->GetFeaturesInArea(pMP->mTrackProjX, pMP->mTrackProjY, r);


            if (vIndices.empty())
                continue;

            const cv::Mat MPdescriptor = pMP->GetDescriptor();

            int bestDist = 256;
            int bestLevel = -1;
            int bestDist2 = 256;
            int bestLevel2 = -1;
            int bestIdx = -1;

            // Get best and second matches with near keypoints
            for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
            {
                const size_t idx = *vit;

                if (F->tMapPoints[idx])
                    if (F->tMapPoints[idx]->getNObservations() > 0)
                        continue;


                const cv::Mat &d = F->Descriptors.row(idx);

                const int dist = DescriptorDistance(MPdescriptor, d);

                if (dist < bestDist)
                {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    // bestLevel2 = bestLevel;
                    // bestLevel = F.mvKeysUn[idx].octave;
                    bestIdx = idx;
                }
                else if (dist < bestDist2)
                {
                    // bestLevel2 = F.mvKeysUn[idx].octave;
                    bestDist2 = dist;
                }
            }

            // Apply ratio to second match (only if best and second are in the same scale level)
            if (bestDist <= TH_HIGH)
            {
                if (bestLevel == bestLevel2 && bestDist > nnratio * bestDist2)
                    continue;

                F->tMapPoints[bestIdx] = pMP;
                nmatches++;
            }
        }

        return nmatches;
    }

    int Matcher::SearchBySim3(std::shared_ptr<Frame> pKF1, std::shared_ptr<Frame> pKF2, std::vector<std::shared_ptr<MapPoint>> &vpMatches12, const float &s12, const Mat33f &R12, const Vec3f &t12, const float th)
    {
        const float &fx = pKF1->HCalib->fxl();
        const float &fy = pKF1->HCalib->fyl();
        const float &cx = pKF1->HCalib->cxl();
        const float &cy = pKF1->HCalib->cyl();

        // Camera 1 from world
        auto poseKf1 = pKF1->fs->getPoseOpti(); //getPoseInverse();
        Mat33f R1w = poseKf1.rotationMatrix().cast<float>();
        Vec3f t1w = poseKf1.translation().cast<float>();

        //Camera 2 from world
        auto poseKf2 = pKF2->fs->getPoseOpti(); //getPoseInverse();
        Mat33f R2w = poseKf2.rotationMatrix().cast<float>();
        Vec3f t2w = poseKf2.translation().cast<float>();

        //Transformation between cameras
        Mat33f sR12 = s12 * R12;
        Mat33f sR21 = (1.0 / s12) * R12.transpose();
        Vec3f t21 = -sR21 * t12;

        const std::vector<std::shared_ptr<MapPoint>> vpMapPoints1 = pKF1->getMapPointsV();
        const int N1 = vpMapPoints1.size();

        const vector<std::shared_ptr<MapPoint>> vpMapPoints2 = pKF2->getMapPointsV();
        const int N2 = vpMapPoints2.size();

        vector<bool> vbAlreadyMatched1(N1, false);
        vector<bool> vbAlreadyMatched2(N2, false);

        for (int i = 0; i < N1; i++)
        {
            std::shared_ptr<MapPoint> pMP = vpMatches12[i];
            if (pMP)
            {
                vbAlreadyMatched1[i] = true;
                int idx2 = pMP->getIndexInKF(pKF2);
                if (idx2 >= 0 && idx2 < N2)
                    vbAlreadyMatched2[idx2] = true;
            }
        }

        vector<int> vnMatch1(N1, -1);
        vector<int> vnMatch2(N2, -1);

        // Transform from KF1 to KF2 and search
        for (int i1 = 0; i1 < N1; i1++)
        {
            std::shared_ptr<MapPoint> pMP = vpMapPoints1[i1];

            if (!pMP || vbAlreadyMatched1[i1])
                continue;

            if (pMP->isBad())
                continue;

            Vec3f p3Dw = pMP->getWorldPose();
            Vec3f p3Dc1 = R1w * p3Dw + t1w;
            Vec3f p3Dc2 = sR21 * p3Dc1 + t21;

            // Depth must be positive
            if (p3Dc2(2) < 0.0)
                continue;

            const float invz = 1.0 / p3Dc2(2);
            const float x = p3Dc2(0) * invz;
            const float y = p3Dc2(1) * invz;

            const float u = fx * x + cx;
            const float v = fy * y + cy;

            // Point must be inside the image
            if (u < mnMinX || u > mnMaxX || v < mnMinY || v > mnMaxY)
                continue;

            // const float maxDistance = pMP->GetMaxDistanceInvariance();
            // const float minDistance = pMP->GetMinDistanceInvariance();
            // const float dist3D = cv::norm(p3Dc2);

            // // Depth must be inside the scale invariance region
            // if (dist3D < minDistance || dist3D > maxDistance)
            //     continue;

            // // Compute predicted octave
            // const int nPredictedLevel = pMP->PredictScale(dist3D, pKF2);

            // Search in a radius
            const float radius = th * 1.0f; // pKF2->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u, v, radius);

            if (vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = INT_MAX;
            int bestIdx = -1;
            for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
            {
                const size_t idx = *vit;

                const cv::KeyPoint &kp = pKF2->mvKeys[idx];

                // if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
                //     continue;

                const cv::Mat &dKF = pKF2->Descriptors.row(idx);

                const int dist =  DescriptorDistance(dMP, dKF);

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if (bestDist <= TH_HIGH)
            {
                vnMatch1[i1] = bestIdx;
            }
        }

        // Transform from KF2 to KF2 and search
        for (int i2 = 0; i2 < N2; i2++)
        {
            std::shared_ptr<MapPoint> pMP = vpMapPoints2[i2];

            if (!pMP || vbAlreadyMatched2[i2])
                continue;

            if (pMP->isBad())
                continue;

            Vec3f p3Dw = pMP->getWorldPose();
            Vec3f p3Dc2 = R2w * p3Dw + t2w;
            Vec3f p3Dc1 = sR12 * p3Dc2 + t12;

            // Depth must be positive
            if (p3Dc1(2) < 0.0)
                continue;

            const float invz = 1.0 / p3Dc1(2);
            const float x = p3Dc1(0) * invz;
            const float y = p3Dc1(1) * invz;

            const float u = fx * x + cx;
            const float v = fy * y + cy;

            // Point must be inside the image
            if (u < mnMinX || u > mnMaxX || v < mnMinY || v > mnMaxY)
                continue;

            // const float maxDistance = pMP->GetMaxDistanceInvariance();
            // const float minDistance = pMP->GetMinDistanceInvariance();
            // const float dist3D = cv::norm(p3Dc1);

            // // Depth must be inside the scale pyramid of the image
            // if (dist3D < minDistance || dist3D > maxDistance)
            //     continue;

            // // Compute predicted octave
            // const int nPredictedLevel = pMP->PredictScale(dist3D, pKF1);

            // Search in a radius of 2.5*sigma(ScaleLevel)
            const float radius = th * 1.0f; //pKF1->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u, v, radius);

            if (vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = INT_MAX;
            int bestIdx = -1;
            for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
            {
                const size_t idx = *vit;

                const cv::KeyPoint &kp = pKF1->mvKeys[idx];

                // if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
                //     continue;

                const cv::Mat &dKF = pKF1->Descriptors.row(idx);

                const int dist = DescriptorDistance(dMP, dKF);

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if (bestDist <= TH_HIGH)
            {
                vnMatch2[i2] = bestIdx;
            }
        }

        // Check agreement
        int nFound = 0;

        for (int i1 = 0; i1 < N1; i1++)
        {
            int idx2 = vnMatch1[i1];

            if (idx2 >= 0)
            {
                int idx1 = vnMatch2[idx2];
                if (idx1 == i1)
                {
                    vpMatches12[i1] = vpMapPoints2[idx2];
                    nFound++;
                }
            }
        }

        return nFound;
    }

    int Matcher::SearchBySim3Projection(std::shared_ptr<Frame> pKF, Sim3 Scw, const std::vector<std::shared_ptr<MapPoint>> &vpPoints, std::vector<std::shared_ptr<MapPoint>> &vpMatched, int th)
    {
        // Get Calibration Parameters for later projection
        const float &fx = pKF->HCalib->fxl();
        const float &fy = pKF->HCalib->fyl();
        const float &cx = pKF->HCalib->cxl();
        const float &cy = pKF->HCalib->cyl();

        // Decompose Scw
        Mat44 sim3Transform = Scw.matrix();
        Mat33 Rcw = Scw.rotationMatrix();

        Vec3 tcw = Scw.translation(); // / Scw.scale();
        Vec3 Ow = -Rcw.transpose() * tcw;

        // cv::Mat sRcw = Scw.rowRange(0, 3).colRange(0, 3);
        // const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
        // cv::Mat Rcw = sRcw / scw;
        // cv::Mat tcw = Scw.rowRange(0, 3).col(3) / scw;
        // cv::Mat Ow = -Rcw.t() * tcw;

        // Set of MapPoints already found in the KeyFrame
        std::set<std::shared_ptr<MapPoint>> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(nullptr);

        int nmatches = 0;

        // For each Candidate MapPoint Project and Match
        for (int iMP = 0, iendMP = vpPoints.size(); iMP < iendMP; iMP++)
        {
            std::shared_ptr<MapPoint> pMP = vpPoints[iMP];

            // Discard Bad MapPoints and already found
            if (pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            Vec3 p3Dw = pMP->getWorldPose().cast<double>();

            // Transform into Camera Coords.
            Vec3 p3Dc = Rcw * p3Dw + tcw;

            // Depth must be positive
            if (p3Dc(2) < 0.0)
                continue;

            // Project into Image
            const float invz = 1 / p3Dc(2);
            const float x = p3Dc(0) * invz;
            const float y = p3Dc(1) * invz;

            const float u = fx * x + cx;
            const float v = fy * y + cy;

            // Point must be inside the image
            if (u < mnMinX || u > mnMaxX || v < mnMinY || v > mnMaxY)
                continue;

            // // Depth must be inside the scale invariance region of the point
            // const float maxDistance = pMP->GetMaxDistanceInvariance();
            // const float minDistance = pMP->GetMinDistanceInvariance();
            Vec3 PO = p3Dw - Ow;
            const float dist = PO.norm();

            // if (dist < minDistance || dist > maxDistance)
            //     continue;

            // Viewing angle must be less than 60 deg
            Vec3f Pn = pMP->GetNormal();

            if (PO.dot(Pn.cast<double>()) < 0.5 * dist)
                continue;

            // int nPredictedLevel = pMP->PredictScale(dist, pKF);

            // Search in a radius
            const float radius = th * 1.0f; //pKF->mvScaleFactors[nPredictedLevel];

            const std::vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

            if (vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = 256;
            int bestIdx = -1;
            for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
            {
                const size_t idx = *vit;
                if (vpMatched[idx])
                    continue;

                // const int &kpLevel = pKF->mvKeys[idx].octave;

                // if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                //     continue;

                const cv::Mat &dKF = pKF->Descriptors.row(idx);

                const int dist = DescriptorDistance(dMP, dKF);

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if (bestDist <= TH_LOW)
            {
                vpMatched[bestIdx] = pMP;
                nmatches++;
            }
        }

        return nmatches;
    }

    int Matcher::Fuse(std::shared_ptr<Frame> pKF, Sim3 Scw, const std::vector<std::shared_ptr<MapPoint>> &vpPoints, float th, std::vector<std::shared_ptr<MapPoint>> &vpReplacePoint) //this is run from mapping thread or loop closure thread -> access mvpMapPoints
    {
        // Get Calibration Parameters for later projection
        const float &fx = pKF->HCalib->fxl();
        const float &fy = pKF->HCalib->fyl();
        const float &cx = pKF->HCalib->cxl();
        const float &cy = pKF->HCalib->cyl();

        // Decompose Scw
        Mat33f sRcw = Scw.rotationMatrix().cast<float>();
        // float scw = sRcw.row(0).dot(sRcw.row(0));
        // if (scw != 1.0)
        //     sRcw = sRcw / scw;
        Vec3f tcw = Scw.translation().cast<float>(); // / Scw.scale();
        Vec3f Ow = -sRcw.transpose() * tcw;

        // cv::Mat sRcw = Scw.rowRange(0, 3).colRange(0, 3);
        // const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
        // cv::Mat Rcw = sRcw / scw;
        // cv::Mat tcw = Scw.rowRange(0, 3).col(3) / scw;
        // cv::Mat Ow = -Rcw.t() * tcw;

        // Set of MapPoints already found in the KeyFrame
        const std::set<std::shared_ptr<MapPoint>> spAlreadyFound = pKF->getMapPointsS();

        int nFused = 0;

        const int nPoints = vpPoints.size();

        // For each candidate MapPoint project and match
        for (int iMP = 0; iMP < nPoints; iMP++)
        {
            std::shared_ptr<MapPoint> pMP = vpPoints[iMP];

            // Discard Bad MapPoints and already found
            if (pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            Vec3f p3Dw = pMP->getWorldPose();

            // Transform into Camera Coords.
            Vec3f p3Dc = sRcw * p3Dw + tcw;

            // Depth must be positive
            if (p3Dc(2) < 0.0f)
                continue;

            // Project into Image
            const float invz = 1.0 / p3Dc(2);
            const float x = p3Dc(0) * invz;
            const float y = p3Dc(1) * invz;

            const float u = fx * x + cx;
            const float v = fy * y + cy;

            // Point must be inside the image
            if (u < mnMinX || u > mnMaxX || v < mnMinY || v > mnMaxY)
                continue;

            // Depth must be inside the scale pyramid of the image
            // const float maxDistance = pMP->GetMaxDistanceInvariance();
            // const float minDistance = pMP->GetMinDistanceInvariance();
            Vec3f PO = p3Dw - Ow;
            const float dist3D = PO.norm();

            // if (dist3D < minDistance || dist3D > maxDistance)
            //     continue;

            // Viewing angle must be less than 60 deg
            Vec3f Pn = pMP->GetNormal();

            if (PO.dot(Pn) < 0.5 * dist3D)
                continue;

            // Compute predicted scale level
            // const int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

            // Search in a radius
            // const float radius = th * pKF->mvScaleFactors[nPredictedLevel];
            const float radius = th * 1.0f;

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

            if (vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius

            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = INT_MAX;
            int bestIdx = -1;
            for (vector<size_t>::const_iterator vit = vIndices.begin(); vit != vIndices.end(); vit++)
            {
                const size_t idx = *vit;
                // const int &kpLevel = pKF->mvKeysUn[idx].octave;

                // if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                //     continue;

                const cv::Mat &dKF = pKF->Descriptors.row(idx);

                int dist = DescriptorDistance(dMP, dKF);

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            // If there is already a MapPoint replace otherwise add new measurement
            if (bestDist <= TH_LOW)
            {
                std::shared_ptr<MapPoint> pMPinKF = pKF->getMapPoint(bestIdx);
                if (pMPinKF)
                {
                    if (!pMPinKF->isBad())
                        vpReplacePoint[iMP] = pMPinKF;
                }
                else
                {
                    pMP->AddObservation(pKF, bestIdx);
                    pKF->addMapPointMatch(pMP, bestIdx);
                }
                nFused++;
            }
        }

        return nFused;
    }

    int Matcher::Fuse(std::shared_ptr<Frame> pKF, const std::vector<std::shared_ptr<MapPoint>> &vpMapPoints, const float th) //this is run from mapping thread or loop closure thread -> access mvpMapPoints
    {

        //Mat33f Rcw = pKF->fs->getPoseInverse().rotationMatrix().cast<float>();
        Mat33f Rcw = pKF->fs->getPoseOpti().rotationMatrix().cast<float>();

        // Vec3f tcw = pKF->fs->getPoseInverse().translation().cast<float>();
        Vec3f tcw = pKF->fs->getPoseOpti().translation().cast<float>();


        const float &fx = pKF->HCalib->fxl();
        const float &fy = pKF->HCalib->fyl();
        const float &cx = pKF->HCalib->cxl();
        const float &cy = pKF->HCalib->cyl();
        // const float &bf = pKF->mbf;

        Vec3f Ow = pKF->fs->getCameraCenter().cast<float>();

        int nFused = 0;

        const int nMPs = vpMapPoints.size();

        for (int i = 0; i < nMPs; i++)
        {
            std::shared_ptr<MapPoint> pMP = vpMapPoints[i];

            if (!pMP)
                continue;

            if (pMP->isBad() || pMP->isInKeyframe(pKF))
                continue;

            Vec3f p3Dw = pMP->getWorldPose();
            Vec3f p3Dc = Rcw * p3Dw + tcw;

            // Depth must be positive
            if (p3Dc(2) < 0.0f)
                continue;

            const float invz = 1 / p3Dc(2);
            const float x = p3Dc(0) * invz;
            const float y = p3Dc(1) * invz;

            const float u = fx * x + cx;
            const float v = fy * y + cy;

            // Point must be inside the image
            if (u < mnMinX || u > mnMaxX || v < mnMinY || v > mnMaxY)
                continue;

            // const float maxDistance = pMP->GetMaxDistanceInvariance();
            // const float minDistance = pMP->GetMinDistanceInvariance();
            Vec3f PO = p3Dw - Ow;
            const float dist3D = PO.norm();

            // Depth must be inside the scale pyramid of the image
            // if (dist3D < minDistance || dist3D > maxDistance)
            //     continue;

            // Viewing angle must be less than 60 deg
            Vec3f Pn = pMP->GetNormal();

            if (PO.dot(Pn) < 0.5 * dist3D)
                continue;

            // int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

            // Search in a radius
            // const float radius = th * pKF->mvScaleFactors[nPredictedLevel];
            const float radius = th * 1.0f;

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

            if (vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius

            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = 256;
            int bestIdx = -1;
            for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
            {
                const size_t idx = *vit;

                const cv::KeyPoint &kp = pKF->mvKeys[idx];

                const int &kpLevel = kp.octave;

                // if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                //     continue;

                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = u - kpx;
                const float ey = v - kpy;
                const float e2 = ex * ex + ey * ey;

                // if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 5.99)
                //     continue;

                if (e2 * 1.0f > 5.99)
                    continue;

                const cv::Mat &dKF = pKF->Descriptors.row(idx);

                const int dist = DescriptorDistance(dMP, dKF);

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            // If there is already a MapPoint replace otherwise add new measurement
            if (bestDist <= TH_LOW)
            {
                std::shared_ptr<MapPoint> pMPinKF = pKF->getMapPoint(bestIdx);
                if (pMPinKF)
                {
                    if (!pMPinKF->isBad())
                    {

                        // pMP->Replace(pMPinKF); //enable this and comment the if-else to keep the old Mps only
                        if (pMPinKF->getNObservations() > pMP->getNObservations())
                            pMP->Replace(pMPinKF);
                        else
                            pMPinKF->Replace(pMP);
                    }
                }
                else
                {
                    pMP->AddObservation(pKF, bestIdx);
                    pKF->addMapPointMatch(pMP, bestIdx);
                }
                nFused++;
            }
        }

        return nFused;
    }

    int Matcher::SearchByProjectionFrameToFrame(std::shared_ptr<Frame> CurrentFrame, const std::shared_ptr<Frame> LastFrame, const float th, bool mbCheckOrientation) //this is run from tracking thread -> access tMapPoints only!!
    {
        int nmatches = 0;

        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f / HISTO_LENGTH;

        const Mat33f Rcw = CurrentFrame->fs->getPoseInverse().rotationMatrix().cast<float>(); //currFrame not added to map yet (does not have poseOpti!!)  CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
        const Vec3f tcw = CurrentFrame->fs->getPoseInverse().translation().cast<float>();     //CurrentFrame.mTcw.rowRange(0, 3).col(3);

        const Vec3f twc = -Rcw.transpose() * tcw;

        const Mat33f Rlw = LastFrame->fs->getPoseInverse().rotationMatrix().cast<float>(); //mTcw.rowRange(0, 3).colRange(0, 3);
        const Vec3f tlw = LastFrame->fs->getPoseInverse().translation().cast<float>();       //mTcw.rowRange(0, 3).col(3);

   
        for (int i = 0; i < LastFrame->nFeatures; ++i)
        {
            std::shared_ptr<MapPoint> pMP = LastFrame->tMapPoints[i];

            if (pMP)
            {
                if (!LastFrame->mvbOutlier[i])
                {
                    // Project
                    Vec3f x3Dw = pMP->getWorldPose();
                    Vec3f x3Dc = Rcw * x3Dw + tcw;

                    const float xc = x3Dc(0);
                    const float yc = x3Dc(1);
                    const float invzc = 1.0 / x3Dc(2);

                    if (invzc < 0)
                        continue;

                    float u = CurrentFrame->HCalib->fxl() * xc * invzc + CurrentFrame->HCalib->cxl();
                    float v = CurrentFrame->HCalib->fyl() * yc * invzc + CurrentFrame->HCalib->cyl();

                    if (u < mnMinX || u > mnMaxX)
                        continue;
                    if (v < mnMinY || v > mnMaxY)
                        continue;

                    // int nLastOctave = LastFrame.mvKeys[i].octave;

                    // Search in a window. Size depends on scale
                    // float radius = th * CurrentFrame.mvScaleFactors[nLastOctave];
                    float radius = th * 1.0f;


                    vector<size_t> vIndices2;

                    // if (bForward)
                    //     vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nLastOctave);
                    // else if (bBackward)
                    //     vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, 0, nLastOctave);
                    // else
                    vIndices2 = CurrentFrame->GetFeaturesInArea(u, v, radius);

                    if (vIndices2.empty())
                        continue;

                    const cv::Mat dMP = pMP->GetDescriptor();

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    for (vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++)
                    {
                        const size_t i2 = *vit;
                        if (CurrentFrame->tMapPoints[i2])
                            if (CurrentFrame->tMapPoints[i2]->getNObservations() > 0)
                                continue;

                        const cv::Mat &d = CurrentFrame->Descriptors.row(i2);

                        const int dist = DescriptorDistance(dMP, d);

                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            bestIdx2 = i2;
                        }
                    }

                    if (bestDist <= TH_HIGH)
                    {
                        CurrentFrame->tMapPoints[bestIdx2] = pMP;
                        nmatches++;

                        if (mbCheckOrientation)
                        {
                            float rot = LastFrame->mvKeys[i].angle - CurrentFrame->mvKeys[bestIdx2].angle;
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2);
                        }
                    }
                }
            }
        }

        //Apply rotation consistency
        if (mbCheckOrientation)
        {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; i++)
            {
                if (i != ind1 && i != ind2 && i != ind3)
                {
                    for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                    {
                        CurrentFrame->tMapPoints[rotHist[i][j]].reset();
                        nmatches--;
                    }
                }
            }
        }

        return nmatches;
    }









































    int Matcher::SearchByProjection(shared_ptr<Frame> &CurrentFrame, shared_ptr<Frame> &pKF, const set<shared_ptr<MapPoint>> &sAlreadyFound, const float th, const int ORBdist, bool mbCheckOrientation)
    {
        int nmatches = 0;
        auto CurrPoseInv = CurrentFrame->fs->getPoseInverse();
        auto currPose = CurrentFrame->fs->getPose();
        auto Rcw = CurrPoseInv.rotationMatrix();
        auto tcw = CurrPoseInv.translation();
        auto Ow = CurrentFrame->fs->getCameraCenter();
    

        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f / HISTO_LENGTH;

        const vector<shared_ptr<MapPoint>> vpMPs = pKF->getMapPointsV();

        for (size_t i = 0, iend = vpMPs.size(); i < iend; ++i)
        {

            shared_ptr<MapPoint> pMP = vpMPs[i];

            if (pMP)
            {
                if (!pMP->isBad() && !sAlreadyFound.count(pMP))
                {
                    //Project
                    Vec3f x3Dw = pMP->getWorldPose(); //GetWorldPos();
                    Vec3f x3Dc = Rcw.cast<float>() * x3Dw + tcw.cast<float>();

                    const float xc = x3Dc(0);
                    const float yc = x3Dc(1);
                    const float invzc = 1.0 / x3Dc(2);

                    const float u =  CurrentFrame->HCalib->fxl() * xc * invzc + CurrentFrame->HCalib->cxl();
                    const float v = CurrentFrame->HCalib->fyl() * yc * invzc + CurrentFrame->HCalib->cyl();

                    if (u < mnMinX || u > mnMaxX)
                        continue;
                    if (v < mnMinY || v > mnMaxY)
                        continue;

                    // Compute predicted scale level
                    // Vec3f PO = x3Dw - Ow.cast<float>();
                    
                    // float dist3D = PO.norm();

                    // const float maxDistance = pMP->GetMaxDistanceInvariance();
                    // const float minDistance = pMP->GetMinDistanceInvariance();

                    // // Depth must be inside the scale pyramid of the image
                    // if (dist3D < minDistance || dist3D > maxDistance)
                    //     continue;

                    // int nPredictedLevel = pMP->PredictScale(dist3D, &CurrentFrame);

                    // Search in a window
                    const float radius = th;

                    const vector<size_t> vIndices2 = CurrentFrame->GetFeaturesInArea(u, v, radius);

                    if (vIndices2.empty())
                        continue;

                    const cv::Mat dMP = pMP->GetDescriptor();

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    for (vector<size_t>::const_iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
                    {
                        const size_t i2 = *vit;
                        if (CurrentFrame->getMapPoint(i2))
                            continue;

                        const cv::Mat &d = CurrentFrame->Descriptors.row(i2);

                        const int dist = DescriptorDistance(dMP, d);

                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            bestIdx2 = i2;
                        }
                    }

                    if (bestDist <= ORBdist)
                    {
                        CurrentFrame->addMapPointMatch(pMP, bestIdx2);
                        nmatches++;

                        if (mbCheckOrientation)
                        {
                            float rot = pMP->sourceFrame->mvKeys[pMP->index].angle - CurrentFrame->mvKeys[bestIdx2].angle; // pKF->mvKeys[i].angle
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2);
                        }
                    }
                }
            }
        }

        if (mbCheckOrientation)
        {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; i++)
            {
                if (i != ind1 && i != ind2 && i != ind3)
                {
                    for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                    {
                        CurrentFrame->addMapPointMatch(nullptr, rotHist[i][j]);
                        nmatches--;
                    }
                }
            }
        }

        return nmatches;
    }

    int Matcher::searchWithEpipolar(shared_ptr<Frame> pKF1, shared_ptr<Frame> pKF2, vector<pair<size_t, size_t> > &vMatchedPairs, bool mbCheckOrientation)
    {
        const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

        //Compute epipole in second image
        Vec3 Cw = pKF1->fs->getCameraCenter();
        SE3 Kf1Pose = pKF1->fs->getPose();
        SO3 R1w = Kf1Pose.rotationMatrix();
        Vec3 t1w = Kf1Pose.translation();

        SE3 Kf2Pose = pKF2->fs->getPose();
        SO3 R2w = Kf2Pose.rotationMatrix();
        Vec3 t2w = Kf2Pose.translation();
        Vec3 C2 = R2w * Cw + t2w;

        SO3 R2wt = R2w.inverse();
        SO3 R12 = R1w * R2wt;
        Vec3 t12 = - (R1w * R2wt * t2w) + t1w;

        Mat33 t12x = Skew(t12);
       

        //compute fundamental matrix
        Mat33 K1ti = pKF1->HCalib->getInvCalibMatrix().transpose().cast<double>();
        Mat33 K2i = pKF1->HCalib->getInvCalibMatrix().cast<double>();
        Mat33f Fundamental = (K1ti * t12x * R12.matrix() * K2i).cast<float>();

        const float invz = 1.0f / C2(2);
        const float ex = pKF2->HCalib->fxl() * C2(0) * invz + pKF2->HCalib->cxl();
        const float ey = pKF2->HCalib->fyl() * C2(1) * invz + pKF2->HCalib->cyl();

        // Find matches between not tracked keypoints
        // Matching speed-up by ORB Vocabulary
        // Compare only ORB that share the same node

        int nmatches = 0;
        vector<bool> vbMatched2(pKF2->nFeatures, false);
        vector<int> vMatches12(pKF1->nFeatures, -1);

        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);

        const float factor = 1.0f / HISTO_LENGTH;

        DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

        while (f1it != f1end && f2it != f2end)
        {
            if (f1it->first == f2it->first)
            {
                for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];

                    shared_ptr<MapPoint> pMP1 = pKF1->getMapPoint(idx1);

                    // If there is already a MapPoint skip
                    if (pMP1)
                        continue;

                    const cv::KeyPoint &kp1 = pKF1->mvKeys[idx1];

                    const cv::Mat &d1 = pKF1->Descriptors.row(idx1);

                    int bestDist = TH_LOW;
                    int bestIdx2 = -1;

                    for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++)
                    {
                        size_t idx2 = f2it->second[i2];

                        shared_ptr<MapPoint> pMP2 = pKF2->getMapPoint(idx2);

                        // If we have already matched or there is a MapPoint skip
                        if (vbMatched2[idx2] || pMP2)
                            continue;

                        const cv::Mat &d2 = pKF2->Descriptors.row(idx2);

                        int dist = DescriptorDistance(d1, d2);

                        if (dist > TH_LOW || dist > bestDist)
                            continue;

                        const cv::KeyPoint &kp2 = pKF2->mvKeys[idx2];

                       
                        float distex = ex - kp2.pt.x;
                        float distey = ey - kp2.pt.y;
                        if (distex * distex + distey * distey < 100 )
                                continue;
                        

                        if (CheckDistEpipolarLine(kp1, kp2, Fundamental))
                        {
                            bestIdx2 = idx2;
                            bestDist = dist;
                        }
                    }

                    if (bestIdx2 >= 0)
                    {
                        const cv::KeyPoint &kp2 = pKF2->mvKeys[bestIdx2];
                        vMatches12[idx1] = bestIdx2;
                        nmatches++;

                        if (mbCheckOrientation)
                        {
                            float rot = kp1.angle - kp2.angle;
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if (f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        if (mbCheckOrientation)
        {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; i++)
            {
                if (i == ind1 || i == ind2 || i == ind3)
                    continue;
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    vMatches12[rotHist[i][j]] = -1;
                    nmatches--;
                }
            }
        }

        vMatchedPairs.clear();
        vMatchedPairs.reserve(nmatches);

        for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
        {
            if (vMatches12[i] < 0)
                continue;
            vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
        }

        return nmatches;
    }

    bool Matcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, Mat33f &F12)
    {
        // Epipolar line in second image l = x1'F12 = [a b c]
         float a = kp1.pt.x * F12(0, 0) + kp1.pt.y * F12(1, 0) + F12(2, 0);
        const float b = kp1.pt.x * F12(0, 1) + kp1.pt.y * F12(1, 1) + F12(2, 1);
        const float c = kp1.pt.x * F12(0, 2) + kp1.pt.y * F12(1, 2) + F12(2, 2);

        const float num = a * kp2.pt.x + b * kp2.pt.y + c;

        const float den = a * a + b * b;

        if (den == 0)
            return false;

        const float dsqr = num * num / den;

        return dsqr < 3.84;
    }



} // namespace HSLAM
