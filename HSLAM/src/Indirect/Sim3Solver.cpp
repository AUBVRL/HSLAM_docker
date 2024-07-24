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

#include "Indirect/Sim3Solver.h"

//#include <vector>
#include <cmath>
// #include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "Indirect/Frame.h"
#include "Indirect/MapPoint.h"
#include "util/FrameShell.h"
#include "FullSystem/HessianBlocks.h"
// #include "ORBmatcher.h"

// #include "Thirdparty/DBoW2/DUtils/Random.h"

namespace HSLAM
{

    using namespace std;
    Sim3Solver::Sim3Solver(std::shared_ptr<Frame> pKF1, std::shared_ptr<Frame> pKF2, const vector<std::shared_ptr<MapPoint>> &vpMatched12, const bool bFixScale) : mnIterations(0), mnBestInliers(0), mbFixScale(bFixScale)
    {
        mpKF1 = pKF1;
        mpKF2 = pKF2;

        vector<std::shared_ptr<MapPoint>> vpKeyFrameMP1 = pKF1->getMapPointsV();
        mN1 = vpMatched12.size();

        mvpMapPoints1.reserve(mN1);
        mvpMapPoints2.reserve(mN1);
        mvpMatches12 = vpMatched12;
        mvnIndices1.reserve(mN1);
        mvX3Dc1.reserve(mN1);
        mvX3Dc2.reserve(mN1);

        mvAllIndices.reserve(mN1);

        rng = cv::RNG();

        size_t idx = 0;
        for (int i1 = 0; i1 < mN1; i1++)
        {
            if (vpMatched12[i1])
            {
                std::shared_ptr<MapPoint> pMP1 = vpKeyFrameMP1[i1];
                std::shared_ptr<MapPoint> pMP2 = vpMatched12[i1];

                if (!pMP1)
                    continue;

                if (pMP1->isBad() || pMP2->isBad())
                    continue;

                int indexKF1 = pMP1->getIndexInKF(pKF1);
                int indexKF2 = pMP2->getIndexInKF(pKF2);

                if (indexKF1 < 0 || indexKF2 < 0)
                    continue;

                const cv::KeyPoint &kp1 = pKF1->mvKeys[indexKF1];
                const cv::KeyPoint &kp2 = pKF2->mvKeys[indexKF2];

                const float sigmaSquare1 = 1.0f; //pKF1->mvLevelSigma2[kp1.octave];
                const float sigmaSquare2 = 1.0f; //pKF2->mvLevelSigma2[kp2.octave];

                mvnMaxError1.push_back(9.210 * sigmaSquare1);
                mvnMaxError2.push_back(9.210 * sigmaSquare2);

                mvpMapPoints1.push_back(pMP1);
                mvpMapPoints2.push_back(pMP2);
                mvnIndices1.push_back(i1);

                Vec3f X3D1w = pKF1->fs->getPoseInverse().cast<float>() * pMP1->getWorldPose();
                mvX3Dc1.push_back((cv::Mat_<float>(3, 1) << X3D1w(0), X3D1w(1), X3D1w(2)));

                Vec3f X3D2w = pKF2->fs->getPoseInverse().cast<float>() * pMP2->getWorldPose();
                mvX3Dc2.push_back((cv::Mat_<float>(3, 1) << X3D2w(0), X3D2w(1), X3D2w(2)));

                mvAllIndices.push_back(idx);
                idx++;
            }
        }

        mK1 = pKF1->HCalib->getCalibMatrix();
        mK2 = pKF2->HCalib->getCalibMatrix();

        FromCameraToImage(mvX3Dc1, mvP1im1, mK1);
        FromCameraToImage(mvX3Dc2, mvP2im2, mK2);

        SetRansacParameters();
    }

    void Sim3Solver::SetRansacParameters(double probability, int minInliers, int maxIterations)
    {
        mRansacProb = probability;
        mRansacMinInliers = minInliers;
        mRansacMaxIts = maxIterations;

        N = mvpMapPoints1.size(); // number of correspondences

        mvbInliersi.resize(N);

        // Adjust Parameters according to number of correspondences
        float epsilon = (float)mRansacMinInliers / N;

        // Set RANSAC iterations according to probability, epsilon, and max iterations
        int nIterations;

        if (mRansacMinInliers == N)
            nIterations = 1;
        else
            nIterations = ceil(log(1 - mRansacProb) / log(1 - pow(epsilon, 3)));

        mRansacMaxIts = max(1, min(nIterations, mRansacMaxIts));

        mnIterations = 0;
    }

    cv::Mat Sim3Solver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
    {
        bNoMore = false;
        vbInliers = vector<bool>(mN1, false);
        nInliers = 0;

        if (N < mRansacMinInliers)
        {
            bNoMore = true;
            return cv::Mat();
        }

        vector<size_t> vAvailableIndices;

        cv::Mat P3Dc1i(3, 3, CV_32F);
        cv::Mat P3Dc2i(3, 3, CV_32F);

        int nCurrentIterations = 0;
        while (mnIterations < mRansacMaxIts && nCurrentIterations < nIterations)
        {
            nCurrentIterations++;
            mnIterations++;

            vAvailableIndices = mvAllIndices;

            // Get min set of points
            for (short i = 0; i < 3; ++i)
            {
                int randi = rng.uniform(0, vAvailableIndices.size() - 1); // DUtils::Random::RandomInt(0, );
                int idx = vAvailableIndices[randi];

                mvX3Dc1[idx].copyTo(P3Dc1i.col(i));
                mvX3Dc2[idx].copyTo(P3Dc2i.col(i));

                vAvailableIndices[randi] = vAvailableIndices.back();
                vAvailableIndices.pop_back();
            }

            ComputeSim3(P3Dc1i, P3Dc2i);

            CheckInliers();

            if (mnInliersi >= mnBestInliers)
            {
                mvbBestInliers = mvbInliersi;
                mnBestInliers = mnInliersi;
                mBestT12 = mT12i.clone();
                mBestRotation = mR12i.clone();
                mBestTranslation = mt12i.clone();
                mBestScale = ms12i;

                if (mnInliersi > mRansacMinInliers)
                {
                    nInliers = mnInliersi;
                    for (int i = 0; i < N; i++)
                        if (mvbInliersi[i])
                            vbInliers[mvnIndices1[i]] = true;

                    return mBestT12;
                }
            }
        }

        if (mnIterations >= mRansacMaxIts)
            bNoMore = true;

        return cv::Mat();
    }

    cv::Mat Sim3Solver::find(vector<bool> &vbInliers12, int &nInliers)
    {
        bool bFlag;
        return iterate(mRansacMaxIts, bFlag, vbInliers12, nInliers);
    }

    void Sim3Solver::ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C)
    {
        cv::reduce(P, C, 1, CV_REDUCE_SUM);
        C = C / P.cols;

        for (int i = 0; i < P.cols; i++)
        {
            Pr.col(i) = P.col(i) - C;
        }
    }

    void Sim3Solver::ComputeSim3(cv::Mat &P1, cv::Mat &P2)
    {
        // Custom implementation of:
        // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

        // Step 1: Centroid and relative coordinates

        cv::Mat Pr1(P1.size(), P1.type()); // Relative coordinates to centroid (set 1)
        cv::Mat Pr2(P2.size(), P2.type()); // Relative coordinates to centroid (set 2)
        cv::Mat O1(3, 1, Pr1.type());      // Centroid of P1
        cv::Mat O2(3, 1, Pr2.type());      // Centroid of P2

        ComputeCentroid(P1, Pr1, O1);
        ComputeCentroid(P2, Pr2, O2);

        // Step 2: Compute M matrix

        cv::Mat M = Pr2 * Pr1.t();

        // Step 3: Compute N matrix

        double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

        cv::Mat N(4, 4, P1.type());

        N11 = M.at<float>(0, 0) + M.at<float>(1, 1) + M.at<float>(2, 2);
        N12 = M.at<float>(1, 2) - M.at<float>(2, 1);
        N13 = M.at<float>(2, 0) - M.at<float>(0, 2);
        N14 = M.at<float>(0, 1) - M.at<float>(1, 0);
        N22 = M.at<float>(0, 0) - M.at<float>(1, 1) - M.at<float>(2, 2);
        N23 = M.at<float>(0, 1) + M.at<float>(1, 0);
        N24 = M.at<float>(2, 0) + M.at<float>(0, 2);
        N33 = -M.at<float>(0, 0) + M.at<float>(1, 1) - M.at<float>(2, 2);
        N34 = M.at<float>(1, 2) + M.at<float>(2, 1);
        N44 = -M.at<float>(0, 0) - M.at<float>(1, 1) + M.at<float>(2, 2);

        N = (cv::Mat_<float>(4, 4) << N11, N12, N13, N14,
             N12, N22, N23, N24,
             N13, N23, N33, N34,
             N14, N24, N34, N44);

        // Step 4: Eigenvector of the highest eigenvalue

        cv::Mat eval, evec;

        cv::eigen(N, eval, evec); //evec[0] is the quaternion of the desired rotation

        cv::Mat vec(1, 3, evec.type());
        (evec.row(0).colRange(1, 4)).copyTo(vec); //extract imaginary part of the quaternion (sin*axis)

        // Rotation angle. sin is the norm of the imaginary part, cos is the real part
        double ang = atan2(norm(vec), evec.at<float>(0, 0));

        vec = 2 * ang * vec / norm(vec); //Angle-axis representation. quaternion angle is the half

        mR12i.create(3, 3, P1.type());

        cv::Rodrigues(vec, mR12i); // computes the rotation matrix from angle-axis

        // Step 5: Rotate set 2

        cv::Mat P3 = mR12i * Pr2;

        // Step 6: Scale

        if (!mbFixScale)
        {
            double nom = Pr1.dot(P3);
            cv::Mat aux_P3(P3.size(), P3.type());
            aux_P3 = P3;
            cv::pow(P3, 2, aux_P3);
            double den = 0;

            for (int i = 0; i < aux_P3.rows; i++)
            {
                for (int j = 0; j < aux_P3.cols; j++)
                {
                    den += aux_P3.at<float>(i, j);
                }
            }

            ms12i = nom / den;
        }
        else
            ms12i = 1.0f;

        // Step 7: Translation

        mt12i.create(1, 3, P1.type());
        mt12i = O1 - ms12i * mR12i * O2;

        // Step 8: Transformation

        // Step 8.1 T12
        mT12i = cv::Mat::eye(4, 4, P1.type());

        cv::Mat sR = ms12i * mR12i;

        sR.copyTo(mT12i.rowRange(0, 3).colRange(0, 3));
        mt12i.copyTo(mT12i.rowRange(0, 3).col(3));

        // Step 8.2 T21

        mT21i = cv::Mat::eye(4, 4, P1.type());

        cv::Mat sRinv = (1.0 / ms12i) * mR12i.t();

        sRinv.copyTo(mT21i.rowRange(0, 3).colRange(0, 3));
        cv::Mat tinv = -sRinv * mt12i;
        tinv.copyTo(mT21i.rowRange(0, 3).col(3));
    }

    void Sim3Solver::CheckInliers()
    {
        vector<cv::Mat> vP1im2, vP2im1;
        Project(mvX3Dc2, vP2im1, mT12i, mK1);
        Project(mvX3Dc1, vP1im2, mT21i, mK2);

        mnInliersi = 0;

        for (size_t i = 0; i < mvP1im1.size(); i++)
        {
            cv::Mat dist1 = mvP1im1[i] - vP2im1[i];
            cv::Mat dist2 = vP1im2[i] - mvP2im2[i];

            const float err1 = dist1.dot(dist1);
            const float err2 = dist2.dot(dist2);

            if (err1 < mvnMaxError1[i] && err2 < mvnMaxError2[i])
            {
                mvbInliersi[i] = true;
                mnInliersi++;
            }
            else
                mvbInliersi[i] = false;
        }
    }

    Mat33f Sim3Solver::GetEstimatedRotation()
    {
        
        Mat33f rotation;
        rotation << mBestRotation.at<float>(0, 0), mBestRotation.at<float>(0, 1), mBestRotation.at<float>(0, 2),
            mBestRotation.at<float>(1, 0), mBestRotation.at<float>(1, 1), mBestRotation.at<float>(1, 2),
            mBestRotation.at<float>(2, 0), mBestRotation.at<float>(2, 1), mBestRotation.at<float>(2, 2);

        return rotation;
    }

    Vec3f Sim3Solver::GetEstimatedTranslation()
    {
        Vec3f translation;
        translation << mBestTranslation.at<float>(0), mBestTranslation.at<float>(1), mBestTranslation.at<float>(2);
        return translation;
    }

    float Sim3Solver::GetEstimatedScale()
    {
        return mBestScale;
    }

    void Sim3Solver::Project(const vector<cv::Mat> &vP3Dw, vector<cv::Mat> &vP2D, cv::Mat Tcw, Mat33f K)
    {
        cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
        cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
        const float &fx = K(0, 0);
        const float &fy = K(1, 1);
        const float &cx = K(0, 2);
        const float &cy = K(1, 2);

        vP2D.clear();
        vP2D.reserve(vP3Dw.size());

        for (size_t i = 0, iend = vP3Dw.size(); i < iend; i++)
        {
            cv::Mat P3Dc = Rcw * vP3Dw[i] + tcw;
            const float invz = 1 / (P3Dc.at<float>(2));
            const float x = P3Dc.at<float>(0) * invz;
            const float y = P3Dc.at<float>(1) * invz;

            vP2D.push_back((cv::Mat_<float>(2, 1) << fx * x + cx, fy * y + cy));
        }
    }

    void Sim3Solver::FromCameraToImage(const std::vector<cv::Mat> &vP3Dc, vector<cv::Mat> &vP2D, Mat33f K)
    {
        const float &fx = K(0, 0);
        const float &fy = K(1, 1);
        const float &cx = K(0, 2);
        const float &cy = K(1, 2);

        vP2D.clear();
        vP2D.reserve(vP3Dc.size());

        for (size_t i = 0, iend = vP3Dc.size(); i < iend; i++)
        {
            const float invz = 1.0f / (vP3Dc[i].at<float>(2));
            const float x = vP3Dc[i].at<float>(0) * invz;
            const float y = vP3Dc[i].at<float>(1) * invz;

            vP2D.push_back((cv::Mat_<float>(2, 1) << fx * x + cx, fy * y + cy));
        }
    }

} // namespace HSLAM
