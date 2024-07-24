#pragma once
#ifndef __IndirectTracker_H__
#define __IndirectTracker_H__

#include "util/NumType.h"


namespace HSLAM
{
    class MapPoint;
    class CalibHessian;

    void projectMatchesToRef(std::shared_ptr<Frame> currFrame, SE3 refFramePose, std::vector<Vec10f> &dataOut);

    static std::vector<Vec10f> MatchesinRef; // uMatch, vMatch, uinRef, vinRef, idepthInRef, idHessianNormalized, uInCurr, vInCurr, idepthInCurr, HuberW, 
    static std::vector<Vec2f> Residuals; // uMatch, vMatch, uinRef, vinRef, idepthInRef, idHessianNormalized, uInCurr, vInCurr, idepthInCurr, HuberW, 

    Vec3 calcRes(CalibHessian *HCalib, SE3 &refToNew, std::vector<Vec10f> &vMPs, std::vector<Vec2f>& _Residuals, float cutoffTH);

    void calcGSSSE(std::shared_ptr<Frame> frame, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew);

    bool trackNewestCoarse(std::shared_ptr<Frame> newFrame, std::shared_ptr<Frame> refFrame, SE3 &lastToNew_out, int coarsestLvl);


} // namespace HSLAM

#endif