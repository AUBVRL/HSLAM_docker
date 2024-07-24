#include "IndirectTracker.h"
#include "Indirect/Frame.h"
#include "Indirect/MapPoint.h"
#include "util/FrameShell.h"
#include "FullSystem/HessianBlocks.h"

namespace HSLAM
{

    void projectMatchesToRef(std::shared_ptr<Frame> currFrame, SE3 refFramePose, std::vector<Vec10f> & dataOut)
    {
        releaseVec(dataOut);
        dataOut.reserve(currFrame->nFeatures);
        std::vector<double> idHs;
        idHs.reserve(currFrame->nFeatures);

        for (int i = 0; i < currFrame->nFeatures; ++i)
        {
            auto Mp = currFrame->tMapPoints[i]; //global map points that matched to current frame
            if (!Mp)
                continue;


        SE3 leftToLeft_0 = refFramePose * Mp->sourceFrame->fs->getPose() ;
	    auto R = (leftToLeft_0.rotationMatrix()).cast<float>();
	    auto t = (leftToLeft_0.translation()).cast<float>();
        
        auto KliP = Vec3f(
            ((float)Mp->pt[0] - currFrame->HCalib->cxl()) * currFrame->HCalib->fxli(),
            ((float)Mp->pt[1] - currFrame->HCalib->cyl()) * currFrame->HCalib->fyli(),
            1);

        float scale = Mp->sourceFrame->fs->getPoseOptiInv().scale();
        float idepth = Mp->getidepth() / scale;
        Vec3f ptp = R * KliP + t * idepth;
        float drescale = 1.0 / ptp[2];
        float new_idepth = idepth * drescale;
        std::cout << drescale << std::endl;
        if (!(drescale > 0))
            continue;

        float u = ptp[0] * drescale;
        float v = ptp[1] * drescale;
        float Ku = u * currFrame->HCalib->fxl() + currFrame->HCalib->cxl();
        float Kv = v * currFrame->HCalib->fyl() + currFrame->HCalib->cyl();





        // Vec3f PtinRef = refFramePose.cast<float>() * Mp->getWorldPose(); //Project the global map points to reference frame
        // float u = PtinRef[0];                                            // PtinRef[2];
        // float v = PtinRef[1];                                            // / PtinRef[2];
        // // float u = (currFrame->HCalib->fxl() * PtinRef[0] / PtinRef[2]) + currFrame->HCalib->cxl();// +0.5;
        // // float v = (currFrame->HCalib->fyl() * PtinRef[1] / PtinRef[2]) + currFrame->HCalib->cyl(); // + 0.5;
        // float depth = PtinRef[2];
        float idH = sqrtf(Mp->getidepthHessian());

        auto matchedLoc = currFrame->mvKeys[i].pt;
        Vec10f Out;
        Out << matchedLoc.x, matchedLoc.y, Ku, Kv, new_idepth, idH, NAN, NAN, NAN, NAN; //last 3 elements are populated by KRes (u,v, id) proejcted to currEstimate
        dataOut.push_back(Out);
        idHs.push_back(idH);
        }
        //normalize weights
        float idHsNormalizer = (float)getStdDev(idHs);
        for (int i = 0, iend = dataOut.size(); i < iend; ++i)
        {
            dataOut[i][5] = dataOut[i][5] / (idHsNormalizer + 0.0001);
        }
    }

Vec3 calcRes(CalibHessian* HCalib, SE3& refToNew, std::vector<Vec10f>& vMPs, std::vector<Vec2f>& _Residuals, float cutoffTH )
{
    releaseVec(_Residuals);
    int n = vMPs.size();
    _Residuals.reserve(n);
    int numTermsInE = 0;
    int numSaturated = 0;

    float maxEnergy = 2*setting_huberTH_Ind*cutoffTH-setting_huberTH_Ind*setting_huberTH_Ind;	// energy for r=setting_coarseCutoffTH.

    Mat33f RKi = (refToNew.rotationMatrix().cast<float>() *  HCalib->getInvCalibMatrix() ); //project from ref to current given latest estimate
    Vec3f t = (refToNew.translation()).cast<float>();
    
    float E = 0;
    
    for (int i = 0, iend = vMPs.size(); i < iend; ++i)
    {
        Vec3f Ptin (vMPs[i][2], vMPs[i][3], vMPs[i][4]); //(u,v, idepth) in refFrame
        Vec3f pt = RKi * Vec3f(Ptin[0], Ptin[1], 1) + t*Ptin[2];
		float u = pt[0] / pt[2];
		float v = pt[1] / pt[2];
		float Ku = HCalib->fxl() * u + HCalib->cxl();
		float Kv = HCalib->fyl() * v + HCalib->cyl();
		float new_idepth = Ptin[2]/pt[2];

        // Vec3f pt = refToNew.cast<float>() * Ptin; //project point to currFrame with currEstimate
        // float u = pt[0] / pt[2];
        // float v = pt[1] / pt[2];

        // float Ku = HCalib->fxl() * u + HCalib->cxl();
        // float Kv = HCalib->fyl() * v + HCalib->cyl();
        // float new_idepth = 1.0 / pt[2];

        vMPs[i][6] = u;
        vMPs[i][7] = v;
        vMPs[i][8] = new_idepth;

        Vec2f residual = Vec2f(vMPs[i][0], vMPs[i][1]) - Vec2f(Ku, Kv);
        float error = residual.dot(residual);
        // if (error > cutoffTH)
        // {
		// 	E += maxEnergy;
		// 	numTermsInE++;
		// 	numSaturated++;
        //     vMPs[i][9] = 0;
        // }
        // else
        {
            float hw = error < setting_huberTH_Ind ? 1 : setting_huberTH_Ind / sqrtf(error);
            _Residuals.push_back(residual);
            vMPs[i][9] = hw;
            E += hw  * error * (2 - hw); // error   (Mat22f::Identity() * MatchesinRef[i][5]) * 
            numTermsInE++;
        }
    }

    float saturatedRatio = numSaturated / (float)numTermsInE;

    return Vec3(E, saturatedRatio, numTermsInE);
}

void calcGSSSE(std::shared_ptr<Frame>frame, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew)
{
    H_out.setZero();
    b_out.setZero();
    
    float fx = frame->HCalib->fxl();
    float fy = frame->HCalib->fyl();
    int residsUsed = 0;
    for (int i = 0, iend = MatchesinRef.size(); i < iend; ++i)
    {
        if(MatchesinRef[i][9] == 0)
            continue;
        residsUsed++;
        Mat28f Jac;
        Jac(0, 0) = fx * MatchesinRef[i][8]; // fx*id
        Jac(0, 1) = 0;
        Jac(0, 2) = - fx * MatchesinRef[i][8] * MatchesinRef[i][6]; // - fx * id * x
        Jac(0, 3) = - fx * MatchesinRef[i][6] * MatchesinRef[i][7]; // - fx * x * y
        Jac(0, 4) =  fx * (1 + (MatchesinRef[i][6] * MatchesinRef[i][6]) ); //fx*(1+x^2)
        Jac(0, 5) = - fx * MatchesinRef[i][7]; // -fx * y
        Jac(0, 6) = 0.0;
        Jac(0, 7) = 0.0;

        Jac(1, 0) = 0;
        Jac(1, 1) = fy * MatchesinRef[i][8]; // fy * id
        Jac(1, 2) = -fy * MatchesinRef[i][8] * MatchesinRef[i][7]; // - fy * id * y
        Jac(1, 3) = - fy * (1+ (MatchesinRef[i][7]* MatchesinRef[i][7])); // -fy * (1+y^2)
        Jac(1, 4) = fy * MatchesinRef[i][6] * MatchesinRef[i][7]; // fy*u*v
        Jac(1, 5) = fy * MatchesinRef[i][6]; // u * fy
        Jac(1, 6) = 0.0;
        Jac(1, 7) = 0.0;
        Jac = -Jac;
        Mat22f IndWeights = Mat22f::Identity()  * MatchesinRef[i][9]; // W = diag(hw * NormalizedIdepthHessian)   * MatchesinRef[i][5]
        H_out += (Jac.transpose() * (IndWeights * Jac)).cast<double>(); //J^T W J
        b_out += (Jac.transpose() * IndWeights * Residuals[i]).cast<double>();
    }

   
	H_out = H_out * (1.0f/(float)residsUsed);
	b_out = b_out * (1.0f/(float)residsUsed);

	H_out.block<8,3>(0,0) *= SCALE_XI_ROT;
	H_out.block<8,3>(0,3) *= SCALE_XI_TRANS;
	H_out.block<8,1>(0,6) *= SCALE_A;
	H_out.block<8,1>(0,7) *= SCALE_B;
	H_out.block<3,8>(0,0) *= SCALE_XI_ROT;
	H_out.block<3,8>(3,0) *= SCALE_XI_TRANS;
	H_out.block<1,8>(6,0) *= SCALE_A;
	H_out.block<1,8>(7,0) *= SCALE_B;
	b_out.segment<3>(0) *= SCALE_XI_ROT;
	b_out.segment<3>(3) *= SCALE_XI_TRANS;
	b_out.segment<1>(6) *= SCALE_A;
	b_out.segment<1>(7) *= SCALE_B;
}


bool trackNewestCoarse(std::shared_ptr<Frame> newFrame, std::shared_ptr<Frame> refFrame, SE3 & lastToNew_out, int coarsestLvl)
{
    projectMatchesToRef(newFrame, refFrame->fs->getPoseInverse(), MatchesinRef);

    int maxIterations[] = {10, 20, 50, 50, 50};
    float lambdaExtrapolationLimit = 0.001;

    SE3 refToNew_current = lastToNew_out;
    float setting_IndCutoffTH = 20.0;
    for (int lvl = coarsestLvl; lvl >= 0; lvl--)
    {
        Mat88 H;
        Vec8 b;
        float IndCutoffRepeat=1.0f;
        
        Vec3 resOld = calcRes(newFrame->HCalib, refToNew_current, MatchesinRef, Residuals, setting_IndCutoffTH*IndCutoffRepeat);
        
		// while(resOld[1] > 0.6 && IndCutoffRepeat < 50)
		// {
		// 	IndCutoffRepeat*=2;
        //     resOld = calcRes(newFrame->HCalib, refToNew_current, MatchesinRef, Residuals, setting_IndCutoffTH*IndCutoffRepeat);
        
		// }

        calcGSSSE(newFrame, H, b, refToNew_current);

        float lambda = 0.01;

        for (int iteration = 0; iteration < maxIterations[lvl]; iteration++)
        {
            Mat88 Hl = H;
            for (int i = 0; i < 8; i++)
                Hl(i, i) *= (1 + lambda);
            
            //Vec8 inc = Hl.ldlt().solve(-b);
            Vec8 inc = Vec8::Zero();
            inc.head<6>() = Hl.topLeftCorner<6, 6>().ldlt().solve(-b.head<6>());
            inc.tail<2>().setZero();

            float extrapFac = 1;
            if (lambda < lambdaExtrapolationLimit)
                extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));
            inc *= extrapFac;

            Vec8 incScaled = inc;
            incScaled.segment<3>(0) *= SCALE_XI_ROT;
            incScaled.segment<3>(3) *= SCALE_XI_TRANS;
            incScaled.segment<1>(6) *= SCALE_A;
            incScaled.segment<1>(7) *= SCALE_B;

            if (!std::isfinite(incScaled.sum()))
                incScaled.setZero();

            SE3 refToNew_new = SE3::exp((Vec6)(incScaled.head<6>())) * refToNew_current;

            Vec3 resNew = calcRes(newFrame->HCalib, refToNew_new, MatchesinRef, Residuals, setting_IndCutoffTH*IndCutoffRepeat);

            bool accept = (resNew[0]/resNew[2] < resOld[0]/resOld[2]); //nbre of residuals is equal in both; I should include a cutoff to remove large outliers... then these would be normalized by their respective number of measurements

            std::cout << resNew[0] / resNew[2] << std::endl;
            if (accept)
            {
                calcGSSSE(newFrame, H, b, refToNew_new);
                resOld = resNew;
                refToNew_current = refToNew_new;
                lambda *= 0.5;
            }
            else
            {
                lambda *= 4;
                if (lambda < lambdaExtrapolationLimit)
                    lambda = lambdaExtrapolationLimit;
            }

            if (!(inc.norm() > 1e-3))
            {
                // printf("inc too small, break!\n");
                break;
            }
        }
        std::cout << std::endl;
    }

    // set!
    lastToNew_out = refToNew_current;

    return true;
}
}