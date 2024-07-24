#pragma once
#include "util/NumType.h"
#include <iostream>

#if _WIN32 || _WIN64
#if _WIN64
#define Env64
#else
#define Env32
#endif
#endif

// Check GCC
#if __GNUC__
#if __x86_64__ || __ppc64__
#define Env64
#else
#define Env32
#endif
#endif

namespace HSLAM
{
    class MapPoint;
    class Frame;
    class FrameShell;

    class Matcher
    {
    private:
        
        inline float RadiusByViewingCos(const float &viewCos)
        {
            if (viewCos > 0.998)
                return 2.5;
            else
                return 4.0;
        }

        inline void ComputeThreeMaxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3)
        {
            int max1=0;
            int max2=0;
            int max3=0;

            for(int i=0; i<L; i++)
            {
                const int s = histo[i].size();
                if(s>max1)
                {
                    max3=max2; max2=max1; max1=s;
                    ind3=ind2; ind2=ind1; ind1=i;
                }
                else if(s>max2)
                {
                    max3 = max2; max2 = s; ind3=ind2; ind2=i;
                }
                else if(s>max3)
                {
                    max3=s; ind3=i;
                }
            }

            if(max2<0.1f*(float)max1)
            {
                ind2=-1; 
                ind3=-1;
            }
            else if(max3<0.1f*(float)max1)
            {
                ind3=-1;
            }
        }

    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        Matcher(){}
        ~Matcher() {}

        int SearchByBoWTracking(std::shared_ptr<Frame> pKF, std::shared_ptr<Frame> F, float nnRatio, bool mbCheckOrientation, std::vector<std::shared_ptr<MapPoint>> &vpMapPointMatches);
        int SearchByBow(std::shared_ptr<Frame> frame1, std::shared_ptr<Frame> frame2, float nnRatio, bool mbCheckOrientation, std::vector<std::shared_ptr<MapPoint>> &matches);

        int SearchBySim3(std::shared_ptr<Frame> pKF1, std::shared_ptr<Frame> pKF2, std::vector<std::shared_ptr<MapPoint>> &vpMatches12, const float &s12, const Mat33f &R12, const Vec3f &t12, const float th);
        int SearchBySim3Projection(std::shared_ptr<Frame> pKF, Sim3 Scw, const std::vector<std::shared_ptr<MapPoint>> &vpPoints, std::vector<std::shared_ptr<MapPoint>> &vpMatched, int th);


        int searchWithEpipolar(std::shared_ptr<Frame> pKF1, std::shared_ptr<Frame> pKF2, std::vector<std::pair<size_t, size_t>> &vMatchedPairs, bool mbCheckOrientation = true);
        int SearchByProjection(std::shared_ptr<Frame> &CurrentFrame, std::shared_ptr<Frame> &pKF, const std::set<std::shared_ptr<MapPoint>> &sAlreadyFound, const float th, const int ORBdist, bool mbCheckOrientation = true);
        bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, Mat33f &F12);

        
        int SearchLocalMapByProjection(std::shared_ptr<Frame> F, std::vector<std::shared_ptr<MapPoint>> &vpMapPoints, float th, float nnratio);
        int SearchByProjectionFrameToFrame(std::shared_ptr<Frame> CurrentFrame, const std::shared_ptr<Frame> LastFrame, const float th, bool mbCheckOrientation = true);

        int Fuse(std::shared_ptr<Frame> pKF, Sim3 Scw, const std::vector<std::shared_ptr<MapPoint>> &vpPoints, float th, std::vector<std::shared_ptr<MapPoint>> &vpReplacePoint);
        int Fuse(std::shared_ptr<Frame> pKF, const std::vector<std::shared_ptr<MapPoint>> &vpMapPoints, const float th);

        inline static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
        {
            int dist = 0;
#ifdef __SSE2__

#ifdef Env64
            
            const uint64_t* pa = a.ptr<uint64_t>();
            const uint64_t *pb = b.ptr<uint64_t>();

            for (int i = 0; i < 4; i++, pa++, pb++)
            {
                uint64_t v = *pa ^ *pb;
                dist += _mm_popcnt_u64(v);
            }

            
            return dist;
#else
            const int *pa = a.ptr<int32_t>();
            const int *pb = b.ptr<int32_t>();
            for (int i = 0; i < 8; i++, pa++, pb++)
            {
                unsigned int v = *pa ^ *pb; 
                dist += _mm_popcnt_u32(v);
            }

            return dist;
#endif

#else
            const int *pa = a.ptr<int32_t>();
            const int *pb = b.ptr<int32_t>();

            for (int i = 0; i < 8; i++, pa++, pb++)
            {
                unsigned int v = *pa ^ *pb;
                v = v - ((v >> 1) & 0x55555555);
                v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
                dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
            }
            return dist;

#endif
        }

        static const int TH_LOW;
        static const int TH_HIGH;
        static const int HISTO_LENGTH;
    };
}
