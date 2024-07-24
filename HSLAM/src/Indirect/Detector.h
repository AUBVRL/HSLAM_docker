#pragma once

#include <opencv2/core.hpp>
#include "util/NumType.h"


namespace HSLAM
{

template <typename Type> class IndexThreadReduce;
class PixelSelector;
class FeatureDetector
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    FeatureDetector();
    ~FeatureDetector();
    // void ComputeThreeMaxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3);
    // void ExtractFeatures(cv::Mat &Image, FeatureType FeatType, std::vector<Vec3f*>&DirPyr, int id, std::vector<float*>& GradPyr, std::vector<cv::KeyPoint> &mvKeys, cv::Mat &Descriptors, int &NumFeatures, int NumFeaturesToExtract, shared_ptr<IndexThreadReduce<Vec10>> ThreadPool);
    void ExtractFeatures(cv::Mat &Image, cv::Mat &Occupancy ,std::vector<cv::KeyPoint> &mvKeys, cv::Mat &Descriptors, int &NumFeatures, int NumFeaturesToExtract);
   
private:
    std::vector<int> InitUmax();
    std::vector<cv::Point> InitPattern();
    void computeOrbDescriptor(const cv::Mat &Orig, const cv::Mat &img, std::vector<cv::KeyPoint> *Keys, cv::Mat &Descriptors_, int min, int max);
    float IC_Angle(const cv::Mat &image, cv::Point2f pt, const std::vector<int> &u_max);
    std::vector<cv::KeyPoint> Ssc(std::vector<cv::KeyPoint> keyPoints, int numRetPoints, int minDist, float tolerance, int cols, int rows);
    
    float ShiTomasiScore(std::vector<Vec3f*>&DirPyr, cv::Size imgSize, const float &u, const float &v, int halfbox = 4);

    std::vector<int> umax;
    cv::Point2f minIndDistPt;
    cv::Point2f minDirDistPt;

    int HALF_PATCH_SIZE;
    int PATCH_SIZE;
    int EDGE_THRESHOLD;
    std::vector<cv::Point> pattern;
    std::shared_ptr<PixelSelector> PixSelector;
    float* selectionMap;
   
    cv::Mat ImageBlurred;
    std::shared_ptr<IndexThreadReduce<Vec10>> ThreadPool;
};

} // namespace SLAM

