#pragma once
#define MAX_ACTIVE_FRAMES 100

#include <deque>
#include "util/NumType.h"
#include "util/globalCalib.h"
#include "vector"
 
#include <iostream>
#include <fstream>
#include "util/NumType.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include "util/IndexThreadReduce.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "FullSystem/PixelSelector2.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <math.h>


namespace HSLAM
{
namespace IOWrap
{
class Output3DWrapper;
}

class PixelSelector;
class PCSyntheticPoint;
class CoarseTracker;
struct FrameHessian;
struct PointHessian;
class CoarseInitializer;
struct ImmaturePointTemporaryResidual;
class ImageAndExposure;
class CoarseDistanceMap;

class FeatureDetector;

class EnergyFunctional;

class Map;
class Matcher;
class LoopCloser;

template<typename T> inline void deleteOut(std::vector<T*> &v, const int i)
{
	delete v[i];
	v[i] = v.back();
	v.pop_back();
}
template<typename T> inline void deleteOutPt(std::vector<T*> &v, const T* i)
{
	delete i;

	for(unsigned int k=0;k<v.size();k++)
		if(v[k] == i)
		{
			v[k] = v.back();
			v.pop_back();
		}
}
template<typename T> inline void deleteOutOrder(std::vector<T*> &v, const int i)
{
	delete v[i];
	for(unsigned int k=i+1; k<v.size();k++)
		v[k-1] = v[k];
	v.pop_back();
}
template<typename T> inline void deleteOutOrder(std::vector<T*> &v, const T* element)
{
	int i=-1;
	for(unsigned int k=0; k<v.size();k++)
	{
		if(v[k] == element)
		{
			i=k;
			break;
		}
	}
	assert(i!=-1);

	for(unsigned int k=i+1; k<v.size();k++)
		v[k-1] = v[k];
	v.pop_back();

	delete element;
	element = nullptr;
}

inline bool eigenTestNan(const MatXX &m, std::string msg)
{
	bool foundNan = false;
	for(int y=0;y<m.rows();y++)
		for(int x=0;x<m.cols();x++)
		{
			if(!std::isfinite((double)m(y,x))) foundNan = true;
		}

	if(foundNan)
	{
		printf("NAN in %s:\n",msg.c_str());
		std::cout << m << "\n\n";
	}


	return foundNan;
}





class FullSystem
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	FullSystem();
	virtual ~FullSystem();

	// adds a new frame, and creates point & residual structs.
	void addActiveFrame(ImageAndExposure* image, int id);

	// marginalizes a frame. drops / marginalizes points & residuals.
	void marginalizeFrame(FrameHessian* frame);
	void blockUntilMappingIsFinished();

	float optimize(int mnumOptIts);

	void printResult(std::string file, bool printSim = false);
	void saveMap(std::string file, bool printSim = false);

	std::vector<SE3> getPath();
	std::vector<Eigen::Vector3f> getMap();


	void debugPlot(std::string name);

	void printFrameLifetimes();
	// contains pointers to active frames

    std::vector<IOWrap::Output3DWrapper*> outputWrapper;

	bool isLost;
	bool initFailed;
	bool initialized;
	bool linearizeOperation;


	void setGammaFunction(float* BInv);
	void setOriginalCalib(const VecXf &originalCalib, int originalW, int originalH);

	boost::mutex mapMutex;

	CalibHessian Hcalib;
	std::shared_ptr<Matcher> matcher;
	std::shared_ptr<Map> globalMap;
	std::vector<FrameHessian*> frameHessians;	// ONLY changed in marginalizeFrame and addFrame.
	EnergyFunctional* ef;
	std::vector<FrameShell*> allKeyFramesHistory;
	boost::mutex trackMutex;
	void BAatExit();


private:

	// opt single point
	int optimizePoint(PointHessian* point, int minObs, bool flagOOB);
	PointHessian* optimizeImmaturePoint(ImmaturePoint* point, int minObs, ImmaturePointTemporaryResidual* residuals);

	double linAllPointSinle(PointHessian* point, float outlierTHSlack, bool plot);

	// mainPipelineFunctions
	Vec5 trackNewCoarse(FrameHessian *fh, bool writePose = false);
	void traceNewCoarse(FrameHessian* fh);
	void activatePoints();
	void activatePointsMT();
	void activatePointsOldFirst();
	void flagPointsForRemoval();
	void makeNewTraces(FrameHessian* newFrame, float* gtDepth);
	void initializeFromInitializer(FrameHessian* newFrame);
	void flagFramesForMarginalization(FrameHessian* newFH);


	void removeOutliers();


	// set precalc values.
	void setPrecalcValues();


	// solce. eventually migrate to ef.
	void solveSystem(int iteration, double lambda);
	Vec3 linearizeAll(bool fixLinearization);
	bool doStepFromBackup(float stepfacC,float stepfacT,float stepfacR,float stepfacA,float stepfacD);
	void backupState(bool backupLastStep);
	void loadSateBackup();
	double calcLEnergy();
	double calcMEnergy();
	void linearizeAll_Reductor(bool fixLinearization, std::vector<PointFrameResidual*>* toRemove, int min, int max, Vec10* stats, int tid);
	void activatePointsMT_Reductor(std::vector<PointHessian*>* optimized,std::vector<ImmaturePoint*>* toOptimize,int min, int max, Vec10* stats, int tid);
	void applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid);

	void printOptRes(const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a, float b);

	void debugPlotTracking();

	std::vector<VecX> getNullspaces(
			std::vector<VecX> &nullspaces_pose,
			std::vector<VecX> &nullspaces_scale,
			std::vector<VecX> &nullspaces_affA,
			std::vector<VecX> &nullspaces_affB);

	void setNewFrameEnergyTH();


	void printLogLine();
	void printEvalLine();
	void printEigenValLine();
	std::ofstream* calibLog;
	std::ofstream* numsLog;
	std::ofstream* errorsLog;
	std::ofstream* eigenAllLog;
	std::ofstream* eigenPLog;
	std::ofstream* eigenALog;
	std::ofstream* DiagonalLog;
	std::ofstream* variancesLog;
	std::ofstream* nullspacesLog;

	std::ofstream* coarseTrackingLog;

	// statistics
	long int statistics_lastNumOptIts;
	long int statistics_numDroppedPoints;
	long int statistics_numActivatedPoints;
	long int statistics_numCreatedPoints;
	long int statistics_numForceDroppedResBwd;
	long int statistics_numForceDroppedResFwd;
	long int statistics_numMargResFwd;
	long int statistics_numMargResBwd;
	float statistics_lastFineTrackRMSE;







	// =================== changed by tracker-thread. protected by trackMutex ============
	std::vector<FrameShell*> allFrameHistory;
	CoarseInitializer* coarseInitializer;
	Vec5 lastCoarseRMSE;


	// ================== changed by mapper-thread. protected by mapMutex ===============
	

	IndexThreadReduce<Vec10> treadReduce;

	float* selectionMap;
	PixelSelector* pixelSelector;
	CoarseDistanceMap* coarseDistanceMap;


	std::vector<PointFrameResidual*> activeResiduals;
	float currentMinActDist;


	std::vector<float> allResVec;



	// mutex etc. for tracker exchange.
	boost::mutex coarseTrackerSwapMutex;			// if tracker sees that there is a new reference, tracker locks [coarseTrackerSwapMutex] and swaps the two.
	CoarseTracker* coarseTracker_forNewKF;			// set as as reference. protected by [coarseTrackerSwapMutex].
	CoarseTracker* coarseTracker;					// always used to track new frames. protected by [trackMutex].
	float minIdJetVisTracker, maxIdJetVisTracker;
	float minIdJetVisDebug, maxIdJetVisDebug;





	// mutex for camToWorl's in shells (these are always in a good configuration).
	boost::mutex shellPoseMutex;



/*
 * tracking always uses the newest KF as reference.
 *
 */

	void makeKeyFrame( FrameHessian* fh);
	void makeNonKeyFrame( FrameHessian* fh);
	void deliverTrackedFrame(FrameHessian* fh, bool needKF);
	void mappingLoop();

	// tracking / mapping synchronization. All protected by [trackMapSyncMutex].
	boost::mutex trackMapSyncMutex;
	boost::condition_variable trackedFrameSignal;
	boost::condition_variable mappedFrameSignal;
	std::deque<FrameHessian*> unmappedTrackedFrames;
	int needNewKFAfter;	// Otherwise, a new KF is *needed that has ID bigger than [needNewKFAfter]*.
	boost::thread mappingThread;
	bool runMapping;
	bool needToKetchupMapping;

	int lastRefStopID;



	void IndirectMapper(std::shared_ptr<Frame> frame);
	bool TrackLocalMap(std::shared_ptr<Frame> frame);

	void updateLocalKeyframes(std::shared_ptr<Frame> frame);
	void updateLocalKeyframesOld(std::shared_ptr<Frame> frame);
	void updateLocalPoints(std::shared_ptr<Frame> frame);
	void updateLocalPointsOld(std::shared_ptr<Frame> frame);
	void CheckReplacedInLastFrame();
	int SearchLocalPoints(std::shared_ptr<Frame> frame, int th = 1, float nnratio = 0.8);
	int updatePoseOptimizationData(std::shared_ptr<Frame> frame, int & nmatches, bool istrackingLastFrame = true);
	void SearchInNeighbors(std::shared_ptr<Frame> currKF);
	void KeyFrameCulling(std::shared_ptr<Frame> currKF);
	// SE3 cumulativeForm();


	std::shared_ptr<FeatureDetector> detector;

	size_t mnLastKeyFrameId;
	std::shared_ptr<Frame> mpLastKeyFrame;
	std::vector<std::shared_ptr<MapPoint>> mvpLocalMapPoints;
	std::vector<std::shared_ptr<Frame>> mvpLocalKeyFrames;
	std::shared_ptr<Frame> mpReferenceKF;
	std::shared_ptr<Frame> mLastFrame;

	SE3 Velocity;
	FixedQueue<SE3, 4> vVelocity;
	boost::mutex localMapMtx;

	int nIndmatches;
	bool isUsable;

	std::shared_ptr<LoopCloser> loopCloser;

	//sort localKeyframes while updating localkeyframes from best to worst:
	static bool cmpAscending(std::pair<std::shared_ptr<Frame>, int> &a, std::pair<std::shared_ptr<Frame>, int> &b)
	{
		return a.second < b.second;
	}
	static bool cmpDescending(std::pair<std::shared_ptr<Frame>, int> &a, std::pair<std::shared_ptr<Frame>, int> &b)
	{
		return a.second > b.second;
	}
	static std::vector<std::pair<std::shared_ptr<Frame>, int>> sortLocalKFs(std::map<std::shared_ptr<Frame>, int> &M, bool descendingOrder = false)
	{
		std::vector<std::pair<std::shared_ptr<Frame>, int>> A;
		for (auto &it : M)
		{
			A.push_back(it);
		}
		if(descendingOrder)
			std::sort(A.begin(), A.end(), cmpDescending);
		else
			std::sort(A.begin(), A.end(), cmpAscending);
		return A;
	}
};
}

