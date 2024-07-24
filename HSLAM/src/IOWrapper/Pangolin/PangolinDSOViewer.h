#pragma once
#include <pangolin/pangolin.h>
#include "boost/thread.hpp"
#include "util/MinimalImage.h"
#include "IOWrapper/Output3DWrapper.h"
#include <map>
#include <deque>

static const std::string main_window_name = "HSLAM";

namespace HSLAM
{

class FrameHessian;
class CalibHessian;
class FrameShell;

class Map;
class MapPoint;

struct InternalImage
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	InternalImage() {}
	~InternalImage()
	{
		if (Image != nullptr)
		{
			delete Image;
			Image = nullptr;
		}
	}
	pangolin::GlTexture FeatureFrameTexture;
	bool IsTextureGood = false;
	bool HaveNewImage = false;
	unsigned char *Image = nullptr;
	int Width = 0;
	int Height = 0;
};

namespace IOWrap
{

class KeyFrameDisplay;

struct GraphConnection
{
	KeyFrameDisplay* from;
	KeyFrameDisplay* to;
	int fwdMarg, bwdMarg, fwdAct, bwdAct;
};


class PangolinDSOViewer : public Output3DWrapper
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    PangolinDSOViewer(int w, int h, bool startRunThread=true);
	virtual ~PangolinDSOViewer();

	void run();
	void close();


	// ==================== Output3DWrapper Functionality ======================
    virtual void publishGraph(const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity) override;
    virtual void publishKeyframes( std::vector<FrameHessian*> &frames, bool final, CalibHessian* HCalib) override;
    virtual void publishCamPose(FrameShell* frame, CalibHessian* HCalib) override;

	virtual void publishGlobalMap(std::shared_ptr<Map> _globalMap) override;

	virtual void pushLiveFrame(FrameHessian* image, int nIndmatches) override;
    virtual void pushDepthImage(MinimalImageB3* image) override;
    virtual bool needPushDepthImage() override;

    virtual void join() override;

    virtual void reset() override;

	bool isDead = false;
	

private:
	bool needReset;
	void reset_internal();
	void drawConstraints();

	boost::thread runThread;
	bool running;
	int w,h;

	pangolin::OpenGlRenderState scene_cam;
	pangolin::View *display_cam;

	// images rendering
	boost::mutex openImagesMutex;
	std::unique_ptr<InternalImage> FrameImage;
	pangolin::View *FeatureFrame;

	std::unique_ptr<InternalImage> DepthKfImage;
	pangolin::View *DepthKeyFrame;
	void renderInternalFrame(std::unique_ptr<InternalImage> &ImageToRender, pangolin::View* CanvasFrame);
    void setInternalImageData(std::unique_ptr<InternalImage> &InternalImage, Vec3b* Img);
    void setInternalImageData(std::unique_ptr<InternalImage> &InternalImage, FrameHessian* image);

	// MinimalImageB3* internalVideoImg;
	// MinimalImageB3* internalKFImg;
	// bool videoImgChanged, kfImgChanged;

	// 3D model rendering
	boost::mutex model3DMutex;
	KeyFrameDisplay* currentCam;
	std::vector<KeyFrameDisplay*> keyframes;
	std::vector<Vec3f,Eigen::aligned_allocator<Vec3f>> allFramePoses;
	std::map<int, KeyFrameDisplay*> keyframesByKFID;
	std::vector<GraphConnection,Eigen::aligned_allocator<GraphConnection>> connections;

	// render settings
	pangolin::View *panel;
	pangolin::View *Nopanel;
	pangolin::View *fpsPanel;
	pangolin::View *IndStats;

	pangolin::Var<bool> *ShowPanel;
	pangolin::Var<bool> *HidePanel;
	pangolin::Var<bool> *settings_showKFCameras;
	pangolin::Var<bool> *settings_showCurrentCamera;
	pangolin::Var<bool> *settings_showTrajectory;
	pangolin::Var<bool> *settings_showFullTrajectory;
	pangolin::Var<bool> *settings_showActiveConstraints;
	pangolin::Var<bool> *settings_showAllConstraints;
	pangolin::Var<bool> *settings_drawFeatureMatches;
	pangolin::Var<bool> *settings_drawIndMap;
	pangolin::Var<bool> *settings_drawExtractedFeats;
	pangolin::Var<bool> *settings_drawFrameMatches;
	pangolin::Var<bool> *settings_drawMatchRays;
	pangolin::Var<bool> *settings_drawObservations;



	pangolin::Var<bool> *settings_drawIndCov;

	pangolin::Var<bool> * setting_render_displayDepth;
	pangolin::Var<bool> * setting_render_displayVideo;
	pangolin::Var<bool> * setting_render_display3D;
    pangolin::Var<bool> *RecordScreen;
	pangolin::Var<bool> *_Pause;
	pangolin::Var<bool> *bFollow;
	pangolin::Var<int> *settings_pointCloudMode;
	pangolin::Var<double> *settings_scaledVarTH;
	pangolin::Var<double> *settings_absVarTH;
	pangolin::Var<double> *settings_minRelBS;
	pangolin::Var<int> *settings_sparsity;

	pangolin::Var<bool> *settings_resetButton;

	pangolin::Var<int> *settings_KfIdToDraw;


	pangolin::Var<int> *settings_nPts;
	pangolin::Var<int> *settings_nCandidates;
	pangolin::Var<int> *settings_nMaxFrames;
	pangolin::Var<double> *settings_kfFrequency;
	pangolin::Var<double> *settings_gradHistAdd;

	pangolin::Var<double> *settings_trackFps;
	pangolin::Var<double> *settings_mapFps;
	pangolin::Var<double> *memUse;

	pangolin::Var<int> *Mps;
	pangolin::Var<int> *Kfs;
	pangolin::Var<int> *nMatches;


	pangolin::Var<bool> *settings_showFramesWindow;
	pangolin::Var<bool> *settings_showFullTracking;
	pangolin::Var<bool> *settings_showCoarseTracking;

	// timings
	struct timeval last_track;
	struct timeval last_map;


	std::deque<float> lastNTrackingMs;
	std::deque<float> lastNMappingMs;

	void DrawIndirectMap(bool bDrawGraph = false);
	std::shared_ptr<Map> globalmap;
	pangolin::GlBuffer IndvertexBuffer;
	pangolin::GlBuffer IndcolorBuffer;
	int ngoodPoints;
	int nGlobalPoints;
	bool bufferValid;

	std::vector<std::shared_ptr<MapPoint>> vCurrMatches;
	std::vector<cv::KeyPoint> vCurrKeys;
};
}



}
