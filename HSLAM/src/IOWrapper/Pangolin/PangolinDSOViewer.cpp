#include "PangolinDSOViewer.h"
#include "KeyFrameDisplay.h"

#include "util/settings.h"
#include "util/globalCalib.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/ImmaturePoint.h"
#include "OverwritePangolin.h"

#include "Indirect/Map.h"
#include "Indirect/MapPoint.h"
#include "Indirect/Frame.h"
#include "util/memUsage.h"

namespace HSLAM
{
namespace IOWrap
{
using namespace pangolin;


PangolinDSOViewer::PangolinDSOViewer(int w, int h, bool startRunThread)
{
	this->w = w;
	this->h = h;
	running=true;

	// {
	// 	boost::unique_lock<boost::mutex> lk(openImagesMutex);
	// 	// internalVideoImg = new MinimalImageB3(w,h);
	// 	// internalKFImg = new MinimalImageB3(w,h);
	// 	// //internalResImg = new MinimalImageB3(w,h);
	// 	// videoImgChanged = kfImgChanged; //=resImgChanged=true;

	// 	// internalVideoImg->setBlack();
	// 	// internalKFImg->setBlack();
	// 	// //internalResImg->setBlack();
	// }


	{
		currentCam = new KeyFrameDisplay();
	}

	needReset = false;

	nGlobalPoints = 0;
	bufferValid = false;
	ngoodPoints = 0;

	pangolin::CreateWindowAndBind(main_window_name,1920,1080);
	const int UI_WIDTH = 270; //180

	glEnable(GL_DEPTH_TEST);

	// 3D visualization
	scene_cam = OpenGlRenderState(ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000), ModelViewLookAt(-0, -0.1, -5, 0, 0.1, 0, 0.0, -100.0, 0.0)); 
	// (
	// 	pangolin::ProjectionMatrix(w,h,400,400,w/2,h/2,0.1,1000),
	// 	pangolin::ModelViewLookAt(-0,-5,-10, 0,0,0, pangolin::AxisNegY)
	// 	);
    
	display_cam = &CreateDisplay().SetBounds(0.0, 1.0, 0.0, 1.0, -w / (float)h).SetHandler(new Handler3D(scene_cam));

	// pangolin::View& Visualization3D_display = pangolin::CreateDisplay()
	// 	.SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -w/(float)h)
	// 	.SetHandler(new pangolin::Handler3D(scene_cam));

	

	// setup image displays
	FrameImage = std::unique_ptr<InternalImage>(new InternalImage());
	FeatureFrame = &Display("FeatureFrame").SetAspect(w/(float)h);
    DepthKfImage = std::unique_ptr<InternalImage>(new InternalImage());
	DepthKeyFrame = &Display("DepthKeyFrame").SetAspect(w/(float)h);

    pangolin::CreateDisplay()
		  .SetBounds(0.0, 0.2, pangolin::Attach::Pix(0.0), 1.0)
		  .SetLayout(pangolin::LayoutEqual)
		  .AddDisplay(*DepthKeyFrame)
		  .AddDisplay(*FeatureFrame)
		  .SetHandler(new pangolin::HandlerResize());

	// parameter reconfigure gui
    Nopanel = &CreateNewPanel("noui").SetBounds(1.0, Attach::ReversePix(35), 0.0, Attach::Pix(UI_WIDTH));
	panel = &CreateNewPanel("ui").SetBounds(1.0, Attach::ReversePix(2000), 0.0, Attach::Pix(UI_WIDTH));
	fpsPanel = &CreateNewPanel("fps").SetBounds(1.0, Attach::ReversePix(70), Attach::ReversePix(400), Attach::ReversePix(0)).SetLayout(pangolin::Layout::LayoutVertical);
	IndStats = &CreateNewPanel("indStat").SetBounds(Attach::Pix(15), Attach::Pix(110), Attach::ReversePix(250), 1.0).SetLayout(pangolin::Layout::LayoutEqualVertical);
	// pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));
	ShowPanel = new Var<bool>("noui.Show Settings", false, false);
    HidePanel = new Var<bool>("ui.Hide Settings", false, false);
	
	settings_pointCloudMode = new Var<int> ("ui.PC_mode",1,0,2,false);
	bFollow = new Var<bool>("ui.Follow Camera", true, true);
	settings_showKFCameras = new Var<bool>("ui.KFCam",false,true);
	settings_showCurrentCamera = new Var<bool> ("ui.CurrCam",true,true);
	settings_showTrajectory = new pangolin::Var<bool> ("ui.Trajectory",true,true);
	settings_showFullTrajectory = new pangolin::Var<bool> ("ui.FullTrajectory",false,true);
	settings_showActiveConstraints = new pangolin::Var<bool> ("ui.ActiveConst",false,true);
	settings_showAllConstraints = new pangolin::Var<bool> ("ui.AllConst",false,true);
	settings_drawIndCov = new pangolin::Var<bool> ("ui.IndCov",false,true);
	settings_drawIndMap = new pangolin::Var<bool> ("ui.IndMap",true,true);
	settings_drawExtractedFeats = new pangolin::Var<bool> ("ui.Extracted Features",false,true);
	settings_drawFrameMatches = new pangolin::Var<bool> ("ui.Map Matches",true,true);
	settings_drawMatchRays = new pangolin::Var<bool> ("ui.Match Rays",false,true);
	settings_drawObservations = new pangolin::Var<bool> ("ui.Draw Observations",false,true);
	
	setting_render_display3D = new Var<bool> ("ui.show3D",true,true);
	setting_render_displayDepth = new Var<bool> ("ui.showDepth",true,true);
	setting_render_displayVideo = new Var<bool> ("ui.showVideo",true,true);

	settings_showFramesWindow = new pangolin::Var<bool> ("ui.showFramesWindow", false, true);
	settings_showFullTracking = new pangolin::Var<bool> ("ui.showFullTracking",false,true);
	settings_showCoarseTracking = new pangolin::Var<bool> ("ui.showCoarseTracking",false,true);


	settings_sparsity = new pangolin::Var<int> ("ui.sparsity",1,1,20,false);
	settings_scaledVarTH = new pangolin::Var<double> ("ui.relVarTH",0.001,1e-10,1e10, true);
	settings_absVarTH = new pangolin::Var<double> ("ui.absVarTH",0.001,1e-10,1e10, true);
	settings_minRelBS = new pangolin::Var<double> ("ui.minRelativeBS",0.1,0,1, false);


	settings_resetButton = new pangolin::Var<bool> ("ui.Reset",false,false);
    _Pause = new Var<bool>(Pause?"ui.Resume!Pause" :"ui.Pause!Resume", false, false);
    RecordScreen = new Var<bool>("ui.Record Screen!Stop Recording", false, false);

	settings_KfIdToDraw = new pangolin::Var<int> ("ui.KfAgeDisp", 0, 0, 1400, false);


	settings_nPts = new pangolin::Var<int> ("ui.activePoints",setting_desiredPointDensity, 50,5000, false);
	settings_nCandidates = new pangolin::Var<int> ("ui.pointCandidates",setting_desiredImmatureDensity, 50,5000, false);
	settings_nMaxFrames = new pangolin::Var<int> ("ui.maxFrames",setting_maxFrames, 4,10, false);
	settings_kfFrequency = new pangolin::Var<double> ("ui.kfFrequency",setting_kfGlobalWeight,0.1,3, false);
	settings_gradHistAdd = new pangolin::Var<double> ("ui.minGradAdd",setting_minGradHistAdd,0,15, false);

	settings_trackFps = new pangolin::Var<double> ("fps.Track fps",0,0,0,false);
	settings_mapFps = new pangolin::Var<double> ("fps.KF fps",0,0,0,false);
	memUse = new pangolin::Var<double> ("fps.MemoryUse(MB)",0,0,0,false);

	Mps = new pangolin::Var<int>("indStat.Mps",0,0,0,false);
	Kfs = new pangolin::Var<int>("indStat.Kfs",0,0,0,false);
	nMatches = new pangolin::Var<int>("indStat.nMatches",0,0,0,false);

	panel->Show(false); Nopanel->Show(true);
	fpsPanel->Show(true);
	if (startRunThread)
	{
        runThread = boost::thread(&PangolinDSOViewer::run, this);
		GetBoundWindow()->RemoveCurrent(); //detach rendering loop from main thread
	}
}

PangolinDSOViewer::~PangolinDSOViewer()
{
	close();
	runThread.join();
}


void PangolinDSOViewer::run()
{
	if (runThread.joinable())
		BindToContext(main_window_name);

	glEnable(GL_DEPTH_TEST);
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	
	// glEnable(GL_POINT_SMOOTH);
	// glEnable(GL_BLEND);

	// Default hooks for exiting (Esc) and fullscreen (tab).
	while( !pangolin::ShouldQuit() && running )
	{
		// Clear entire screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

		if (setting_render_display3D->Get())
		{
			// Activate efficiently by object
			display_cam->Activate(scene_cam);
			boost::unique_lock<boost::mutex> lk3d(model3DMutex);
			//pangolin::glDrawColouredCube();
			int refreshed=0;
			for(KeyFrameDisplay* fh : keyframes)
			{
				if(fh->originFrame->KfId < settings_KfIdToDraw->Get())
					continue;
				float blue[3] = {0, 0, 1};
				float orange[3] = {0.8, 0.4, 0.0};
				
				bool overWriteRefresh = false;
				if (fh->originFrame->doesNeedRefresh())
				{
					fh->camToWorld = fh->originFrame->getPoseOptiInv();
					fh->needRefresh = true;
					overWriteRefresh = true;
					fh->originFrame->setRefresh(false);
				}
				if (settings_showKFCameras->Get())
					fh->drawCam(1, fh->originFrame->frame ? blue : orange, 0.1, false);

				refreshed += (int)(fh->refreshPC((refreshed < 10 || overWriteRefresh), settings_scaledVarTH->Get(), settings_absVarTH->Get(),
						settings_pointCloudMode->Get(), settings_minRelBS->Get(), settings_sparsity->Get()));
				fh->drawPC(1);
			}

			if(settings_showCurrentCamera->Get()) currentCam->drawCam(2,0,0.2);
			drawConstraints();
			DrawIndirectMap(settings_drawIndCov->Get());
			lk3d.unlock();
		}

		// update fps counters
		{
			boost::unique_lock<boost::mutex> lk(openImagesMutex);
			float sd=0;
			for(float d : lastNMappingMs) sd+=d;
			*settings_mapFps=lastNMappingMs.size()*1000.0f / sd;
			int offsetVOcabSize = 0;
			if (!Vocab.empty())
				offsetVOcabSize = 439.2; //the BovW model used is about 450 MB when loaded
			*memUse = (getCurrentRSS() / 1048576) - offsetVOcabSize;
		}
		{
			model3DMutex.lock();
			float sd=0;
			for(float d : lastNTrackingMs) sd+=d;
			*settings_trackFps = lastNTrackingMs.size()*1000.0f / sd;
			model3DMutex.unlock();
		}


		if(setting_render_displayVideo->Get())
			renderInternalFrame(FrameImage, FeatureFrame);
		
		if(setting_render_displayDepth->Get())
			renderInternalFrame(DepthKfImage, DepthKeyFrame);
		

	    // update parameters
		setting_render_renderWindowFrames = settings_showFramesWindow->Get();
		setting_render_plotTrackingFull = settings_showFullTracking->Get();
		setting_render_displayCoarseTrackingFull = settings_showCoarseTracking->Get();

	    setting_desiredPointDensity = settings_nPts->Get();
	    setting_desiredImmatureDensity = settings_nCandidates->Get();
	    setting_maxFrames = settings_nMaxFrames->Get();
	    setting_kfGlobalWeight = settings_kfFrequency->Get();
	    setting_minGradHistAdd = settings_gradHistAdd->Get();

		if (Pushed(*_Pause)) {Pause = !Pause;}
    	if (Pushed(*ShowPanel)) {Nopanel->Show(false); panel->Show(true);}
		if (Pushed(*HidePanel)) {Nopanel->Show(true); panel->Show(false); }
		if (Pushed(*RecordScreen))
        	DisplayBase().RecordOnRender("ffmpeg:[fps=15,bps=90388608,flip=true,unique_filename]//screencap.avi"); //8388608 45388608

		if (Pushed(*settings_resetButton)) 
	    {
	    	printf("RESET!\n");
			settings_resetButton->Reset();
			setting_fullResetRequested = true;
	    }

		// Swap frames and Process Events
		pangolin::FinishFrame();


	    if(needReset) reset_internal();

		usleep(5000);
	}

	isDead = true;
	Pause = false;

}


void PangolinDSOViewer::close()
{
	running = false;
}

void PangolinDSOViewer::join()
{
	runThread.join();
	printf("JOINED Pangolin thread!\n");
}

void PangolinDSOViewer::reset()
{
	needReset = true;
}

void PangolinDSOViewer::reset_internal()
{
	model3DMutex.lock();
    scene_cam.SetModelViewMatrix(ModelViewLookAt(-0, -0.1, -5, 0, 0.1, 0, 0.0, -100.0, 0.0));

	for(size_t i=0; i<keyframes.size();i++) delete keyframes[i];
	keyframes.clear();
	allFramePoses.clear();
	keyframesByKFID.clear();
	connections.clear();
	currentCam->width = currentCam->height = 0; //prevent current camera from getting drawn (causes crash at reset)
	model3DMutex.unlock();

	//openImagesMutex.lock();
	boost::unique_lock<boost::mutex> lk(openImagesMutex);
	FrameImage.reset(); FrameImage = std::unique_ptr<InternalImage>(new InternalImage());
    DepthKfImage.reset(); DepthKfImage = std::unique_ptr<InternalImage>(new InternalImage());
	lk.unlock();

	globalmap.reset();
	nGlobalPoints = 0;
	ngoodPoints = 0;
	IndcolorBuffer.Free();
	IndvertexBuffer.Free();
	bufferValid = false;	

	needReset = false;
}


void PangolinDSOViewer::drawConstraints()
{
	if(settings_showAllConstraints->Get())
	{
		// draw constraints
		glLineWidth(1);
		glColor3f(0,1,0);
		glBegin(GL_LINES);
		for(unsigned int i=0;i<connections.size();i++)
		{
			if(connections[i].to == 0 || connections[i].from==0) continue;
			int nAct = connections[i].bwdAct + connections[i].fwdAct;
			int nMarg = connections[i].bwdMarg + connections[i].fwdMarg;
			if(nAct==0 && nMarg>0  )
			{
				Sophus::Vector3f t = connections[i].from->camToWorld.translation().cast<float>();
				glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
				t = connections[i].to->camToWorld.translation().cast<float>();
				glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
			}
		}
		glEnd();
	}

	if(settings_showActiveConstraints->Get())
	{
		glLineWidth(3);
		glColor3f(0,0,1);
		glBegin(GL_LINES);
		for(unsigned int i=0;i<connections.size();i++)
		{
			if(connections[i].to == 0 || connections[i].from==0) continue;
			int nAct = connections[i].bwdAct + connections[i].fwdAct;

			if(nAct>0)
			{
				Sophus::Vector3f t = connections[i].from->camToWorld.translation().cast<float>();
				glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
				t = connections[i].to->camToWorld.translation().cast<float>();
				glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
			}
		}
		glEnd();
	}

	if(settings_showTrajectory->Get())
	{
		float colorRed[3] = {1,0,0};
		glColor3f(colorRed[0],colorRed[1],colorRed[2]);
		glLineWidth(3);

		glBegin(GL_LINE_STRIP);
		for(unsigned int i=0;i<keyframes.size();i++)
		{
			glVertex3f((float)keyframes[i]->camToWorld.translation()[0],
					(float)keyframes[i]->camToWorld.translation()[1],
					(float)keyframes[i]->camToWorld.translation()[2]);
		}
		glEnd();
	}

	if(settings_showFullTrajectory->Get())
	{
		float colorGreen[3] = {0,1,0};
		glColor3f(colorGreen[0],colorGreen[1],colorGreen[2]);
		glLineWidth(3);

		glBegin(GL_LINE_STRIP);
		for(unsigned int i=0;i<allFramePoses.size();i++)
		{
			glVertex3f((float)allFramePoses[i][0],
					(float)allFramePoses[i][1],
					(float)allFramePoses[i][2]);
		}
		glEnd();
	}
}

void PangolinDSOViewer::DrawIndirectMap(bool bDrawGraph)
{
	if (!globalmap)
		return;

	*Mps = globalmap->MapPointsInMap();
	*Kfs = globalmap->KeyFramesInMap();

	if (settings_drawMatchRays->Get())
	{

		GLfloat lineWidth = 2.0;
		glLineWidth(lineWidth);
		glColor4f(1.0f, 1.0f, 0.0f, 1.0f); //light yellow
		glBegin(GL_LINES);

		for (int i = 0, iend = vCurrMatches.size(); i < iend; ++i)
			if (vCurrMatches[i])
			{
				Vec3f Ow = currentCam->originFrame->getCameraCenter().cast<float>(); //image->shell->getCameraCenter().cast<float>();
				auto Mp = vCurrMatches[i]->getWorldPose();
				glVertex3f(Ow(0), Ow(1), Ow(2));
				glVertex3f(Mp(0), Mp(1), Mp(2));
			}
		glEnd();
	}

	if (settings_drawIndMap->Get())
	{
		// // control update frequency
		// static int needUpdate = 0;
		// needUpdate = needUpdate + 1;

		// if (needUpdate >= 10)
		// {
		// 	needUpdate = 0;
		std::vector<std::shared_ptr<MapPoint>> vpMPs;
		globalmap->GetAllMapPoints(vpMPs);
		auto vpRefMPs = globalmap->GetReferenceMapPoints();

		if (vpMPs.empty())
			return;

		std::set<std::shared_ptr<MapPoint>> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());
		Vec3f *tmpIndirectBuffer = new Vec3f[vpMPs.size()];
		Vec3b *tmpIndirectColorBuffer = new Vec3b[vpMPs.size()];

		// glPointSize(mPointSize);
		// glBegin(GL_POINTS);
		// glColor3f(0.0,0.0,0.0);
		Vec3b blue(0, 0, 255);
		Vec3b red(255, 0, 0);

		double scaledVarThresh = settings_scaledVarTH->Get();
		double absVarTH = settings_absVarTH->Get();
		ngoodPoints = 0;

		if(settings_drawObservations->Get())
		{
			glLineWidth(1);
			glBegin(GL_LINES);
			glColor3f(0,1,0);
		}

		for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
		{
			if (vpMPs[i]->isBad())
				continue;

			if(!vpMPs[i]->checkVar())
				continue;
			
			if(settings_drawObservations->Get())
			{
				auto observations = vpMPs[i]->GetObservations();
				for (auto it: observations)
				{
					Vec3f Ow = it.first->fs->getCameraCenter().cast<float>();
					auto MpPose = vpMPs[i]->getWorldPose();
					glVertex3f(Ow(0), Ow(1), Ow(2));
					glVertex3f(MpPose(0), MpPose(1), MpPose(2));
				}
			}

			tmpIndirectBuffer[ngoodPoints] = vpMPs[i]->getWorldPose();
			if (spRefMPs.count(vpMPs[i]))
				tmpIndirectColorBuffer[ngoodPoints] = red;
			else
				tmpIndirectColorBuffer[ngoodPoints] = blue;
			ngoodPoints = ngoodPoints + 1;
		}

		if(settings_drawObservations->Get())
			glEnd();

		if (ngoodPoints > nGlobalPoints)
		{
			nGlobalPoints = ngoodPoints * 1.3;
			IndvertexBuffer.Reinitialise(pangolin::GlArrayBuffer, nGlobalPoints, GL_FLOAT, 3, GL_DYNAMIC_DRAW);
			IndcolorBuffer.Reinitialise(pangolin::GlArrayBuffer, nGlobalPoints, GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW);
		}
		if (ngoodPoints <= 0)
			return;

		IndvertexBuffer.Upload(tmpIndirectBuffer, sizeof(float) * 3 * ngoodPoints, 0);
		IndcolorBuffer.Upload(tmpIndirectColorBuffer, sizeof(unsigned char) * 3 * ngoodPoints, 0);
		bufferValid = true;
		delete[] tmpIndirectBuffer;
		delete[] tmpIndirectColorBuffer;
		// }

		if (!bufferValid)
			return;

		GLfloat mPointSize = 5;

		glPointSize(mPointSize);
		IndcolorBuffer.Bind();
		glColorPointer(IndcolorBuffer.count_per_element, IndcolorBuffer.datatype, 0, 0);
		glEnableClientState(GL_COLOR_ARRAY);

		IndvertexBuffer.Bind();
		glVertexPointer(IndvertexBuffer.count_per_element, IndvertexBuffer.datatype, 0, 0);
		glEnableClientState(GL_VERTEX_ARRAY);
		glDrawArrays(GL_POINTS, 0, ngoodPoints);
		glDisableClientState(GL_VERTEX_ARRAY);
		IndvertexBuffer.Unbind();

		glDisableClientState(GL_COLOR_ARRAY);
		IndcolorBuffer.Unbind();
	}

	if (bDrawGraph)
	{
		GLfloat mGraphLineWidth = 2.0;
		glLineWidth(mGraphLineWidth);

		glColor4f(0.8f, 0.8f, 0.0f, 0.7f); //light yellow
		glBegin(GL_LINES);

		std::vector<std::shared_ptr<Frame>> vpKFs;
		globalmap->GetAllKeyFrames(vpKFs);

		for (size_t i = 0; i < vpKFs.size(); i++)
		{
			// Covisibility Graph
			const std::vector<std::shared_ptr<Frame>> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
			Vec3f Ow = vpKFs[i]->fs->getCameraCenter().cast<float>();
			if (!vCovKFs.empty())
			{
				for (std::vector<std::shared_ptr<Frame>>::const_iterator vit = vCovKFs.begin(), vend = vCovKFs.end(); vit != vend; vit++)
				{
					if ((*vit)->fs->KfId < vpKFs[i]->fs->KfId)
						continue;
					Vec3f Ow2 = (*vit)->fs->getCameraCenter().cast<float>();
					glVertex3f(Ow(0), Ow(1), Ow(2));
					glVertex3f(Ow2(0), Ow2(1), Ow2(2));
				}
			}

			// Spanning tree
			std::shared_ptr<Frame> pParent = vpKFs[i]->GetParent();
			if (pParent)
			{
				Vec3f Owp = pParent->fs->getCameraCenter().cast<float>();
				glVertex3f(Ow(0), Ow(1), Ow(2));
				glVertex3f(Owp(0), Owp(1), Owp(2));
			}

			// Loops
			std::set<std::shared_ptr<Frame>> sLoopKFs = vpKFs[i]->GetLoopEdges();
			for (std::set<std::shared_ptr<Frame>>::iterator sit = sLoopKFs.begin(), send = sLoopKFs.end(); sit != send; sit++)
			{
				if ((*sit)->fs->KfId < vpKFs[i]->fs->KfId)
					continue;
				Vec3f Owl = (*sit)->fs->getCameraCenter().cast<float>();
				glVertex3f(Ow(0), Ow(1), Ow(2));
				glVertex3f(Owl(0), Owl(1), Owl(2));
			}
		}

		glEnd();
	}
}





void PangolinDSOViewer::publishGraph(const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity)
{
    if(!setting_render_display3D->Get()) return;
    if(disableAllDisplay) return;

	model3DMutex.lock();
    connections.resize(connectivity.size());
	int runningID=0;
	int totalActFwd=0, totalActBwd=0, totalMargFwd=0, totalMargBwd=0;
    for(std::pair<uint64_t,Eigen::Vector2i> p : connectivity)
	{
		int host = (int)(p.first >> 32);
        int target = (int)(p.first & (uint64_t)0xFFFFFFFF);

		assert(host >= 0 && target >= 0);
		if(host == target)
		{
			assert(p.second[0] == 0 && p.second[1] == 0);
			continue;
		}

		if(host > target) continue;

		connections[runningID].from = keyframesByKFID.count(host) == 0 ? 0 : keyframesByKFID[host];
		connections[runningID].to = keyframesByKFID.count(target) == 0 ? 0 : keyframesByKFID[target];
		connections[runningID].fwdAct = p.second[0];
		connections[runningID].fwdMarg = p.second[1];
		totalActFwd += p.second[0];
		totalMargFwd += p.second[1];

        uint64_t inverseKey = (((uint64_t)target) << 32) + ((uint64_t)host);
		
		if(connectivity.find(inverseKey) == connectivity.end())
			continue;

		Eigen::Vector2i st = connectivity.at(inverseKey);
		connections[runningID].bwdAct = st[0];
		connections[runningID].bwdMarg = st[1];

		totalActBwd += st[0];
		totalMargBwd += st[1];

		runningID++;
	}


	model3DMutex.unlock();
}
void PangolinDSOViewer::publishKeyframes(
		std::vector<FrameHessian*> &frames,
		bool final,
		CalibHessian* HCalib)
{
	if(!setting_render_display3D->Get()) return;
    if(disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(model3DMutex);
	for(FrameHessian* fh : frames)
	{
		if(keyframesByKFID.find(fh->frameID) == keyframesByKFID.end())
		{
			KeyFrameDisplay* kfd = new KeyFrameDisplay();
			keyframesByKFID[fh->frameID] = kfd;
			keyframes.push_back(kfd);
		}
		keyframesByKFID[fh->frameID]->setFromKF(fh, HCalib);
	}
}
void PangolinDSOViewer::publishCamPose(FrameShell* frame,
		CalibHessian* HCalib)
{
    if(!setting_render_display3D->Get()) return;
    if(disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(model3DMutex);
	struct timeval time_now;
	gettimeofday(&time_now, NULL);
	lastNTrackingMs.push_back(((time_now.tv_sec-last_track.tv_sec)*1000.0f + (time_now.tv_usec-last_track.tv_usec)/1000.0f));
	if(lastNTrackingMs.size() > 10) lastNTrackingMs.pop_front();
	last_track = time_now;

	if(!setting_render_display3D->Get()) return;

	currentCam->setFromF(frame, HCalib);
	allFramePoses.push_back(currentCam->camToWorld.translation().cast<float>());
}


void PangolinDSOViewer::publishGlobalMap(std::shared_ptr<Map> _globalMap)
{
	globalmap = _globalMap;
}

void PangolinDSOViewer::pushLiveFrame(FrameHessian* image, int nIndmatches)
{
	
    if(disableAllDisplay) return;
	vCurrMatches = image->shell->frame->tMapPoints;
	vCurrKeys = image->shell->frame->mvKeys;
	*nMatches = nIndmatches;

	if (!(setting_render_displayVideo->Get()))
		return;
	setInternalImageData(FrameImage, image);
}

bool PangolinDSOViewer::needPushDepthImage()
{
    return setting_render_displayDepth->Get();
}
void PangolinDSOViewer::pushDepthImage(MinimalImageB3* image)
{

    if(!setting_render_displayDepth->Get()) return;
    if(disableAllDisplay) return;

	setInternalImageData(DepthKfImage, image->data);
	
	boost::unique_lock<boost::mutex> lk(openImagesMutex);
	struct timeval time_now;
	gettimeofday(&time_now, NULL);
	lastNMappingMs.push_back(((time_now.tv_sec-last_map.tv_sec)*1000.0f + (time_now.tv_usec-last_map.tv_usec)/1000.0f));
	if(lastNMappingMs.size() > 10) lastNMappingMs.pop_front();
	last_map = time_now;

}

void PangolinDSOViewer::setInternalImageData(std::unique_ptr<InternalImage> &InternalImage, Vec3b* Img)
{
	boost::unique_lock<boost::mutex> lk(openImagesMutex);

	if (InternalImage->Width == 0 || InternalImage->Height == 0)
    {
		InternalImage->Image = new uchar [w * h * 3];
        InternalImage->Width = w;
        InternalImage->Height = h;
    }
	memcpy(InternalImage->Image, Img, w*h*3*sizeof(uchar));
    InternalImage->HaveNewImage = true;
}

void PangolinDSOViewer::setInternalImageData(std::unique_ptr<InternalImage> &InternalImage, FrameHessian* image)// Vec3f* Img)
{
	
	boost::unique_lock<boost::mutex> lk(openImagesMutex);

	if (InternalImage->Width == 0 || InternalImage->Height == 0)
    {
		InternalImage->Image = new uchar [w * h * 3];
		InternalImage->Width = w;
		InternalImage->Height = h;
    }

	for(int i=0, j=0;i<w*h*3;i+=3, ++j)
		InternalImage->Image[i] =
		InternalImage->Image[i+1] =
		InternalImage->Image[i+2] =
			image->dI[j][0]*0.8 > 255.0f ? 255 :  image->dI[j][0]*0.8;

	int radiusMp = 2;
	int radiusPt = 1;

	if (settings_drawExtractedFeats->Get() || settings_drawFrameMatches->Get())
		for (int i = 0; i < image->shell->frame->nFeatures; ++i)
		{
			bool mapPoint = image->shell->frame->tMapPoints[i] ? true : false;

			if (mapPoint && settings_drawFrameMatches->Get())
			{
				for (int j = -radiusMp; j <= radiusMp; ++j)
					for (int k = -radiusMp; k <= radiusMp; ++k)
					{
						cv::Point2f Pt = image->shell->frame->mvKeys[i].pt + cv::Point2f(j, k);
						int index = (Pt.x + Pt.y * w) * 3;
						InternalImage->Image[index + 1] = 255;
						InternalImage->Image[index] = InternalImage->Image[index + 2] = 0;
					}
			}
			else if(settings_drawExtractedFeats->Get())
			{
				for (int j = -radiusPt; j <= radiusPt; ++j)
					for (int k = -radiusPt; k <= radiusPt; ++k)
					{
						cv::Point2f Pt = image->shell->frame->mvKeys[i].pt + cv::Point2f(j, k);
						int index = (Pt.x + Pt.y * w) * 3;
						InternalImage->Image[index + 2] = 255;
						InternalImage->Image[index] = InternalImage->Image[index + 1] = 0;
					}
			}
		}
		InternalImage->HaveNewImage = true;
}


void PangolinDSOViewer::renderInternalFrame(std::unique_ptr<InternalImage> &ImageToRender, View* CanvasFrame)
{
	
    if (!ImageToRender->IsTextureGood)
    {        
        if (ImageToRender->Width + ImageToRender->Height != 0)
        {
            CanvasFrame->SetAspect((float)ImageToRender->Width / (float)ImageToRender->Height);
            ImageToRender->FeatureFrameTexture.Reinitialise(ImageToRender->Width, ImageToRender->Height, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
            ImageToRender->IsTextureGood = true;
        }
    }

    if (ImageToRender->IsTextureGood)
    {
		boost::unique_lock<boost::mutex> lk(openImagesMutex);
		if (ImageToRender->HaveNewImage)
			ImageToRender->FeatureFrameTexture.Upload(&ImageToRender->Image[0], GL_BGR, GL_UNSIGNED_BYTE);
        CanvasFrame->Activate();
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        ImageToRender->FeatureFrameTexture.RenderToViewportFlipY();
    }
}

}
}
