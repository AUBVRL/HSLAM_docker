#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "IOWrapper/Output3DWrapper.h"

#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/DatasetReader.h"
#include "util/globalCalib.h"

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"


#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"


using namespace HSLAM;

void my_exit_handler(int s)
{
	printf("Caught signal %d\n",s);
	exit(1);
}

void exitThread()
{
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_exit_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);

	while(true) pause();
}

const cv::String keys =
	"{help h usage ?  |      | print this message}"
	"{files           |<none>| Input images path - mandatory input}"
	"{calib           |<none>| Camera intrinsic callibration - mandatory input}"
	"{vocabPath       |      | Vocabulary path}"
	"{vignette        |      | Path to Vignette model}"
	"{gamma           |      | Path to gamma response Model}"
	"{LoopClosure     | False| Enable-Disable loop closure}"
	"{reverse         | False| Play a sequence in reverse}"
	"{preload         | False| preload all images into memory}"
	"{useSampleOutput | False| replace pangolinViewer with another output wrapper}"
	"{nolog           | False| disable logging optimization data}"
	"{nogui           | False| disable GUI}"
	"{save            | False| save debug images}"
	"{quiet           | True | disable message printing }"
	"{nomt            | False| when set to true it turns off multiThreading}"
	"{startIndex      | 0    | Image index to start from}"
	"{endIndex        |100000| Last image to be processed }"
	"{mode            | 0    | system mode: 0: use precalibrated gamma and vignette -1: photometric mode without calibration - 2: photometric mode with perfect images}"
	"{preset          | 0    | preset configuration}"
	"{speed           | 0.0  | Enforce playback Speed to real-time}";


int main(int argc, char **argv)
{	
	printf("MAIN is called from HSLAM original"); //debugNA
	boost::thread exThread = boost::thread(exitThread); // hook crtl+C.

	//Parse the arguments given.
	cv::CommandLineParser parser(argc, argv, keys);
	std::string vignette = parser.get<std::string>("vignette");
	std::string gammaCalib = parser.get<std::string>("gamma");
	std::string source = parser.get<std::string>("files");
	std::string calib = parser.get<std::string>("calib");
	std::string vocabPath = parser.get<std::string>("vocabPath");
	LoopClosure = parser.get<bool>("LoopClosure");
	bool reverse = parser.get<bool>("reverse");
	bool preload = parser.get<bool>("preload");
	bool useSampleOutput = parser.get<bool>("useSampleOutput");
	setting_debugout_runquiet = parser.get<bool>("quiet");
	setting_logStuff = !parser.get<bool>("nolog");
	disableAllDisplay = parser.get<bool>("nogui");
	debugSaveImages = parser.get<bool>("save");
	multiThreading = !parser.get<bool>("nomt");
	int startIndex = parser.get<int>("startIndex");
	int endIndex = parser.get<int>("endIndex");
	int preset = parser.get<int>("preset");
	int mode =  parser.get<int>("mode");
	float playbackSpeed = parser.get<float>("speed"); // 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
	
	// Print help info and exit
	if (parser.has("help") || parser.has("h") || parser.has("usage") || parser.has("?"))
	{
    	parser.printMessage();
    	return 0;
	}

	//Print error message and exit.
	if(source.empty() || calib.empty()) { std::cout<< "path to images or calibration not provided! cannot function without them. exit." << std::endl; return(0);}

	if (!parser.check()) {parser.printErrors(); return 0; }

	//This is only here to silence the compiler
	if (debugSaveImages)
	{
		if(42==system("rm -rf images_out")) std::cout<<"system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n"<<std::endl;
			if(42==system("mkdir images_out")) std::cout<<"system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n"<<std::endl;
			if(42==system("rm -rf images_out")) std::cout<<"system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n"<<std::endl;
			if(42==system("mkdir images_out")) std::cout<<"system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n"<<std::endl;
	}

	//setup mode parameters
	switch (mode)
	{
	case 1:
		setting_photometricCalibration = 0;
		setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
		setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
		break;
	case 2:
		setting_photometricCalibration = 0;
		setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
		setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
		setting_minGradHistAdd = 3;
		break;
	default:
		break;
	}

	//Load vocab necessary for loop closure. if no vocab path is given loopclosure is disabled.
	if(LoopClosure && !vocabPath.empty())
	{
		Vocab.load(vocabPath.c_str());

		printf("loading Vocabulary from %s!\n", vocabPath.c_str());
		if (Vocab.empty())
		{
			printf("failed to load vocabulary! Exit\n");
			exit(1);
		}
	}
	else
	{
		std::cout << "no vocabulary path provided! disabling loop closure." << std::endl;
		LoopClosure = false; 
	}

	// set playback speed
	if(preset == 0 || preset == 1)
	{
		printf("DEFAULT settings:\n"
				"- %s real-time enforcing\n"
				"- 2000 active points\n"
				"- 5-7 active frames\n"
				"- 1-6 LM iteration each KF\n"
				"- original image resolution\n", preset==0 ? "no " : "1x");

		playbackSpeed = (preset==0 ? 0 : 1);
		// preload = preset==1;
	}
	else if(preset == 2 || preset == 3)
	{
		printf("FAST settings:\n"
				"- %s real-time enforcing\n"
				"- 800 active points\n"
				"- 4-6 active frames\n"
				"- 1-4 LM iteration each KF\n"
				"- 424 x 320 image resolution\n", preset==0 ? "no " : "5x");

		playbackSpeed = (preset==2 ? 0 : 5);
		preload = preset==3;
		setting_desiredImmatureDensity = 600;
		setting_desiredPointDensity = 800;
		setting_minFrames = 4;
		setting_maxFrames = 6;
		setting_maxOptIterations=4;
		setting_minOptIterations=1;

		benchmarkSetting_width = 424;
		benchmarkSetting_height = 320;

		setting_logStuff = false;
	}
	
	//set frame size based on undist_width and height. (from size in calib ) 
	//set geometric calib from calib file (k matrix and size)
	ImageFolderReader* reader = new ImageFolderReader(source, calib, gammaCalib, vignette);
	reader->setGlobalCalibration();
	set_frame_sz(reader->get_undist_width(), reader->get_undist_height());

	// check for error in providing photometric params
	if(setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0)
	{
		printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
		exit(1);
	}


	// set arguments for reverse execution if requested.
	int lstart=startIndex;
	int lend = endIndex;
	int linc = 1;
	if(reverse)
	{
		printf("REVERSE!!!!");
		lstart=endIndex-1;
		if(lstart >= reader->getNumImages())
			lstart = reader->getNumImages()-1;
			
		lend = startIndex;
		linc = -1;
	}


	// Populate the System, photocalib, speed,viewer
	FullSystem* fullSystem = new FullSystem();
	fullSystem->setGammaFunction(reader->getPhotometricGamma());
	fullSystem->linearizeOperation = (playbackSpeed == 0);

	IOWrap::PangolinDSOViewer* viewer = 0;
	if(!disableAllDisplay)
    {
        viewer = new IOWrap::PangolinDSOViewer(wG[0],hG[0], false);
        fullSystem->outputWrapper.push_back(viewer);
    }



    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());


	// to make MacOS happy: run this in dedicated thread -- and use this one to run the GUI.
    std::thread runthread([&]() {
        std::vector<int> idsToPlay;
        std::vector<double> timesToPlayAt;
		//populate timesToPlayAt and idsToPlay vector.
        for(int i=lstart;i>= 0 && i< reader->getNumImages() && linc*i < linc*lend;i+=linc)
        {
            idsToPlay.push_back(i);
            if(timesToPlayAt.size() == 0)
            {
                timesToPlayAt.push_back((double)0);
            }
            else
            {
                double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size()-1]);
                double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size()-2]);
                timesToPlayAt.push_back(timesToPlayAt.back() +  fabs(tsThis-tsPrev)/playbackSpeed);
            }
        }

		//load all images
        std::vector<ImageAndExposure*> preloadedImages;
        if(preload)
        {
            printf("LOADING ALL IMAGES!\n");
            for(int ii=0;ii<(int)idsToPlay.size(); ii++)
            {
                int i = idsToPlay[ii];
                preloadedImages.push_back(reader->getImage(i));
            }
        }

        struct timeval tv_start;
        gettimeofday(&tv_start, NULL);
        clock_t started = clock();
        double sInitializerOffset=0;


        for(int ii=0;ii<(int)idsToPlay.size(); ii++)
        {	
			//printf("IDs to Play size: %li \n", idsToPlay.size());//debugNA
			while (Pause)
			{
				usleep(5000);
			}
				
				
            if(!fullSystem->initialized)	// if not initialized: reset start time.
            {
                gettimeofday(&tv_start, NULL);
                started = clock();
                sInitializerOffset = timesToPlayAt[ii];
            }

            int i = idsToPlay[ii];


            ImageAndExposure* img;
            if(preload)
                img = preloadedImages[ii];
            else
                img = reader->getImage(i);


			//setting frame speed and deciding wether a frame should be skipped or no.
            bool skipFrame=false;
            if(playbackSpeed!=0)
            {
                struct timeval tv_now; gettimeofday(&tv_now, NULL);
                double sSinceStart = sInitializerOffset + ((tv_now.tv_sec-tv_start.tv_sec) + (tv_now.tv_usec-tv_start.tv_usec)/(1000.0f*1000.0f));

                if(sSinceStart < timesToPlayAt[ii])
                    usleep((int)((timesToPlayAt[ii]-sSinceStart)*1000*1000));
                else if(sSinceStart > timesToPlayAt[ii]+0.5+0.1*(ii%2))
                {
                    printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii, timesToPlayAt[ii], sSinceStart);
                    skipFrame=true;
                }
            }

			
			if (!skipFrame) fullSystem->addActiveFrame(img, i);
			

			delete img;

			//exit if pangolin is dead
			if(viewer!=0)
				if(viewer->isDead)
					break;

			//reset if init failed or reset requested
	        if(fullSystem->initFailed || setting_fullResetRequested)
            {
                if(ii < 250 || setting_fullResetRequested)
                {
                    printf("RESETTING!\n");
                    std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
                    for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();
					usleep(20000); //hack - wait for display wrapper to clean up.
					if(fullSystem)
					{
						delete fullSystem;
						fullSystem = nullptr;
					}
						
					fullSystem = new FullSystem();
					fullSystem->setGammaFunction(reader->getPhotometricGamma());
					fullSystem->linearizeOperation = (playbackSpeed == 0);

					fullSystem->outputWrapper = wraps;

                    setting_fullResetRequested=false;
                }
            }

			//exit if lost
            if(fullSystem->isLost)
            {
                printf("LOST!!\n");
                break;
            }

        }
		// fullSystem->BAatExit();
		fullSystem->blockUntilMappingIsFinished();
		clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);


        fullSystem->printResult("result.txt");
        fullSystem->saveMap("map.pcd");


        int numFramesProcessed = abs(idsToPlay[0]-idsToPlay.back());
        double numSecondsProcessed = fabs(reader->getTimestamp(idsToPlay[0])-reader->getTimestamp(idsToPlay.back()));
        double MilliSecondsTakenSingle = 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC);
        double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
        printf("\n======================"
                "\n%d Frames (%.1f fps)"
                "\n%.2fms per frame (single core); "
                "\n%.2fms per frame (multi core); "
                "\n%.3fx (single core); "
                "\n%.3fx (multi core); "
                "\n======================\n\n",
                numFramesProcessed, numFramesProcessed/numSecondsProcessed,
                MilliSecondsTakenSingle/numFramesProcessed,
                MilliSecondsTakenMT / (float)numFramesProcessed,
                1000 / (MilliSecondsTakenSingle/numSecondsProcessed),
                1000 / (MilliSecondsTakenMT / numSecondsProcessed));
        //fullSystem->printFrameLifetimes();
        if(setting_logStuff)
        {
            std::ofstream tmlog;
            tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
            tmlog << 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC*reader->getNumImages()) << " "
                  << ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f) / (float)reader->getNumImages() << "\n";
            tmlog.flush();
            tmlog.close();
        }

    });


	// run the viewer
	if(viewer != 0)
	    viewer->run();

	//add the main thread
	runthread.join();

	for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
	{
		ow->join();
		delete ow;
	}



	printf("DELETE FULLSYSTEM!\n");
	if (fullSystem)
	{
		delete fullSystem;
		fullSystem = nullptr;
	}

	printf("DELETE READER!\n");
	delete reader;

	printf("EXIT NOW!\n");
	return 0;
}
