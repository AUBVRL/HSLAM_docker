/**
* This file is part of FLSAM_ROS.
Based on and inspired by DSO project by Jakob Engel

*/

#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>



#include "IOWrapper/Output3DWrapper.h"

#include <boost/thread.hpp>
#include "util/settings.h"
#include "FullSystem/FullSystem.h"
#include "util/Undistort.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"


#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>
#include "cv_bridge/cv_bridge.h"
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace HSLAM;

std::string calib = "";
std::string vignetteFile = "";
std::string gammaFile = "";
std::string saveFile = "";
std::string vocabPath = "";
bool useSampleOutput=false;
int mode = 0;
int preset= 0;

ros::Publisher map_pub ;
ros::Publisher pose_pub ;
ros::Publisher path_pub ;

void parseArgument(char* arg)
{
	int option;
	char buf[1000];
	if(1==sscanf(arg,"savefile=%s",buf))
	{
		saveFile = buf;
		printf("saving to %s on finish!\n", saveFile.c_str());
		return;
	}

	if(1==sscanf(arg,"sampleoutput=%d",&option))
	{
		if(option==1)
		{
			useSampleOutput = true;
			printf("USING SAMPLE OUTPUT WRAPPER!\n");
		}
		return;
	}

	if(1==sscanf(arg,"quiet=%d",&option))
	{
		if(option==1)
		{
			setting_debugout_runquiet = true;
			printf("QUIET MODE, I'll shut up!\n");
		}
		return;
	}


	if(1==sscanf(arg,"nolog=%d",&option))
	{
		if(option==1)
		{
			setting_logStuff = false;
			printf("DISABLE LOGGING!\n");
		}
		return;
	}

	if(1==sscanf(arg,"nogui=%d",&option))
	{
		if(option==1)
		{
			disableAllDisplay = true;
			printf("NO GUI!\n");
		}
		return;
	}
	if(1==sscanf(arg,"nomt=%d",&option))
	{
		if(option==1)
		{
			multiThreading = false;
			printf("NO MultiThreading!\n");
		}
		return;
	}
	if(1==sscanf(arg,"calib=%s",buf))
	{
		calib = buf;
		printf("loading calibration from %s!\n", calib.c_str());
		return;
	}
	if(1==sscanf(arg,"vignette=%s",buf))
	{
		vignetteFile = buf;
		printf("loading vignette from %s!\n", vignetteFile.c_str());
		return;
	}

	if(1==sscanf(arg,"gamma=%s",buf))
	{
		gammaFile = buf;
		printf("loading gammaCalib from %s!\n", gammaFile.c_str());
		return;
	}

	if(1==sscanf(arg,"LoopClosure=%d",&option))
	{
		if(option==1)
		{
			LoopClosure = true;
			printf("hslam_ros :LOOP CLOSURE IS TURNED ON!\n");
		}
		return;
	}

	if(1==sscanf(arg,"vocabPath=%s",buf))
	{
		vocabPath = buf;
		printf("hslam_ros : loading Vocabulary from %s!\n", vocabPath.c_str());
		return;
	}

	if (1==sscanf(arg,"mode=%d",&option))
	{
		if(option==1)
		{
			setting_photometricCalibration = 0;
			setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
			
		}
		if(option==2)
		{
			setting_photometricCalibration = 0;
			setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_minGradHistAdd = 3;

		}
	
	}
	if (1==sscanf(arg,"preset=%d",&option))
	{
		if(option == 0 || option == 1)
		{
			printf("DEFAULT settings:\n"
					"- %s real-time enforcing\n"
					"- 2000 active points\n"
					"- 5-7 active frames\n"
					"- 1-6 LM iteration each KF\n"
					"- original image resolution\n", preset==0 ? "no " : "1x");
		}
		else if(option == 2 || option == 3)
		{
			printf("FAST settings:\n"
					"- %s real-time enforcing\n"
					"- 800 active points\n"
					"- 4-6 active frames\n"
					"- 1-4 LM iteration each KF\n"
					"- 424 x 320 image resolution\n", preset==0 ? "no " : "5x");
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
	}
	if(LoopClosure && !vocabPath.empty())
	{
		Vocab.load(vocabPath.c_str());
		printf("Loop Closure ON and loading Vocabulary from %s!\n", vocabPath.c_str());
		if (Vocab.empty())
		{
			printf("failed to load vocabulary! Exit\n");
			exit(1);
		}
	}

	printf("could not parse argument \"%s\"!!\n", arg);
}


FullSystem* fullSystem = 0;
Undistort* undistorter = 0;
int frameID = 0;



void publishResults() {        
		nav_msgs::Path path;
		geometry_msgs::PoseStamped pose_stamped;
		sensor_msgs::PointCloud2 map;
		pcl::PointCloud<pcl::PointXYZ> cloud;

		std::vector<SE3> points;
		std::vector<Eigen::Vector3f>map_points;

		path.header.frame_id="map";
		pose_stamped.header.frame_id="map";

		points=fullSystem->getPath();
		for (size_t i = 0; i < points.size(); i++)
		{
			pose_stamped.pose.position.x=points[i].translation().transpose().x();
			pose_stamped.pose.position.y=points[i].translation().transpose().y();
			pose_stamped.pose.position.z=points[i].translation().transpose().z();
			pose_stamped.pose.orientation.x=points[i].so3().unit_quaternion().x();
			pose_stamped.pose.orientation.y=points[i].so3().unit_quaternion().y();
			pose_stamped.pose.orientation.z=points[i].so3().unit_quaternion().z();
			pose_stamped.pose.orientation.w=points[i].so3().unit_quaternion().w();

			path.poses.push_back(pose_stamped);
		}

		path_pub.publish(path);
		pose_pub.publish(pose_stamped);

		map_points=fullSystem->getMap();
		for (size_t i = 0; i < map_points.size(); i++)
		{
			pcl::PointXYZ point;
			point.x=map_points[i].x();
			point.y=map_points[i].y();
			point.z=map_points[i].z();
			cloud.push_back(point);
		}
		pcl::toROSMsg(cloud, map);
		map.header.frame_id="map";
		map_pub.publish(map);
    
}




void vidCb(const sensor_msgs::ImageConstPtr img)
{
	cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
	assert(cv_ptr->image.type() == CV_8U);
	assert(cv_ptr->image.channels() == 1);


	if(setting_fullResetRequested)
	{
		std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
		delete fullSystem;
		for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();
		fullSystem = new FullSystem();
		fullSystem->linearizeOperation=false;
		fullSystem->outputWrapper = wraps;
	    if(undistorter->photometricUndist != 0)
	    	fullSystem->setGammaFunction(undistorter->photometricUndist->getG());
		setting_fullResetRequested=false;
	}

	MinimalImageB minImg((int)cv_ptr->image.cols, (int)cv_ptr->image.rows,(unsigned char*)cv_ptr->image.data);
	ImageAndExposure* undistImg = undistorter->undistort<unsigned char>(&minImg, 1,0, 1.0f);
	undistImg->timestamp=img->header.stamp.toSec(); // relay the timestamp to HSLAM
	fullSystem->addActiveFrame(undistImg, frameID);
	frameID++;
	if (frameID>50)
	{
	publishResults();
	}
	
	delete undistImg;

}



//NA: Adding interruption code
bool interrupted = false;
void interruptHandler(int signal)
{
	    interrupted = true;
}



//boost exit handler to exit all threads
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


int main( int argc, char** argv )
{		
	boost::thread exThread = boost::thread(exitThread); // hook crtl+C.
	ros::init(argc, argv, "hslam_live");

	for(int i=1; i<argc;i++) parseArgument(argv[i]);


	setting_desiredImmatureDensity = 1000;
	setting_desiredPointDensity = 1200;
	setting_minFrames = 5;
	setting_maxFrames = 7;
	setting_maxOptIterations=4;
	setting_minOptIterations=1;
	setting_logStuff = false;
	setting_kfGlobalWeight = 1.3;



    undistorter = Undistort::getUndistorterForFile(calib, gammaFile, vignetteFile);

    setGlobalCalib(
            (int)undistorter->getSize()[0],
            (int)undistorter->getSize()[1],
            undistorter->getK().cast<float>());


    fullSystem = new FullSystem();
    fullSystem->linearizeOperation=false;
	
	
	IOWrap::PangolinDSOViewer* viewer = 0;
	if(!disableAllDisplay)
    {
        viewer = new IOWrap::PangolinDSOViewer(
	    		 (int)undistorter->getSize()[0],
	    		 (int)undistorter->getSize()[1]);
        fullSystem->outputWrapper.push_back(viewer);
    }

    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());


    if(undistorter->photometricUndist != 0)
    	fullSystem->setGammaFunction(undistorter->photometricUndist->getG());

    ros::NodeHandle nh;
	//ros::Rate loop_rate(10);
    ros::Subscriber imgSub = nh.subscribe("image", 1, &vidCb);
	map_pub = nh.advertise<sensor_msgs::PointCloud2>("/hslam_map", 10);
	pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/hslam_pose", 10);
	path_pub = nh.advertise<nav_msgs::Path>("/hslam_path", 10);

    //NA: replacing ros_spin with interruptable sequence
    //ros::spin();
	
    signal(SIGINT, interruptHandler);

	while (ros::ok() && !interrupted) //&& frameID <999999 NA
		{	
			//printf("ROS IS OKAY!");
			//printf("FrameID: %d ",frameID);
			ros::spinOnce();
			//loop_rate.sleep();
			if(viewer!=0 && viewer->isDead)
					break;
			
			if(fullSystem->isLost)
            {
                printf("LOST!!\n");
                break;
            }
			
			
			if(fullSystem->initFailed || setting_fullResetRequested)
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
    			fullSystem->setGammaFunction(undistorter->photometricUndist->getG());
				fullSystem->linearizeOperation = false;

				fullSystem->outputWrapper = wraps;

				setting_fullResetRequested=false;
            }
			
			
		}
	fullSystem->blockUntilMappingIsFinished();

	printf("hslam_ros main cpp has been interuppted.\n"); //debug NA
	ros::shutdown();
	ros::waitForShutdown();
	fullSystem->BAatExit();
			
	
	fullSystem->printResult("result.txt"); 
	fullSystem->saveMap("map.txt"); 

	//if(viewer != 0)
	//    viewer->run();
	//Clean-up and exit
    for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
    {
		//printf("DELETE VIEWER IO wrapper\n");
        ow->join();
        delete ow;
    }

	printf("DELETE FULLSYSTEM!\n");
	delete fullSystem;
	printf("DELETE Undistorter\n");
	delete undistorter;
	printf("EXIT NOW\n");
	return 0;
}

