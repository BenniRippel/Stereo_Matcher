#include "StereoMatcher.h"
#include <iostream>

int main(int argc, char** argv) {

    if (argc != 5) {
        std::cout<<"usage:StereoMatcher.out <Input_Video_1> <Input_Video_2> <Output_Video> <Algorithm>"<< std::endl;
        std::cout<<"<Algorithm> may be BM, SGBM, BP, CSBP, Default is BM..."<< std::endl;
        return -1;
    }
    // create video captures for both input videos
    cv::VideoCapture cap_left(argv[1]);
    cv::VideoCapture cap_right(argv[2]);
    if (!cap_left.isOpened() || !cap_right.isOpened()) {
        std::cout<<"Could not open Video(s)!"<<std::endl;
        return -1;
    }

    // video properties
    int frame_w = cap_left.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_h = cap_left.get(CV_CAP_PROP_FRAME_HEIGHT);
    double fps = cap_left.get(CV_CAP_PROP_FPS);
    int fourcc = CV_FOURCC('D','X','5','0');
    // Videowriter
    cv::VideoWriter output_video(argv[3], fourcc, fps, cv::Size(frame_w,frame_h), false);
    // window for colored disp map
    cv::namedWindow("Colored Disparity Map", CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"

    // init frames
    cv::Mat frame_1, frame_2;
    // init number of disparities
    int ndisp=128;

    // get Stereeo Matcher Object dependend on algorithm input
    std::unique_ptr<StereoMatcher> matcher;
    if (std::string(argv[4])=="SGBM"){matcher.reset(new SGBM(frame_w, frame_h, ndisp, fps));}
    else if (std::string(argv[4])=="BP"){matcher.reset(new BP(frame_w, frame_h, ndisp, fps));}
    else if (std::string(argv[4])=="CSBP"){matcher.reset(new CSBP(frame_w, frame_h, ndisp, fps));}
    else {matcher.reset(new BM(frame_w, frame_h, ndisp, fps));}

    matcher->getStereoObj(); // create Stereo Object

    while (cap_left.isOpened() && cap_right.isOpened())  //iterate over video frames
    {
        cap_left>>frame_1;   // get frames from video capture
        cap_right>>frame_2;
        if (!frame_1.data || !frame_2.data) break;  // break if one frame contains no data

        matcher->setFrames(frame_1, frame_2);
        matcher->calcDisparityMap();
        // write disp map to video
        output_video.write(matcher->getDispMap());

        // display colored disp map
        cv::imshow("Colored Disparity Map", matcher->getColoredDispMap());
        cv::waitKey(1000/fps); //waitKey

    }
    cv::destroyWindow("Colored Disparity Map"); //destroy the window with the name, "MyWindow"




    cap_left.release(); // release video capture
    cap_right.release();

    return 0;
}

