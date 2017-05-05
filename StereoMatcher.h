#ifndef DISPARITYTESTS_STEREOMATCHER_H
#define DISPARITYTESTS_STEREOMATCHER_H

#include <opencv2/opencv.hpp>
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <string>
#include <fstream>

class StereoMatcher {

protected:
    // allocate all required gpu memory
    cv::cuda::GpuMat gpu_frame_1, gpu_frame_2, gpu_disp_map, gpu_colored_disp;
    // allocate all required memory
    cv::Mat frame_1, frame_2;  // init frames
    cv::Mat disp_map, colored_disp_map; // init results
    int frame_w, frame_h, ndisp;
    double fps;
    std::map<std::string, std::string> configValues;

public:
    // Constructor
    StereoMatcher(int fr_w, int fr_h, int num_disp, double framerate)
            :frame_w(fr_w), frame_h(fr_h), ndisp(num_disp), fps(framerate){};
    virtual ~StereoMatcher(){}; // deconstructor

    virtual void getStereoObj()=0;
    virtual void calcDisparityMap()=0;

    void setFrames(cv::Mat frame1, cv::Mat frame2);
    void preprocessFrame(cv::Mat& frame);
    cv::Mat getDispMap();
    cv::Mat getColoredDispMap();
    void readConfig();
};


class BM : public StereoMatcher
{
    cv::Ptr<cv::cuda::StereoBM> stereo;
public:
    // Constructor
    BM(int fr_w, int fr_h, int num_disp, double framerate) : StereoMatcher(fr_w, fr_h, num_disp, framerate){};
    void getStereoObj();
    void calcDisparityMap();
};

class SGBM : public StereoMatcher
{
    cv::Ptr<cv::StereoSGBM> stereo;
public:
    // Constructor
    SGBM(int fr_w, int fr_h, int num_disp, double framerate) : StereoMatcher(fr_w, fr_h, num_disp, framerate){};

    void getStereoObj();
    void calcDisparityMap();
};

class BP : public StereoMatcher
{
    cv::Ptr<cv::cuda::StereoBeliefPropagation> stereo;
public:
    // Constructor
    BP(int fr_w, int fr_h, int num_disp, double framerate) : StereoMatcher(fr_w, fr_h, num_disp, framerate){};

    void getStereoObj();
    void calcDisparityMap();
};

class CSBP : public StereoMatcher
{
    cv::Ptr<cv::cuda::StereoConstantSpaceBP> stereo;
public:
    // Constructor
    CSBP(int fr_w, int fr_h, int num_disp, double framerate) : StereoMatcher(fr_w, fr_h, num_disp, framerate){};

    void getStereoObj();
    void calcDisparityMap();
};

#endif //DISPARITYTESTS_STEREOMATCHER_H
