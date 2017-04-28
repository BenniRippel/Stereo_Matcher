#include "StereoMatcher.h"

// Function Definitions
void StereoMatcher::setFrames(cv::Mat frame1, cv::Mat frame2){
    frame_1 = frame1;
    frame_2 = frame2;
}

void StereoMatcher::preprocessFrame(cv::Mat& frame){
    /* Preprocess a frame for stereo matching:
     *  -convert color to gray
     *  -equalize histogram
     *  -gaussian blur
    */
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame, frame);
    cv::GaussianBlur(frame, frame, cv::Size (5,5), 0);
}

cv::Mat StereoMatcher::getDispMap(){
    return disp_map;
}

cv::Mat StereoMatcher::getColoredDispMap(){
    return colored_disp_map;
}

void BM::getStereoObj(){
    int blockSize = 21;
    stereo = cv::cuda::createStereoBM(ndisp, blockSize);
}

void BM::calcDisparityMap(){
    // preprocess frames
    StereoMatcher::preprocessFrame(frame_1);
    StereoMatcher::preprocessFrame(frame_2);

    // upload frames to gpu memory
    gpu_frame_1.upload(frame_1);
    gpu_frame_2.upload(frame_2);
    // compute disparity map using the stereoObj
    stereo->compute(gpu_frame_1, gpu_frame_2, gpu_disp_map);

    // refine Disparity Map with bilateral Filter
    cv::Ptr <cv::cuda::DisparityBilateralFilter> disp_filter = cv::cuda::createDisparityBilateralFilter(ndisp, 5, 1);
    disp_filter->apply(gpu_disp_map, gpu_frame_1, gpu_disp_map);

    // calc colored disparity map
    cv::cuda::drawColorDisp(gpu_disp_map, gpu_colored_disp, ndisp);
    // normalize the disparity map to uint8
    cv::cuda::normalize(gpu_disp_map, gpu_disp_map, 0, 255, cv::NORM_MINMAX, CV_8U);

    // download results from gpu memory
    gpu_disp_map.download(disp_map);
    gpu_colored_disp.download(colored_disp_map);
}

void SGBM::getStereoObj() {
    int blockSize = 11;
    int P1 = 1600;
    int P2 = 4200;
    int disp12MaxDiff = 1;
    int preFilterCap = 10;
    int uniquenessRatio = 5;
    int speckleWindowSize = 100;
    int speckleRange = 32;
    int mode = cv::StereoSGBM::MODE_SGBM;
    stereo = cv::StereoSGBM::create(0, ndisp, blockSize, P1, P2, disp12MaxDiff,
                                                                   preFilterCap,
                                                                   uniquenessRatio, speckleWindowSize, speckleRange,
                                                                   mode);
}

void SGBM::calcDisparityMap(){
    // preprocess frames
    StereoMatcher::preprocessFrame(frame_1);
    StereoMatcher::preprocessFrame(frame_2);

    // compute disparity map using the stereoObj
    stereo->compute(frame_1, frame_2, disp_map);

    // normalize the disparity map to uint8
    cv::normalize(disp_map, disp_map, 0, 255, cv::NORM_MINMAX, CV_8U);
    // calc colored disparity map
    colored_disp_map = disp_map;
}

void BP::getStereoObj(){
    //Create Stereo Matcher Object for cuda::StereoBeliefPropagation and return the pointer
    int iters = 5;
    int levels = 5;
    stereo = cv::cuda::createStereoBeliefPropagation(ndisp, iters, levels, CV_32F);
    stereo->estimateRecommendedParams(frame_w, frame_h, ndisp, iters, levels);
}

void BP::calcDisparityMap(){
    // preprocess frames
    StereoMatcher::preprocessFrame(frame_1);
    StereoMatcher::preprocessFrame(frame_2);

    // upload frames to gpu memory
    gpu_frame_1.upload(frame_1);
    gpu_frame_2.upload(frame_2);
    // compute disparity map using the stereoObj
    stereo->compute(gpu_frame_1, gpu_frame_2, gpu_disp_map);

    // refine Disparity Map with bilateral Filter
    cv::Ptr <cv::cuda::DisparityBilateralFilter> disp_filter = cv::cuda::createDisparityBilateralFilter(ndisp, 5, 1);
    disp_filter->apply(gpu_disp_map, gpu_frame_1, gpu_disp_map);

    // calc colored disparity map
    cv::cuda::drawColorDisp(gpu_disp_map, gpu_colored_disp, ndisp);
    // normalize the disparity map to uint8
    cv::cuda::normalize(gpu_disp_map, gpu_disp_map, 0, 255, cv::NORM_MINMAX, CV_8U);

    // download results from gpu memory
    gpu_disp_map.download(disp_map);
    gpu_colored_disp.download(colored_disp_map);
}

void CSBP::getStereoObj(){
    //Create Stereo Matcher Object for cuda::StereoConstantSpaceBP and return the pointer
    int iters = 8;
    int levels = 4;
    int nr_plane = 4;
    stereo = cv::cuda::createStereoConstantSpaceBP(ndisp, iters, levels, nr_plane, CV_32F);
    stereo->estimateRecommendedParams(frame_w, frame_h, ndisp, iters, levels, nr_plane);
}

void CSBP::calcDisparityMap(){
    // preprocess frames
    StereoMatcher::preprocessFrame(frame_1);
    StereoMatcher::preprocessFrame(frame_2);

    // upload frames to gpu memory
    gpu_frame_1.upload(frame_1);
    gpu_frame_2.upload(frame_2);
    // compute disparity map using the stereoObj
    stereo->compute(gpu_frame_1, gpu_frame_2, gpu_disp_map);

    // refine Disparity Map with bilateral Filter
    cv::Ptr <cv::cuda::DisparityBilateralFilter> disp_filter = cv::cuda::createDisparityBilateralFilter(ndisp, 5, 1);
    disp_filter->apply(gpu_disp_map, gpu_frame_1, gpu_disp_map);

    // calc colored disparity map
    cv::cuda::drawColorDisp(gpu_disp_map, gpu_colored_disp, ndisp);
    // normalize the disparity map to uint8
    cv::cuda::normalize(gpu_disp_map, gpu_disp_map, 0, 255, cv::NORM_MINMAX, CV_8U);

    // download results from gpu memory
    gpu_disp_map.download(disp_map);
    gpu_colored_disp.download(colored_disp_map);
}

