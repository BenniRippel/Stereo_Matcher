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
    int blockSize = std::stoi(configValues.at("BM_blockSize"));
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
    int blockSize = std::stoi(configValues.at("SGBM_blockSize"));
    int P1 = std::stoi(configValues.at("SGBM_P1"));
    int P2 = std::stoi(configValues.at("SGBM_P2"));
    int disp12MaxDiff = std::stoi(configValues.at("SGBM_disp12MaxDiff"));
    int preFilterCap = std::stoi(configValues.at("SGBM_preFilterCap"));
    int uniquenessRatio = std::stoi(configValues.at("SGBM_uniquenessRatio"));
    int speckleWindowSize = std::stoi(configValues.at("SGBM_speckleWindowSize"));
    int speckleRange = std::stoi(configValues.at("SGBM_speckleRange"));
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
    int iters = std::stoi(configValues.at("BP_iters"));
    int levels = std::stoi(configValues.at("BP_levels"));
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
    int iters = std::stoi(configValues.at(("CSBP_iters")));
    int levels = std::stoi(configValues.at("CSBP_levels"));
    int nr_plane = std::stoi(configValues.at("CSBP_nr_plane"));
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

void StereoMatcher::readConfig(){
    // Read Config File and store key-value pairs as strings in configValues map
    std::ifstream fs;  // open filestream
    fs.open ("./config.txt");

    try
    {
        if (!fs) {
            throw 1;
        }
        // if file is opened, get key, value pairs, ignore # and /
        else {
            std::string line;
            while (std::getline(fs, line)) {
                std::istringstream is_line(line);
                std::string key;
                if (std::getline(is_line, key, '=')) {
                    std::string value;
                    if (key[0] == '#' || key[0] == '/') // check for comments in config file
                        continue;

                    if (std::getline(is_line, value)) {
                        configValues[key] = value;
                    }
                }
            }
        }
        fs.close(); // close filestream
    }
    catch (int e)
    {
        std::cout << "Could not open & read config.txt! Please check file and location. Terminating..." << std::endl;
        std::terminate();
    }
}