#Stereo Matcher

## Description
####General Information
The tool takes two rectified colorvideos of a stereo camera as input and calculates 
the according disparity map for every frame. The disparity map is displayed and written to 
an output file.

There are 4 available stereo matching algorithms available in openCV 3.x and thus implemented 
here

   - Block Matching (BM)
   - Semi-Global Block Matching (SGBM)
   - Belief Propagation (BP)
   - Constant Space Belief Propagation (CSBP)
   
The CUDA implementation of all algorithms is used, except for SGBM. 
The displayed disparity-maps are calculated by openCVs drawColorDisp, only the 
SGBM-created disparity maps are displayed in grayscale. 


#### Preprocesing

All used frames are preprocessed using the following steps:

 - Convert to grayscale
 - Equalize intensity histogram
 - Gaussian Blur
  

##Dependencies
OpenCV 3.x with CUDA enabeled


## Usage

StereoMatcher.out 'Input_Video_1' 'Input_Video_2' 'Output_Video' 'Algorithm'

'Algorithm' may be BM, SGBM, BP, CSBP, Default is BM