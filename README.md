# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

## Project Rubrics

### FP.1 Match 3D Objects

Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.

```
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    cv::KeyPoint prevKp, currKp;
    
    int pbBoxes = prevFrame.boundingBoxes.size();
    int cbBoxes = currFrame.boundingBoxes.size();
    int pt_counts[pbBoxes][cbBoxes] = { };

    vector<int> prevBoxesIds, currBoxesIds;

    // loop over matched keypoints 
    for(auto it1=matches.begin(); it1!= matches.end()-1; ++it1)
    {
        prevKp = prevFrame.keypoints[it1->queryIdx];
        currKp = currFrame.keypoints[it1->trainIdx];

        prevBoxesIds.clear();
        currBoxesIds.clear();

        // prev frame bounding boxes contain this keypoint
        for(auto it2 = prevFrame.boundingBoxes.begin(); it2!= prevFrame.boundingBoxes.end(); ++it2)
        {
            if((*it2).roi.contains(prevKp.pt))
                prevBoxesIds.push_back((*it2).boxID);
        }
        
      	// current frame bounding boxes contain this matched keypoint
        for(auto it2 = currFrame.boundingBoxes.begin(); it2!= currFrame.boundingBoxes.end(); ++it2)
        {
            if((*it2).roi.contains(prevKp.pt))
                currBoxesIds.push_back((*it2).boxID);
        }

        // update counter
        for(auto prevId:prevBoxesIds)
        {
            for(auto currId:currBoxesIds)
                pt_counts[prevId][currId]++;
        }
    }
  
    // select best matches boxes
    for(int prevId=0; prevId<pbBoxes; prevId++)
    {
        int maxCount = 0;
      	int maxId = 0;
        for(int currId=0; currId<cbBoxes; currId++)
        {
            if (pt_counts[prevId][currId] > maxCount)
            {
                maxCount = pt_counts[prevId][currId];
                maxId = currId;
            }
        }
        bbBestMatches[prevId] = maxId;
    }
}

```
### FP.2 Compute Lidar-based TTC

Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame. 

```
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dT = 1/frameRate;        // time between two measurements in seconds
    double laneWidth = 4.0; // assumed width of the ego lane

    // find closest distance to Lidar points within ego lane
    vector<double> xPrev, xCurr;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        if (abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            xPrev.push_back(it->x);
        }
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        if (abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            xCurr.push_back(it->x);
        }
    }
	double minXPrev = 0, minXCurr = 0;
  
  	if(xPrev.size()>0)
    {
    	for(auto x : xPrev)
          	minXPrev+=x;
      	minXPrev = minXPrev/xPrev.size();
    }
  
  	if(xCurr.size()>0)
    {
    	for(auto x : xCurr)
          	minXCurr+=x;
      	minXCurr = minXCurr/xCurr.size();
    }
    
  	// compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);
}

```

### FP.3 Associate Keypoint Correspondences with Bounding Boxes

Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.

```
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // for all matches in the current frame
    for (cv::DMatch match : kptMatches) 
    {
        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt))
        {
            boundingBox.kptMatches.push_back(match);
        }
    }
}

```

### FP.4 Compute Camera-based TTC

Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.

```
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
 	// compute median dist. ratio to remove outlier influence
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; 

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}

```
### Performance Evaluation

TTC lidar ranged from about 8-17 seconds. TTC from Lidar is not always correct. example, where first TTC was ~13s, and it jumped to ~16s then decreased to ~11s. It might be resulted from some outliers and some unstable points from preceding vehicle's front mirrors.

Like Lidar, TTC from camera is not always correct as well. Sometimes the result gets unstable(-inf, NaN, etc.), It's probably because of the worse keypints matches. 

Output from all detector descriptor combination is placed in PerformanceEvaluation.csv file. Certain detector/descriptor combinations, especially the Harris and ORB detectors, produced very unreliable camera TTC estimates. Below is a sample of output from the AKAZE detector/descriptor combination 

|Detector + Descriptor|ttcLidar |ttcCamera|ttcCamera - ttcLidar|
|---------------------|---------|---------|--------------------|
|FAST,BRIEF           |12.289100|11.750258|-0.538842           |
|FAST,BRIEF           |13.354725|11.758331|-1.596394           |
|FAST,BRIEF           |16.384452|13.955042|-2.429410           |
|FAST,BRIEF           |14.076445|13.137056|-0.939390           |
|FAST,BRIEF           |12.729945|14.829752|2.099806            |
|FAST,BRIEF           |13.751074|13.594156|-0.156917           |
|FAST,BRIEF           |13.731432|13.277444|-0.453988           |
|FAST,BRIEF           |13.790126|12.775105|-1.015021           |
|FAST,BRIEF           |12.058994|12.800778|0.741784            |
|FAST,BRIEF           |11.864185|13.009249|1.145064            |
|FAST,BRIEF           |11.968197|11.873378|-0.094819           |
|FAST,BRIEF           |9.887113 |11.525555|1.638442            |
|FAST,BRIEF           |9.425037 |11.706558|2.281521            |
|FAST,BRIEF           |9.302148 |10.645161|1.343013            |
|FAST,BRIEF           |8.321200 |11.531169|3.209969            | 
|FAST,BRIEF           |8.898673 |10.323035|1.424362            |
|FAST,BRIEF           |11.030114|9.894525 |-1.135589           |
|FAST,BRIEF           |8.535568 |11.508075|2.972507            |

