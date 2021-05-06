#pragma once
#include <opencv2\imgproc.hpp>

using namespace std;
using namespace cv;


vector<double> ComputeApproximatePupilCenter(vector<Point3d> Rcs, vector<Point3d> pupils);
vector<Point3d> computeApproximateOpticalAxis(vector<double> pupil_center, vector<Point3d> approximate_pupils, vector<Point3d> Rcs);
vector<Point2d> Camera2Picture(vector<double> lamdas, vector<Point3d> centers, const Mat& CameraMatrix);
vector<Point2d> Camera2Picture(vector<Point3d> centers, const Mat& CameraMatrix);
vector<Point3d> locateGazeTarget(vector<Point3d> objectModel, vector<Point2d> boardPoints, vector<Point2d> targetMarker, const CameraIntrinsics& intrinsics);
vector<Point2d> computeAxis(const vector<Point2d>& pupil_in_pic, const vector<Point2d>&  cornea_in_pic);
Point2d Camera2Picture(Point3d centers, const Mat& CameraMatrix);
vector<Point3d> ComputeCorneaCenters(double corneaRadius, const std::vector<double>& corneaDistanceRange, const std::vector<cv::Point3d>& leds, const vector<vector<Point3d>>& reflectionRays, const cv::Mat& cameraMatrix);