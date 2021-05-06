#include <opencv2\imgproc.hpp>
#include <iostream>
#include <EyeTrackerLib\EyeTrackerModel.h>
#include <EyeTrackerLib\Geometry.h>

using namespace std;
using namespace cv;


vector<double> ComputeApproximatePupilCenter(vector<Point3d> Rcs, vector<Point3d> pupils) 
{
	vector<double> distances;
	for (size_t i = 0; i < pupils.size(); i++) {
		double a = pupils[i].x*pupils[i].x + pupils[i].y*pupils[i].y + pupils[i].z*pupils[i].z;
		double b = -2 * (pupils[i].x * Rcs[i].x + pupils[i].y * Rcs[i].y + pupils[i].z * Rcs[i].z);
		double c = Rcs[i].x * Rcs[i].x +  Rcs[i].y * Rcs[i].y + Rcs[i].z * Rcs[i].z - 7.8*7.8;
		double delta = b * b - 4 * a*c;
		double x1 = (-b + sqrt(delta)) / (2 * a);
		double x2 = (-b - sqrt(delta)) / (2 * a);
		if (abs(x1) < abs(x2))
			distances.push_back(x1);
		else
			distances.push_back(x2);
	}
	return distances;
}

vector<Point3d> computeApproximateOpticalAxis(vector<double> pupil_center, vector<Point3d> approximate_pupils, vector<Point3d> Rcs) 
{
	vector<Point3d> approximate_opticalaxis;
	for (size_t i = 0; i < Rcs.size(); i++) {
		Point3d p = Rcs[i] - approximate_pupils[i] * pupil_center[i];
		p /= sqrt(p.ddot(p));
		approximate_opticalaxis.push_back(p);
	}
	return approximate_opticalaxis;
}

Point2d Camera2Picture(Point3d centers, const Mat& CameraMatrix) {
	Point2d pt2d;
	Mat position_in_cam = (cv::Mat_<double>(3, 1) << centers.x, centers.y, centers.z);
	Mat position_in_pic = CameraMatrix * position_in_cam;
	centers.x = position_in_pic.at<double>(0, 0);
	centers.y = position_in_pic.at<double>(1, 0);
	centers.z = position_in_pic.at<double>(2, 0);
	centers /= centers.z;
	pt2d.x = centers.x;
	pt2d.y = centers.y;
	return pt2d;
}

vector<Point2d> Camera2Picture(vector<double> lamdas, vector<Point3d> centers, const Mat& CameraMatrix) {
	vector<Point2d> points;
	for (size_t i = 0; i < lamdas.size(); i++) {
		Point2d pt;
		Point3d pt3 = lamdas[i] * centers[i];
		Mat position_in_cam = (cv::Mat_<double>(3, 1) << pt3.x, pt3.y, pt3.z);
		Mat position_in_pic = CameraMatrix * position_in_cam;
		pt3.x = position_in_pic.at<double>(0, 0);
		pt3.y = position_in_pic.at<double>(1, 0);
		pt3.z = position_in_pic.at<double>(2, 0);
		pt3 /= pt3.z;
		pt.x = pt3.x;
		pt.y = pt3.y;
		points.push_back(pt);
	}
	return points;
}

vector<Point2d> Camera2Picture(vector<Point3d> centers, const Mat& CameraMatrix) {
	vector<Point2d> points;
	for (size_t i = 0; i < centers.size(); i++) {
		Point2d pt;
		Point3d pt3 = centers[i];
		Mat position_in_cam = (cv::Mat_<double>(3, 1) << pt3.x, pt3.y, pt3.z);
		Mat position_in_pic = CameraMatrix * position_in_cam;
		pt3.x = position_in_pic.at<double>(0, 0);
		pt3.y = position_in_pic.at<double>(1, 0);
		pt3.z = position_in_pic.at<double>(2, 0);
		pt3 /= pt3.z;
		pt.x = pt3.x;
		pt.y = pt3.y;
		points.push_back(pt);
	}
	return points;
}

vector<Point3d> locateGazeTarget(vector<Point3d> objectModel, vector<Point2d> boardPoints, vector<Point2d> targetMarker, const CameraIntrinsics& intrinsics)
{
	vector<Point3d> gaze_targets;
	for (int i = 0; i < targetMarker.size(); i++) {
		vector<Point2d> rgb_features;
		for (int j = 8 * i; j < 8 * i + 8; j++) {
			Point2d rgb_point = boardPoints[j];
			rgb_features.push_back(rgb_point);
		}
		Mat board_b_pose = ComputeObjectPose(intrinsics, objectModel, rgb_features);

		Point3d origin;
		origin.x = board_b_pose.at<double>(0, 3);
		origin.y = board_b_pose.at<double>(1, 3);
		origin.z = board_b_pose.at<double>(2, 3);
		Point3d normal;
		normal.x = board_b_pose.at<double>(0, 2);
		normal.y = board_b_pose.at<double>(1, 2);
		normal.z = board_b_pose.at<double>(2, 2);

		Point3d target_ray = ComputeImagingRay(intrinsics, targetMarker[i]);
		Point3d camera_center(0.0, 0.0, 0.0);

		Point3d intersection = ComputeLinePlaneIntersection(origin, normal, camera_center, target_ray);
		gaze_targets.push_back(intersection);
	}
	return gaze_targets;
}

vector<Point2d> computeAxis(const vector<Point2d>& pupil_in_pic, const vector<Point2d>&  cornea_in_pic) {
	vector<Point2d> axis;
	for (size_t i = 0; i < pupil_in_pic.size(); i++) {
		Point2d ax = pupil_in_pic[i] - cornea_in_pic[i];
		axis.push_back(ax);
	}
	return axis;
}

vector<Point3d> ComputeCorneaCenters(double corneaRadius, const std::vector<double>& corneaDistanceRange, const std::vector<cv::Point3d>& leds, const vector<vector<Point3d>>& reflectionRays, const cv::Mat& cameraMatrix) {
	vector<Point3d> corneacenters;
	for (int i = 0; i < reflectionRays.size(); i++) {
		Point3d corneacenter = ComputeCorneaCenter(corneaRadius, corneaDistanceRange, leds, reflectionRays[i], cameraMatrix);
		corneacenters.push_back(corneacenter);
	}
	return corneacenters;
}