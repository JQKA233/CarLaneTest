// #include<iostream>
// #include<opencv2/opencv.hpp>
// using namespace std;
// using namespace cv;

// void GetROI(Mat src, Mat &ROI)
// {
// 	int width = src.cols;
// 	int height = src.rows;

// 	//获取车道ROI区域，只对该部分进行处理
// 	vector<Point>pts;
// 	Point ptA((width / 8) * 0, (height / 20) * 19);
// 	Point ptB((width / 8) * 0, (height / 8) * 7);
// 	Point ptC((width / 10) * 4, (height / 2) * 1);
// 	Point ptD((width / 10) * 6, (height / 2) * 1);
// 	Point ptE((width / 8) * 8, (height / 8) * 6);
// 	Point ptF((width / 8) * 8, (height / 20) * 19);
// 	pts = { ptA ,ptB,ptC,ptD,ptE, ptF };

// 	//opencv4版本 fillPoly需要使用vector<vector<Point>>
// 	vector<vector<Point>>ppts;
// 	ppts.push_back(pts);

// 	Mat mask = Mat::zeros(src.size(), src.type());
// 	fillPoly(mask, ppts, Scalar::all(255));

// 	src.copyTo(ROI, mask);
// 	imshow("src", mask);
	
// }

// void DetectRoadLine(Mat src,Mat &ROI)
// {
// 	Mat gray;
// 	cvtColor(ROI, gray, COLOR_BGR2GRAY);

// 	Mat thresh;
// 	threshold(gray, thresh, 180, 255, THRESH_BINARY);

// 	vector<Point>left_line;
// 	vector<Point>right_line;

// 	//左车道线
// 	for (int i = 0; i < thresh.cols / 2; i++)
// 	{
// 		for (int j = thresh.rows/2; j < thresh.rows; j++)
// 		{
// 			if (thresh.at<uchar>(j, i) == 255)
// 			{
// 				left_line.push_back(Point(i, j));
// 			}
// 		}
// 	}
// 	//右车道线
// 	for (int i = thresh.cols / 2; i < thresh.cols; i++)
// 	{
// 		for (int j = thresh.rows / 2; j < thresh.rows; j++)
// 		{
// 			if (thresh.at<uchar>(j, i) == 255)
// 			{
// 				right_line.push_back(Point(i, j));
// 			}
// 		}
// 	}

// 	//车道绘制
// 	if (left_line.size() > 0 && right_line.size() > 0)
// 	{
// 		Point B_L = (left_line[0]);
// 		Point T_L = (left_line[left_line.size() - 1]);
// 		Point T_R = (right_line[0]);
// 		Point B_R = (right_line[right_line.size() - 1]);

// 		circle(src, B_L, 10, Scalar(0, 0, 255), -1);
// 		circle(src, T_L, 10, Scalar(0, 255, 0), -1);
// 		circle(src, T_R, 10, Scalar(255, 0, 0), -1);
// 		circle(src, B_R, 10, Scalar(0, 255, 255), -1);

// 		line(src, Point(B_L), Point(T_L), Scalar(0, 255, 0), 10);
// 		line(src, Point(T_R), Point(B_R), Scalar(0, 255, 0), 10);

// 		vector<Point>pts;
// 		pts = { B_L ,T_L ,T_R ,B_R };
// 		vector<vector<Point>>ppts;
// 		ppts.push_back(pts);
// 		fillPoly(src, ppts, Scalar(133, 230, 238));
// 	}
// }

// int main()
// {

// 	VideoCapture cap(0);

// 	Mat frame, image;
	
	
// 	while (1)
// 	{	
// 		waitKey(10);

// 		// cap >> frame;
// 		frame = imread("/home/dar/Desktop/test/lane2.jpg");

// 		GetROI(frame, image);

// 		DetectRoadLine(frame, image);

// 		imshow("frame", frame);
// 	}

// 	// capture.release();
// 	// destroyAllWindows();
// 	// system("pause");
// 	return 0;
// }

#include <opencv2/opencv.hpp>  
#include <iostream>  
  
using namespace cv;  
using namespace std;  

int main(int argc, char** argv) {  
    // 1. 读取图像  
    Mat img = imread("/home/WAYI/CarlaneTest/lane2.jpg");  
    if (img.empty()) {  
        cout << "Could not open or find the image!" << endl;  
        return -1;  
    }  
  
    // 2. 转换为灰度图像  
    Mat gray;  
    cvtColor(img, gray, COLOR_BGR2GRAY);  
  
    // 3. 应用高斯模糊以减少图像噪声  
    Mat blurred;  
    GaussianBlur(gray, blurred, Size(5, 5), 0);  
  
    // 4. 使用Canny边缘检测  
    Mat edges;  
    Canny(blurred, edges, 50, 150);  
	imshow("",edges);
  
    // 5. 应用霍夫变换来检测直线  
    vector<Vec4i> lines;  
    HoughLinesP(edges, lines, 1, CV_PI/180, 80, 30, 10);  
  
    // 6. 在原图像上绘制检测到的直线  
    for (size_t i = 0; i < lines.size(); i++) {  
        Vec4i l = lines[i];  
		line(img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 3, LINE_AA);  
		if((fabs(l[0]-l[2])*fabs(l[0]-l[2])+fabs(l[1]-l[3])*fabs(l[1]-l[3]))>50000){
			line(img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);  
		}
		
    }  
  
    // 7. 显示结果  
    imshow("Lane Detection", img);  
    waitKey(0);  
  
    return 0;  
}