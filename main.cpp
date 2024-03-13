// // #include <opencv2/imgcodecs.hpp>
// // #include <opencv2/highgui.hpp>
// // #include <opencv2/imgproc.hpp>
// // #include <iostream>

// // using namespace cv;
// // using namespace std;

// // void getContours(Mat imgDil, Mat img) {

// //     vector<vector<Point>> contours;
// //     vector<Vec4i> hierarchy;

// //     findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
// //     //drawContours(img, contours, -1, Scalar(255, 0, 255), 2);

// //     vector<vector<Point>> conPoly(contours.size());
// //     vector<Rect> boundRect(contours.size());
     
// //     for (int i = 0; i < contours.size(); i++)
// //     {
// //         int area = contourArea(contours[i]);
// //         cout << area << endl;
// //         string objectType;

// //         if (area > 1000)
// //         {
// //             float peri = arcLength(contours[i], true);
// //             approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
// //             cout << conPoly[i].size() << endl;
// //             boundRect[i] = boundingRect(conPoly[i]);
       
// //             int objCor = (int)conPoly[i].size();

// //             if (objCor == 3) { objectType = "Triangle"; }
// //             else if (objCor == 4)
// //             {
// //                 float aspRatio = (float)boundRect[i].width / (float)boundRect[i].height;
// //                 cout << aspRatio << endl;
// //                 if (aspRatio> 0.95 && aspRatio< 1.05){ objectType = "Square"; }
// //                 else { objectType = "Rectangle";}
// //             }
// //             if (objCor == 10) { objectType = "Pentagram"; }
// //             else if (objCor > 10) { objectType = "Circle"; }

// //             drawContours(img, conPoly, i, Scalar(255, 0, 255), 2);
// //             rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 5);
// //             putText(img, objectType, { boundRect[i].x,boundRect[i].y - 5 }, FONT_HERSHEY_PLAIN,1, Scalar(0, 69, 255), 2);
// //         }
// //     }
// // }

// // int main() {
     
// //     string path = "/home/dar/Desktop/test/0331_4.jpg";
// //     Mat img = imread(path);
// //     Mat imgGray, imgBlur, imgCanny, imgDil, imgErode;

// //     // Preprocessing
// //     cvtColor(img, imgGray, COLOR_BGR2GRAY);
// //     GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
// //     Canny(imgBlur, imgCanny, 25, 75);
// //     Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
// //     dilate(imgCanny, imgDil, kernel);

// //     getContours(imgDil,img);

// //     imshow("Image",img);
// //     //imshow("Image Gray", imgGray);
// //     //imshow("Image Blur", imgBlur);
// //     //imshow("Image Canny", imgCanny);
// //     //imshow("Image Dil", imgDil);

// //     waitKey(0);
// //     return 0;
// // }
// #include <opencv2/opencv.hpp>
// #include <iostream>

// using namespace cv;
// using namespace std;

// void getContours(Mat imgDil, Mat img) {

//     vector<vector<Point>> contours;
//     vector<Vec4i> hierarchy;

//     vector<int> a;
//     findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//     //drawContours(img, contours, -1, Scalar(255, 0, 255), 2);

//     vector<vector<Point>> conPoly(contours.size());
//     vector<Rect> boundRect(contours.size());
     
//     for (int i = 0; i < contours.size(); i++)
//     {
//         int area = contourArea(contours[i]);
//         cout << area << endl;
//         string objectType;

//         if (area > 500)
//         {
//             float peri = arcLength(contours[i], true);
//             approxPolyDP(contours[i], conPoly[i], 0.026 * peri, true);
//             cout << conPoly[i].size() << endl;
//             boundRect[i] = boundingRect(conPoly[i]);
            
//             int objCor = (int)conPoly[i].size();

//             if (objCor == 3) { objectType = "3"; }
//             else if (objCor == 4)
//             {
//                 float aspRatio = (float)boundRect[i].width / (float)boundRect[i].height;
//                 cout << aspRatio << endl;
//                 if (aspRatio> 0.95 && aspRatio< 1.05){ objectType = "4"; }
//                 else { objectType = "4";}
//             }
//             if (objCor == 5) { objectType = "5";}
//             if (objCor == 6) { objectType = "6";}
//             if (objCor == 7) { objectType = "7";}
//             if (objCor == 8) { objectType = "8";}
//             if (objCor == 10) { objectType = "10s"; }
//             else if (objCor > 11) { objectType = "Circle"; }

//             drawContours(img, conPoly, i, Scalar(255, 0, 255), 2);
//             rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 5);
//             putText(img, objectType, { boundRect[i].x,boundRect[i].y - 5 }, FONT_HERSHEY_PLAIN,1, Scalar(0, 69, 255), 2);
//         }
//     }
// }

// // void removeNoise(Mat img, Mat &result) {
// //     Mat imgGray, imgBlur, imgCanny, imgDil, imgErode;
// //     cvtColor(img, imgGray, COLOR_BGR2GRAY);
// //     GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
// //     Canny(imgBlur, imgCanny, 25, 75);
// //     Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
// //     dilate(imgCanny, imgDil, kernel);
// //     erode(imgDil, imgErode, kernel);

// //     result = imgErode;
// // }

// int main() {
// string path = "/home/dar/Desktop/test/1_2.jpg";
//     Mat img;
//     Mat imgProcessed, imgContours;
//     img = imread(path);
//     Mat imgGray, imgBlur, imgCanny, imgDil, imgErode;

//     // Preprocessing
//     cvtColor(img, imgGray, COLOR_BGR2GRAY);
//     Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
//     //erode(imgGray, imgErode, kernel);
//     Mat imgMor;
    
//     //threshold(imgGray,imgGray,170,225,THRESH_BINARY);
//     // morphologyEx(img,imgMor,MORPH_ERODE,kernel);
//     // imshow("1",imgMor);
//     // morphologyEx(imgMor,imgMor,MORPH_DILATE,kernel);
//     // imshow("2",imgMor);
//     // morphologyEx(img,imgMor,MORPH_ERODE,kernel);
//     // //通道分离
//     // imshow("2",imgMor);
//     // morphologyEx(img,imgMor,MORPH_DILATE,kernel);
//     // imshow("3",imgMor);
//     // morphologyEx(img,imgMor,MORPH_CLOSE,kernel);
//     // morphologyEx(img,imgMor,MORPH_OPEN,kernel);
//     //imshow("4",imgMor);
//     medianBlur(img, imgMor, 5);
    
//     //GaussianBlur(imgMor,imgMor,Size(3,3),1);
//     //GaussianBlur(img,img,Size(3, 3),3);
//     //morphologyEx(img,img,MORPH_GRADIENT,kernel);
//     //medianBlur(img, img, 11);
//     //imshow("Processed Image", imgMor);
//     Canny(imgMor, imgCanny, 70, 100);
//     imshow("Original Image", imgCanny);
//     dilate(imgCanny, imgDil, kernel);

//     getContours(imgDil,img);

//     imshow("Image" ,img);

//     // Show the output
//     //imshow("Original Image", img);
//     //imshow("Processed Image", imgGray);
//     //imshow("Contours Image", imgContours);
//     waitKey(0);

//     return 0;
// }







// #include <opencv2/opencv.hpp>  
// #include <vector>  
  
// // 函数声明  
// std::string formatColor(const cv::Scalar& color);  
  
// int main() {  
//     // 打开摄像头  
//     cv::VideoCapture cap(0);  
//     if (!cap.isOpened()) {  
//         std::cerr << "无法打开摄像头！" << std::endl;  
//         return -1;  
//     }  
  
//     // 设置摄像头分辨率  
//     cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);  
//     cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);  
  
//     // 转换到HSV颜色空间  
//     cv::Mat hsv;  
  
//     while (true) {  
//         // 读取一帧  
//         cv::Mat frame;  
//         cap >> frame;  
//         if (frame.empty()) break;  
  
//         // 转换到HSV颜色空间  
//         hsv = cv::Mat::zeros(frame.size(), frame.type());  
//         cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);  
  
//         // 定义九宫格色块的位置和大小  
//         int blockSize = frame.cols / 3;  
//         std::vector<cv::Scalar> colors;  
//         for (int i = 0; i < 3; ++i) {  
//             for (int j = 0; j < 3; ++j) {  
//                 cv::Rect block(j * blockSize, i * blockSize, blockSize, blockSize);  
        
//                 // 确保块在图像范围内  
//                 if (block.x + block.width > frame.cols || block.y + block.height > frame.rows) {  
//                     continue; // 跳过超出边界的块  
//                 }  
        
//                 // 提取色块  
//                 cv::Mat blockROI = hsv(block);  
        
//                 // 对色块应用阈值操作，以提取主要颜色  
//                 cv::Mat mask;  
//                 cv::inRange(blockROI, cv::Scalar(0, 100, 100), cv::Scalar(180, 255, 255), mask);  
        
//                 // 检查掩膜是否为空  
//                 if (mask.empty() || cv::countNonZero(mask) == 0) {  
//                     continue; // 跳过空掩膜  
//                 }  
        
//                 // 寻找轮廓  
//                 std::vector<std::vector<cv::Point>> contours;  
//                 cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);  
        
//                 // 如果找到轮廓，计算其颜色  
//                 if (!contours.empty() && !contours[0].empty()) {  
//                     cv::Moments m = cv::moments(contours[0]);  
//                     if (m.m00 != 0) {  
//                         cv::Scalar color;  
//                         cv::meanStdDev(blockROI, color, cv::noArray(), mask);  
        
//                         // 将颜色添加到存储中  
//                         colors.push_back(color);  
        
//                         // 在原图上画出色块轮廓和颜色  
//                         cv::rectangle(frame, block, cv::Scalar(0, 255, 0), 2);  
//                         cv::putText(frame, formatColor(color), block.tl() + cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));  
//                     }  
//                 }  
//             }  
//         }
  
//         // 显示结果  
//         cv::imshow("Color Grid", frame);  
  
//         // 退出条件  
//         char c = (char)cv::waitKey(25);  
//         if (c == 27) break; // 按ESC退出  
//     }  
  
//     // 释放资源  
//     cap.release();  
//     cv::destroyAllWindows();  
  
//     return 0;  
// }  
  
// // 将颜色转换为字符串格式  
// std::string formatColor(const cv::Scalar& color) {  
//     std::stringstream ss;  
//     ss << "Color: (" << static_cast<int>(color[0]) << ", "  
//        << static_cast<int>(color[1]) << ", "  
//        << static_cast<int>(color[2]) << ")";  
//     return ss.str();  
// }
// #include <opencv2/opencv.hpp>  
// #include <vector>  
// // 检查轮廓面积是否在指定范围内
// bool isContourValid(const cv::Mat& frame, const std::vector<cv::Point>& contour, int minArea, int maxArea) {  
//     double area = cv::contourArea(contour);  
//     return area >= minArea && area <= maxArea;  
// }  
// cv::Scalar bgr2hsv(const cv::Scalar& bgr, int hrange = 255, bool full = true) {
// 	float b = bgr[0];
// 	float g = bgr[1];
// 	float r = bgr[2];
// 	float maxVal = std::max({ r, g, b });
// 	float minVal = std::min({ r, g, b });

// 	float hue = 0.0f;

// 	if (maxVal != minVal) {
// 		if (maxVal == b)
// 			hue = 60 * (r - g) / (maxVal - minVal) + 240.0f;
// 		else if (maxVal == g)
// 			hue = 60 * (b - r) / (maxVal - minVal) + 120.0f;
// 		else if (maxVal == r)
// 			hue = 60 * (g - b) / (maxVal - minVal);

// 		if (hue < 0.0f)
// 			hue += 360.0f;
// 	}

// 	float saturation = maxVal == 0.0f ? 0.0f : (maxVal - minVal) / maxVal;
// 	float value = maxVal;

// 	if (full)
// 		hue = hue * hrange / 360.0f;
// 	else
// 		hue /= 2.0f;

// 	hue *= hrange / 360.0f;
// 	saturation *= hrange;
// 	value *= hrange;

// 	hue = cv::saturate_cast<uchar>(hue);
// 	saturation = cv::saturate_cast<uchar>(saturation);
// 	value = cv::saturate_cast<uchar>(value);
// 	return cv::Scalar(hue, saturation, value);
// }
// cv::Scalar upd(cv::Scalar a){
//     return cv::Scalar(bgr2hsv(a)[0]-5,100,100);
// }
// cv::Scalar dod(cv::Scalar a){
//     return cv::Scalar(bgr2hsv(a)[0]+5,255,255);
// }

// int main() {  
//     // 打开摄像头  
//     cv::VideoCapture cap(0);  
//     if (!cap.isOpened()) {  
//         std::cerr << "Error opening camera" << std::endl;  
//         return -1;  
//     }  
  
//     // 定义HSV颜色范围  
//     std::vector<cv::Scalar> lower_colors = {
//         upd(cv::Scalar(116, 5, 202)),    // 红色  
//         upd(cv::Scalar(167, 1, 98)),  // 橙色  
//         upd(cv::Scalar(7, 237, 19)),  // 黄色  
//         upd(cv::Scalar(116, 5, 202)),  // 绿色  
//         upd(cv::Scalar(116, 5, 202)),   // 蓝色  
//         // ... 添加其他颜色  
//     };  
  
//     std::vector<cv::Scalar> upper_colors = {  
//         dod(cv::Scalar(116, 5, 202)),  // 红色  
//         dod(cv::Scalar(167, 1, 98)),  // 橙色  
//         dod(cv::Scalar(7, 237, 19)),  // 黄色  
//         dod(cv::Scalar(116, 5, 202)),  // 绿色  
//         dod(cv::Scalar(116, 5, 202)),  // 蓝色  
//         // ... 添加其他颜色  
//     };  
  
//     // 检查颜色范围的数量是否一致  
//     if (lower_colors.size() != upper_colors.size()) {  
//         std::cerr << "Error: The size of lower_colors and upper_colors must be the same." << std::endl;  
//         return -1;  
//     }  
  
//     cv::Mat frame, hsv_frame, color_mask,filtered_mask;  
//     while (true) {  
//         cap >> frame;  
//         if (frame.empty()) break;  
  
//         // 转换到HSV颜色空间  
//         cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV);  
  
//         // 遍历每种颜色并创建掩码  
//         for (size_t i = 0; i < lower_colors.size(); i++) {  
//             cv::inRange(hsv_frame, lower_colors[i], upper_colors[i], color_mask);  
//             cv::bitwise_or(color_mask, color_mask, color_mask); // 如果需要合并多个颜色，使用bitwise_or  

//             // 形态学操作消除噪点（腐蚀和膨胀）  
//             cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));  
//             cv::morphologyEx(color_mask, filtered_mask, cv::MORPH_OPEN, kernel);  
//             cv::morphologyEx(filtered_mask, filtered_mask, cv::MORPH_CLOSE, kernel);  
//             // 在原始图像上绘制轮廓  
//             std::vector<std::vector<cv::Point>> contours;  
//             cv::findContours(color_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);  
            
            
//             // 定义轮廓的最小和最大面积（根据实际场景调整）  
//             int minContourArea = 6500; // 最小面积阈值  
//             int maxContourArea = frame.total() / 40; // 最大面积阈值，设为图像总面积的1/10  
    
//             // 绘制有效轮廓  
//             for (const auto& contour : contours) {  
//                 if (isContourValid(frame, contour, minContourArea, maxContourArea)) {  
//                     cv::Rect bounding_rect = cv::boundingRect(contour);  
//                     cv::rectangle(frame, bounding_rect.tl(), bounding_rect.br(), cv::Scalar(0, 255, 0), 2);  
//                 }  
//             }  

//         }  
  
//         // 显示结果  
//         cv::imshow("Color Detection", frame);  
  
//         // 退出循环（按'q'键退出）  
//         char c = (char)cv::waitKey(1);  
//         if (c == 'q' || c == 27) {  
//             break;  
//         }  
//     }  
  
//     // 释放资源并关闭窗口  
//     cap.release();  
//     cv::destroyAllWindows();  

//     return 0;  
// }
#include <opencv2/opencv.hpp>  
#include <vector>  
#include <string>  
bool isContourValid(const cv::Mat& frame, const std::vector<cv::Point>& contour, int minArea, int maxArea) {  
    double area = cv::contourArea(contour);  
    return area >= minArea && area <= maxArea;  
}
cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
void Cam_Init(){
    // 测定的相机内参矩阵，畸变矩阵
    cameraMatrix.at<double>(0, 0) = 1.1527e+03;
    cameraMatrix.at<double>(0, 1) = 0;
    cameraMatrix.at<double>(0, 2) = 618.6327;
    cameraMatrix.at<double>(1, 1) = 1.1527e+03;
    cameraMatrix.at<double>(1, 2) = 361.1508;

    distCoeffs.at<double>(0, 0) = 0.0741;
    distCoeffs.at<double>(1, 0) = -0.0685;
    distCoeffs.at<double>(2, 0) = 0;
    distCoeffs.at<double>(3, 0) = 0;
    distCoeffs.at<double>(4, 0) = 0;
}
double solveAngles(std::vector<cv::Point3f> POINTS_3D,std::vector<cv::Point2f> POINTS_2D);
int main() {  
    cv::VideoCapture cap(0); // 打开默认摄像头  
    if (!cap.isOpened()) {  
        std::cerr << "Error opening camera" << std::endl;  
        return -1;  
    }  

    cv::Mat cap_frame,frame, hsv_frame, filtered_mask;  
    std::vector<cv::Scalar> lowerb, upperb; // 用于存储多种颜色的HSV范围  
    std::vector<cv::Mat> color_masks; // 用于存储每个颜色的掩码  
    std::vector<std::string> color_labels; // 用于存储每种颜色的标注信息
    std::vector<cv::Point2f> color_centers;
    


    // 定义多种颜色的HSV范围（这里以红色、绿色和蓝色为例）  
    // 注意：这些范围需要根据实际情况进行调整  
    
    lowerb.push_back(cv::Scalar(118, 250, 250)); // 蓝色 
    lowerb.push_back(cv::Scalar(0, 250, 250)); // 红色  
    lowerb.push_back(cv::Scalar(58, 250, 250)); // 绿色   
    lowerb.push_back(cv::Scalar(88, 250, 250)); // 蓝色 
    lowerb.push_back(cv::Scalar(148, 250, 250)); // 蓝色 
    lowerb.push_back(cv::Scalar(28, 250, 250)); // 蓝色 
    lowerb.push_back(cv::Scalar(12, 200, 200)); // 蓝色 
    lowerb.push_back(cv::Scalar(100, 120, 200)); // 蓝色 
    lowerb.push_back(cv::Scalar(110, 150, 70)); // 蓝色 

    
    upperb.push_back(cv::Scalar(122, 255, 255)); // 蓝色 
    upperb.push_back(cv::Scalar(2, 255, 255)); // 红色  
    upperb.push_back(cv::Scalar(62, 255, 255)); // 绿色   
    upperb.push_back(cv::Scalar(92, 255, 255)); // 蓝色 
    upperb.push_back(cv::Scalar(152, 255, 255)); // 蓝色 
    upperb.push_back(cv::Scalar(32, 255, 255)); // 蓝色 
    upperb.push_back(cv::Scalar(16, 255, 255)); // 蓝色 
    upperb.push_back(cv::Scalar(110, 180, 255)); // 蓝色 
    upperb.push_back(cv::Scalar(130, 255, 200)); // 蓝色 
    // 为每种颜色定义标注信息  
    
    color_labels.push_back("B"); 
    color_labels.push_back("R"); 
    color_labels.push_back("G");
    color_labels.push_back("Q");
    color_labels.push_back("P");
    color_labels.push_back("Y");
    color_labels.push_back("O");
    color_labels.push_back("HL");
    color_labels.push_back("L");
    cv::namedWindow("Multi-Color Detection", cv::WINDOW_AUTOSIZE);  
  
    while (true) {  
        cap >> cap_frame;
        frame = cv::imread("/home/dar/Desktop/test/puzzle.jpg");
        if (frame.empty()) break;
  
        hsv_frame.create(frame.size(), frame.type());  
        cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV);  

        // 为每种颜色创建掩码，并查找轮廓  
        for (size_t i = 0; i < lowerb.size(); i++) {  
            cv::Mat color_mask;  
            cv::inRange(hsv_frame, lowerb[i], upperb[i], color_mask);  

            // 形态学操作消除噪点（腐蚀和膨胀）  
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));  
            cv::morphologyEx(color_mask, filtered_mask, cv::MORPH_OPEN, kernel);  
            cv::morphologyEx(filtered_mask, filtered_mask, cv::MORPH_CLOSE, kernel);  
            
            std::vector<std::vector<cv::Point>> contours;  
            cv::findContours(color_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);  
            // 定义轮廓的最小和最大面积（根据实际场景调整）  
            int minContourArea = 10000; // 最小面积阈值  
            int maxContourArea = frame.total() / 9; // 最大面积阈值，设为图像总面积的1/10  

            // 遍历每个轮廓并添加标注  
            for (const auto& contour : contours) { 
                if (isContourValid(frame, contour, minContourArea, maxContourArea)) {   
                    cv::Rect bounding_rect = cv::boundingRect(contour);  
    
                    // 在原始图像上绘制边界框  
                    cv::rectangle(frame, bounding_rect.tl(), bounding_rect.br(), cv::Scalar(0, 255, 0), 2);  
                    // 计算矩形的中心点  
                    cv::Point2f center(bounding_rect.x + bounding_rect.width / 2, bounding_rect.y + bounding_rect.height / 2);  
                    
                    // 等待解算
                    color_centers.push_back(center);

                    // 在中心点上绘制一个红色的圆点  
                    cv::circle(frame, center, 5, cv::Scalar(0, 0, 255), -1);  
                    // 在边界框上方添加标注信息  
                    std::string label = color_labels[i] + ": " + std::to_string(bounding_rect.area());  
                    int baseLine;  
                    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);  
                    cv::putText(frame, label, cv::Point(center.x-labelSize.width/2.0,center.y+1.5*labelSize.height) ,  
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);  
                    cv::waitKey(1000);
                    cv::imshow("Multi-Color Detection", frame);  
                } 
            }
            cv::Mat rvec,tvec;
            std::vector<cv::Point3f> realPoint{cv::Point3f(-23,23,0),cv::Point3f(0,23,0),
                                                cv::Point3f(23,23,0),cv::Point3f(-23,0,0),
                                                cv::Point3f(0,0,0),cv::Point3f(23,0,0),
                                                cv::Point3f(-23,-23,0),cv::Point3f(0,-23,0),
                                                cv::Point3f(23,-23,0)};
            std::cout<<color_centers<<std::endl;
            if(color_centers.size()==9){
                solveAngles(realPoint,color_centers);
                color_centers.clear();
            }
            
            for(const auto& contour : color_centers){

            }
        }  
  
        // 显示结果  
        // cv::imshow("Multi-Color Detection", frame);  
        // cv::imshow("Cap Frame",cap_frame);
        // 按'q'键退出  
        if (cv::waitKey(1) == 'q') {  
            break;  
        }  
    } 
    // 释放资源并关闭窗口  

    cap.release();  
    cv::destroyAllWindows();  

    return 0;  
}
double solveAngles(std::vector<cv::Point3f> POINTS_3D,std::vector<cv::Point2f> POINTS_2D)
{
	cv::Mat _rvec,_tvec;
	// std::cout<<POINTS_3D<<POINTS_2D<<std::endl;
	if(POINTS_2D.empty())
	{
		std::cout<<"NO POINTS FOUND"<<std::endl;
        return -1;
	}
	else
	{
		cv::solvePnP(POINTS_3D, POINTS_2D, cameraMatrix, distCoeffs, _rvec, _tvec, false, cv::SOLVEPNP_ITERATIVE);

		_tvec.at<double>(1, 0) -= 0;
		double x_pos = _tvec.at<double>(0, 0);
		double y_pos = _tvec.at<double>(1, 0);
		double z_pos = _tvec.at<double>(2, 0);
		double distance = sqrt(x_pos * x_pos + y_pos * y_pos + z_pos * z_pos);
		double angle = atan2(23,distance)/M_PI*180.0;

        std::cout << "distance : " << distance << std::endl;
        std::cout << "angle : " << angle << std::endl;

        return angle;
	}
    
	
}