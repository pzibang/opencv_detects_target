#include"opencv2/opencv.hpp"  
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc/types_c.h"
#include <iostream>  
using namespace cv;
using namespace std;

//实现目标：将图片虫子.png的轮廓用矩形框框出来

int main()
{
    String image_path = "./虫子.png";
    //载入图片
    Mat image = imread(image_path);
    imshow("source image", image);
    //图像灰度
    Mat image_gray;
    cvtColor(image, image_gray, COLOR_BGR2GRAY);
    //进行高斯滤波
    Mat image_blur;
    GaussianBlur(image_gray, image_blur, Size(3, 3), 0, 0);

    //提取图像梯度
    Mat grad_x, grad_y, image_grad, abs_grad_x, abs_grad_y;
    //X方向梯度
    Sobel(image_blur, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
    //Y方向梯度
    Sobel(image_blur, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT); 
    
    //之后在x方向上减去y方向上的梯度，通过这个减法，我们留下具有高水平梯度和低垂直梯度的图像区域
    subtract(grad_x, grad_y, image_grad);
    //实现图像增强等相关操作的快速运算
    convertScaleAbs(image_grad, image_grad);

    imshow("grad", image_grad);

    //再次高斯模糊
    Mat image_blurred;
    GaussianBlur(image_grad, image_blurred, Size(9, 9), 0);
    //二值化，选择90，255为阈值
    Mat image_diff_thresh;
    threshold(image_blurred, image_diff_thresh, 90, 255, CV_THRESH_BINARY);

    //图像形态学
    Mat kernel_dilate = getStructuringElement(MORPH_ELLIPSE, Size(25, 25));
    //closed 操作
    Mat image_closed;
    morphologyEx(image_diff_thresh, image_closed, MORPH_CLOSE, kernel_dilate);

    imshow("图像形态学", image_diff_thresh);

    //执行形态学腐蚀与膨胀
    //腐蚀
    Mat element_erode = getStructuringElement(MORPH_CROSS, Size(10, 10));
    erode(image_closed, image_closed, element_erode);

    //膨胀
    Mat element_dilate = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));
    dilate(image_closed, image_closed, element_dilate);

    imshow("图像形态学", image_closed);
    
    //查找轮廓
    //查找轮廓并绘制轮廓  
    Mat result = image.clone();
    vector<vector<Point> > contours;
    findContours(image_closed, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//找轮廓函数
    //drawContours(result, contours, -1, Scalar(0, 0, 255), 2);//在result上绘制轮廓  
     //查找正外接矩形  
    vector<Rect> boundRect(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        boundRect[i] = boundingRect(contours[i]);
        rectangle(result, boundRect[i], Scalar(0, 255, 0), 2);//在result上绘制正外接矩形  
    }

    imshow("效果图", result);

 
    waitKey(0);
    return 0;
}