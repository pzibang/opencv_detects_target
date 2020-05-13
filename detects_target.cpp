#include"opencv2/opencv.hpp"  
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc/types_c.h"
#include <iostream>  
using namespace cv;
using namespace std;

//ʵ��Ŀ�꣺��ͼƬ����.png�������þ��ο�����

int main()
{
    String image_path = "./����.png";
    //����ͼƬ
    Mat image = imread(image_path);
    imshow("source image", image);
    //ͼ��Ҷ�
    Mat image_gray;
    cvtColor(image, image_gray, COLOR_BGR2GRAY);
    //���и�˹�˲�
    Mat image_blur;
    GaussianBlur(image_gray, image_blur, Size(3, 3), 0, 0);

    //��ȡͼ���ݶ�
    Mat grad_x, grad_y, image_grad, abs_grad_x, abs_grad_y;
    //X�����ݶ�
    Sobel(image_blur, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
    //Y�����ݶ�
    Sobel(image_blur, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT); 
    
    //֮����x�����ϼ�ȥy�����ϵ��ݶȣ�ͨ������������������¾��и�ˮƽ�ݶȺ͵ʹ�ֱ�ݶȵ�ͼ������
    subtract(grad_x, grad_y, image_grad);
    //ʵ��ͼ����ǿ����ز����Ŀ�������
    convertScaleAbs(image_grad, image_grad);

    imshow("grad", image_grad);

    //�ٴθ�˹ģ��
    Mat image_blurred;
    GaussianBlur(image_grad, image_blurred, Size(9, 9), 0);
    //��ֵ����ѡ��90��255Ϊ��ֵ
    Mat image_diff_thresh;
    threshold(image_blurred, image_diff_thresh, 90, 255, CV_THRESH_BINARY);

    //ͼ����̬ѧ
    Mat kernel_dilate = getStructuringElement(MORPH_ELLIPSE, Size(25, 25));
    //closed ����
    Mat image_closed;
    morphologyEx(image_diff_thresh, image_closed, MORPH_CLOSE, kernel_dilate);

    imshow("ͼ����̬ѧ", image_diff_thresh);

    //ִ����̬ѧ��ʴ������
    //��ʴ
    Mat element_erode = getStructuringElement(MORPH_CROSS, Size(10, 10));
    erode(image_closed, image_closed, element_erode);

    //����
    Mat element_dilate = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));
    dilate(image_closed, image_closed, element_dilate);

    imshow("ͼ����̬ѧ", image_closed);
    
    //��������
    //������������������  
    Mat result = image.clone();
    vector<vector<Point> > contours;
    findContours(image_closed, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//����������
    //drawContours(result, contours, -1, Scalar(0, 0, 255), 2);//��result�ϻ�������  
     //��������Ӿ���  
    vector<Rect> boundRect(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        boundRect[i] = boundingRect(contours[i]);
        rectangle(result, boundRect[i], Scalar(0, 255, 0), 2);//��result�ϻ�������Ӿ���  
    }

    imshow("Ч��ͼ", result);

 
    waitKey(0);
    return 0;
}