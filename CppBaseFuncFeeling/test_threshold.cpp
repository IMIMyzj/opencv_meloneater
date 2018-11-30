/********************************************************
*单阈值总是不好达到理想的分割效果
*导入原图像质量越高，颗粒越细腻
********************************************************/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;

#define WINDOW_NAME "【程序窗口】"        //为窗口标题定义的宏 

int g_nThresholdValue = 100;
int g_nThresholdType = 0;
Mat g_srcImage, g_hsvImage, g_dstImage, g_hImage;
vector<Mat> channels;

void on_Threshold( int, void* );//回调函数


int main( )
{
	g_srcImage = imread("./part1/data_20181117_211428.jpg");
	if(!g_srcImage.data ) { printf("读取图片错误，请确定目录下是否有imread函数指定的图片存在~！ \n"); return false; }  
	namedWindow("原始图", WINDOW_NORMAL);
	imshow("原始图",g_srcImage);

	cvtColor( g_srcImage, g_hsvImage, COLOR_RGB2HSV );
	split(g_hsvImage, channels);
	g_hImage = channels.at(0);

	namedWindow( WINDOW_NAME, WINDOW_NORMAL );
	
	createTrackbar("模式", WINDOW_NAME, &g_nThresholdType, 4, on_Threshold);

	createTrackbar("参数值", WINDOW_NAME, &g_nThresholdValue, 255, on_Threshold);

	on_Threshold( 0, 0 );

	while(1)
	{
		int key;
		key = waitKey( 20 );
		if( (char)key == 27 ){ break; }
	}

}

void on_Threshold( int, void* )
{
	//调用阈值函数
	threshold(g_hImage,g_dstImage,g_nThresholdValue,255,g_nThresholdType);

	//更新效果图
	imshow( WINDOW_NAME, g_dstImage );
}
