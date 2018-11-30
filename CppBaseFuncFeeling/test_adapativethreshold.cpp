#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;

#define WINDOW_NAME "�����򴰿ڡ�"        //Ϊ���ڱ��ⶨ��ĺ� 

int g_nThresholdValue = 100;
int g_nThresholdType = 3;
Mat g_srcImage, g_hsvImage, g_dstImage, g_hImage;
vector<Mat> channels;

void on_Threshold( int, void* );//�ص�����


int main( )
{
	g_srcImage = imread("./part1/1.jpg");
	if(!g_srcImage.data ) { printf("��ȡͼƬ������ȷ��Ŀ¼���Ƿ���imread����ָ����ͼƬ����~�� \n"); return false; }  
	namedWindow("ԭʼͼ", WINDOW_NORMAL);
	imshow("ԭʼͼ",g_srcImage);

	cvtColor( g_srcImage, g_hsvImage, COLOR_RGB2HSV );
	split(g_hsvImage, channels);
	g_hImage = channels.at(0);

	namedWindow( WINDOW_NAME, WINDOW_NORMAL );
	
	createTrackbar("ģʽ", WINDOW_NAME, &g_nThresholdType, 4, on_Threshold);

	createTrackbar("����ֵ", WINDOW_NAME, &g_nThresholdValue, 255, on_Threshold);

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
	//������ֵ����
	threshold(g_hImage,g_dstImage,g_nThresholdValue,255,g_nThresholdType);

	//����Ч��ͼ
	imshow( WINDOW_NAME, g_dstImage );
}
