/**************************************************
*�Զ���ֵ������type, ����blockSize, ����C��Ч��
*����typeû���Ĳ��
*blockSizeԽ��, ������Խ��
*CԽ�󣬶�ֵ������ֵҲ��ԽС
*�ֲ�����ֵ���������ܵ���˹���űȽϴ�
***************************************************/

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define WINDOW_NAME "threshold"

int type = 0;
int blockSize = 0;
int C = 0;
Mat HSV, dst, h_img;
vector<Mat> channels;

void on_Threshold(int, void*);


int main( )
{
	Mat src;
	src = imread("./part1/daguang.jpg",1);
	namedWindow("src", WINDOW_NORMAL);
	imshow("src", src);

	cvtColor(src, HSV, CV_BGR2HSV);
	split(HSV, channels);
	h_img = channels.at(0);
	

	namedWindow(WINDOW_NAME, WINDOW_NORMAL);
	createTrackbar("�����㷨", WINDOW_NAME, &type, 1, on_Threshold);
	createTrackbar("��������ߴ�", WINDOW_NAME, &blockSize, 21, on_Threshold);
	createTrackbar("����ֵ", WINDOW_NAME, &C, 100, on_Threshold);
	on_Threshold(0, 0);
	

	while (1) {
		int key;
		key = waitKey(20);
		if ((char)key == 27) {
			destroyAllWindows();
			break;
		}
	}
}

void on_Threshold(int, void*) {
	blockSize = (blockSize << 1)+ 3;//���ڵ���3������
	adaptiveThreshold(h_img, dst, 255, type, THRESH_BINARY, blockSize, C);

	imshow(WINDOW_NAME, dst);
}