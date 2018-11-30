/**************************************************
*自动阈值的两种type, 多种blockSize, 多种C的效果
*两种type没多大的差别
*blockSize越大, 噪点颗粒越大
*C越大，二值化的阈值也就越小
*局部的阈值化操作，受到高斯干扰比较大
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
	createTrackbar("调整算法", WINDOW_NAME, &type, 1, on_Threshold);
	createTrackbar("像素邻域尺寸", WINDOW_NAME, &blockSize, 21, on_Threshold);
	createTrackbar("常数值", WINDOW_NAME, &C, 100, on_Threshold);
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
	blockSize = (blockSize << 1)+ 3;//大于等于3的奇数
	adaptiveThreshold(h_img, dst, 255, type, THRESH_BINARY, blockSize, C);

	imshow(WINDOW_NAME, dst);
}