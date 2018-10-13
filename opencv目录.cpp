1. HighGUI图像操作函数
	imread(const string& filename, int flags=-1/0/1/2/4)

	imshow(const string& winname, InputArray mat)

	InputArray【可以简单地当做Mat类型】

	namedWindow(const string& winname, int flags=WINDOW_NORMAL/WINDOW_AUTOSIZE/WINDOW_OPENGL)

	imwrite(const string& filename, InputArray img, const vector<int>& params=vector<int>())

	creatTrackbar(const string& trackbarname, const string& winname, int* value, int count, TrackbarCallback onChange=0, void* userdata=0)
	 |
	\ /
	creatTrackbar(轨迹条名，窗口名，滑块初始值，滑块最大值(最小始终为0)，回调函数指针，用户传给回调函数的数据)
																			  |
																			 \ /
																	void XXXX(int, void*)

	getTrackbarPos(const srting& trackbarname, const srting& winname)//获取当前轨迹条的位置并且返回

	setMouseCallback(const string& winname, MouseCallback onMouse, void* userdata=0)
													 |
												    \ /
					void XXXX(int event, int x, int y, int flags, void* param)//event为鼠标事件;x,y为鼠标的图像坐标系;flags是EVENT_FLAG组合;param是用户定义的传递参数






2. Core组件初级
	*Mat*:
		a.显式创建
		(1) Mat(int rows, int cols, int type, const Scalar& s);//Scalar的格式, CV_8UC3 -> CV_[位数][带符号与否][类型前缀]C[通道数]4
			-->Mat M(2, 2, CV_8UC3, Scalar(0,0,255))

		(2) Mat(维数，每个维数的尺寸，int type, const Scalar& s)

		(3) IplImage* img = ...
			Mat mtx(img)

		(4) M.create(4, 4, CV_8UC(2))//不能设定初值，只能作为改变尺寸开辟内存用

		(5) matlab式
				Mat E = Mat::eye(4, 4, CV_64F)

		(6)	小矩阵初始化
				Mat C = (Mat_<double>(3,3) << 1,2,3,4,5,6,7,8,9)

		(7) 使用 clone()或者 copyTo()
			Mat RowClone = C.row(1).clone()

		b.其他常用数据结构
			Point2f p(6,2)//二维点
			Point3f p(8,2,0)//三维点
			std::vector<char> v;//基于Mat的向量
			vector<Point2f> points(20);//存放二维点的vector容器

	*Point*:
		(1) Point point;//typedef Point_<int> Point2i; typedef Point2i Point;
			point.x = 10;
			point.y = 9;

		(2) Point point = Point(10,8);

	*Scalar*:
		(1)	Scalar(B, G, R, ALPHA)

	*Size类*:
		Size_(_Tp_width, _Tp_height)//typedef Size_<int> Size2i; typedef Size2i Size;
		--> Size(5, 5)

	*Rect*:
		Rect a(0, 10, 10, 20);//左上角的x,y, width, height
		std::cout << a.area() << a.size() << a.tl() << a.br();// 计算面积；长x宽； 左上角的点； 右上角的点

	*cvtColor()函数*：
		void cvtColor(InputArray src, OutputArray dst, int code, int dstCn = 0)//code为转换的标识符；dstCn为目标图的通道数，=0时候为原图通道数

	*基本图形绘制*：（imgproc.hpp中）
		椭圆：
		ellipse(InputOutputArray img, Point center, Size axes,
                        double angle, double startAngle, double endAngle,
                        const Scalar& color, int thickness = 1,
                        int lineType = LINE_8, int shift = 0);

		圆：
		void circle(InputOutputArray img, Point center, int radius,
                       const Scalar& color, int thickness = 1,
                       int lineType = LINE_8, int shift = 0);

		多边形：
		fillPoly(InputOutputArray img, InputArrayOfArrays pts,
                           const Scalar& color, int lineType = LINE_8, int shift = 0,
                           Point offset = Point() );//pts为多边形的点

		直线：
		line(InputOutputArray img, Point pt1, Point pt2, const Scalar& color,
                     int thickness = 1, int lineType = LINE_8, int shift = 0);//pt1为起点，pt2为终点







3. Core组件进阶
	*访问图像中的像素*
		1. 直接用指针对着每一个像素点进行处理（最快）
		2. 用迭代器 iterator 操作像素,只要在迭代指针前面加上 “ * ” 就可以访问数据
		3. 动态地址。区别于第一种，在于第一种得到的列数是 col*channels ,那么指针访问到的直接就是具体的颜色通道；而此种模式中，
			列数只是col，使用 Img.at<Vec3b>(i,j)[0] / [1] / [2] 来访问对应的BGR通道。

	*ROI*
		1. imageROI = img(Rect(500, 250, logo.cols, logo.rows))
		2. imageROI = img(Range(250,250+logoImage.rows),Range(500,500+logoImage.cols))

		补充copyTo()的知识点：
			image.copyTo(imageROI，mask),作用是把mask和image重叠以后把mask中像素值为0（black）的点对应的image中的点变为透明，而保留其他点。

	*addWeighted() 计算数组加权混合*
		void addWeighted(InputArray src1, double alpha, InputArray src2,
                              double beta, double gamma, OutputArray dst, int dtype = -1);
		-->第一张图，alpha1，第二张图，alpha2，在权重和后面加个gamma，输出数组，输出阵列深度

	*分离颜色通道，多通道混合*
		通道分离 split():
			void split(const Mat& src, Mat* mvbegin);
			void split(InputArray m, OutputArrayOfArrays mv);

			例子：
			mv[c](I) = src(I)c --> split(srcImage,channels);
									imageBlueChannels = channels.at(0);（！！！！注意此处用的.at是 “引用”！！！！）

		通道合并 merge():
			merge(const Mat* mv,size_tcount, OutputArray dst)
			merge(InputArrayOfArrays mv, OutputArray dst)

			例子：
				verctor<Mat> channels;
				merge(channels,mergeImage);

	*调整图像对比度、亮度*
		g(x) = a*f(x)+b --> a代表对比度，b代表明度

	*离散傅里叶变换*      (！！！！不大会！！！！！)
		dft():
			dft(InputArray src, OutputArray dst, int flags = 0, int nonzeroRows = 0)

		getOptimalDFTSize():
			int getOptimalDFTSize(int vecsize)
			返回DFT最优尺寸大小,vecsize为图片的rows/cols

		copyMakeBorder():扩充函数边界
			copyMakeBorder(InputArray src, OutputArray dst, int top, int bottom, int left, int right, int borderType, const Scalar& value=Scalar())
			——> 输入图，输出图，上，下，左，右扩充的像素，边界类型，默认吧
		
		magnitude():计算二维矢量的幅值
			magnitude(InputArray x, InputArray y, OutputArray magnitude)
			——> 实部，虚部，模

		log():计算自然对数
			log(InputArray src, OutputArray dst)

		normalize():矩阵归一化
		void normalize( InputArray src, InputOutputArray dst, double alpha = 1, double beta = 0,
                             int norm_type = NORM_L2, int dtype = -1, InputArray mask = noArray());
		——> 输入图，输出图，归一化后最大值(默认1)，归一化后最小值(默认0)，归一化类型，图像深度，掩膜





4. 图像处理（imgproc.hpp)
	*线性滤波*
		1.方框滤波
			void boxFilter( InputArray src, OutputArray dst, int ddepth,
	                             Size ksize, Point anchor = Point(-1,-1),
		                             bool normalize = true,
		                             int borderType = BORDER_DEFAULT );
			——> 输入图像，输出图像，输出图像深度，内核大小，锚点，内核是否被归一化，默认吧

		2.均值滤波（即归一化的线性滤波）
			void blur( InputArray src, OutputArray dst,
	                        Size ksize, Point anchor = Point(-1,-1),
	                        int borderType = BORDER_DEFAULT );

		3.高斯滤波
			GaussianBlur( InputArray src, OutputArray dst, Size ksize,
	                                double sigmaX, double sigmaY = 0,
	                                int borderType = BORDER_DEFAULT );
			——>sigmaX和sigmaY是两个方向上的标准偏差

	*非线性滤波*
		1.中值滤波
			void medianBlur(InputArray src, OutputArray dst, int ksize)
			——> ksize为孔的线性尺寸，为大于1的奇数

		2.双边滤波
			void bilateralFilter(InputArray src, OutputArray dst, int d, double sigmaColor, double sigmaSpace, int borderType=BORDER_DEFAULT)
			——> d>0时，表示每个像素的领域直径，而d<0时，正比于sigmaSpace；sigmaSpace越大，该像素领域中有越宽广的颜色会混到一起；sigmaSpace越大，越远的像素会被影响


	*形态学滤波(1)*
		1.膨胀
			void dilate( InputArray src, OutputArray dst, InputArray kernel,
                          Point anchor = Point(-1,-1), int iterations = 1,
                          int borderType = BORDER_CONSTANT,
                          const Scalar& borderValue = morphologyDefaultBorderValue() );


		2.腐蚀
			void erode( InputArray src, OutputArray dst, InputArray kernel,
                         Point anchor = Point(-1,-1), int iterations = 1,
                         int borderType = BORDER_CONSTANT,
                         const Scalar& borderValue = morphologyDefaultBorderValue() );

		3.配合函数 getStructuringElement()
			Mat getStructuringElement(int shape, Size ksize, Point anchor = Point(-1,-1));
			——> shape: 
			enum MorphShapes {
					    MORPH_RECT    = 0, //!< a rectangular structuring element:  \f[E_{ij}=1\f]
					    MORPH_CROSS   = 1, //!< a cross-shaped structuring element:
					                       //!< \f[E_{ij} =  \fork{1}{if i=\texttt{anchor.y} or j=\texttt{anchor.x}}{0}{otherwise}\f]
					    MORPH_ELLIPSE = 2 //!< an elliptic structuring element, that is, a filled ellipse inscribed
					                      //!< into the rectangle Rect(0, 0, esize.width, 0.esize.height)
					};

	*形态学滤波(2)*
		1.开运算: 先腐蚀后膨胀
			——>消除小物体，在纤细点分离物体，并且平滑较大物体的边界

		2.闭运算：先膨胀后腐蚀
			——>排除小型黑洞

		3.形态学梯度：膨胀图与腐蚀图之差
			——>将团块的边缘突出出来

		4.顶帽（top hat）：原图和开运算之差
			——>放大裂缝或者局部低亮部分

		5.黑帽： 闭运算与原图之差
			——>突出比原图轮廓周围的区域更暗的区域

		6.核心API void morphologyEx( InputArray src, OutputArray dst,
                                int op, InputArray kernel,
                                Point anchor = Point(-1,-1), int iterations = 1,
                                int borderType = BORDER_CONSTANT,
                                const Scalar& borderValue = morphologyDefaultBorderValue() );
		——>对于op:
				enum MorphTypes{
				    MORPH_ERODE    = 0, //!< see #erode
				    MORPH_DILATE   = 1, //!< see #dilate
				    MORPH_OPEN     = 2, //!< an opening operation
				                        //!< \f[\texttt{dst} = \mathrm{open} ( \texttt{src} , \texttt{element} )= \mathrm{dilate} ( \mathrm{erode} ( \texttt{src} , \texttt{element} ))\f]
				    MORPH_CLOSE    = 3, //!< a closing operation
				                        //!< \f[\texttt{dst} = \mathrm{close} ( \texttt{src} , \texttt{element} )= \mathrm{erode} ( \mathrm{dilate} ( \texttt{src} , \texttt{element} ))\f]
				    MORPH_GRADIENT = 4, //!< a morphological gradient
				                        //!< \f[\texttt{dst} = \mathrm{morph\_grad} ( \texttt{src} , \texttt{element} )= \mathrm{dilate} ( \texttt{src} , \texttt{element} )- \mathrm{erode} ( \texttt{src} , \texttt{element} )\f]
				    MORPH_TOPHAT   = 5, //!< "top hat"
				                        //!< \f[\texttt{dst} = \mathrm{tophat} ( \texttt{src} , \texttt{element} )= \texttt{src} - \mathrm{open} ( \texttt{src} , \texttt{element} )\f]
				    MORPH_BLACKHAT = 6, //!< "black hat"
				                        //!< \f[\texttt{dst} = \mathrm{blackhat} ( \texttt{src} , \texttt{element} )= \mathrm{close} ( \texttt{src} , \texttt{element} )- \texttt{src}\f]
				    MORPH_HITMISS  = 7  //!< "hit or miss"
				                        //!<   .- Only supported for CV_8UC1 binary images. A tutorial can be found in the documentation
				};


	*漫水填充*（比较复杂可以参考PDF）
		无掩膜：
		int floodFill( InputOutputArray image,
                          Point seedPoint, Scalar newVal, CV_OUT Rect* rect = 0,
                          Scalar loDiff = Scalar(), Scalar upDiff = Scalar(),
                          int flags = 4 );

		带掩膜：
		int floodFill( InputOutputArray image, InputOutputArray mask,
                            Point seedPoint, Scalar newVal, CV_OUT Rect* rect=0,
                            Scalar loDiff = Scalar(), Scalar upDiff = Scalar(),
                            int flags = 4 );

	*图像金字塔*（实现图像的尺寸收缩）
		1. 尺寸调整 resize()
			void resize( InputArray src, OutputArray dst,
                          Size dsize, double fx = 0, double fy = 0,
                          int interpolation = INTER_LINEAR );
			——> dsize为输出图像的大小;fx为水平轴的缩放系数；fy为垂直轴的缩放系数；interpolation为插值方法
																		——>/** nearest neighbor interpolation */
																		    INTER_NEAREST        = 0,
																		    /** bilinear interpolation */
																		    INTER_LINEAR         = 1,
																		    /** bicubic interpolation */
																		    INTER_CUBIC          = 2,
																		    /** resampling using pixel area relation. It may be a preferred method for image decimation, as
																		    it gives moire'-free results. But when the image is zoomed, it is similar to the INTER_NEAREST
																		    method. */
																		    INTER_AREA           = 3,
																		    /** Lanczos interpolation over 8x8 neighborhood */
																		    INTER_LANCZOS4       = 4,
																		    /** Bit exact bilinear interpolation */
																		    INTER_LINEAR_EXACT = 5,
																		    /** mask for interpolation codes */
																		    INTER_MAX            = 7,
																		    /** flag, fills all of the destination image pixels. If some of them correspond to outliers in the
																		    source image, they are set to zero */
																		    WARP_FILL_OUTLIERS   = 8,
			
			例子：	resize(srcImage, dstImage, dstImage.size())
					resize(srcImage, dstImage, Size(), 0.5, 0.5)

		2. 向上采样 pyrUp()
			void pyrUp( InputArray src, OutputArray dst,
                         const Size& dstsize = Size(), int borderType = BORDER_DEFAULT );
			——> Size()默认时，则长宽为放大两倍

		3. 向下采样 pyrDown()
			void pyrDown( InputArray src, OutputArray dst,
                           const Size& dstsize = Size(), int borderType = BORDER_DEFAULT );


	*阈值化*（PS的色彩范围）
		1.固定阈值操作
		double threshold( InputArray src, OutputArray dst,
                               double thresh, double maxval, int type );
		——> thresh表示阈值的具体值；maxval为在type取THRESH_BINARY和THRESH_BINARY_INV时候的阈值最大值。

		2.自适应阈值操作
		void adaptiveThreshold( InputArray src, OutputArray dst,
                                     double maxValue, int adaptiveMethod,
                                     int thresholdType, int blockSize, double C );
		——>adaptiveMethod 为自适应阈值算法（ADAPTIVE_THRESH_MEAN_C / ADAPTIVE_THRESH_GAUSSIAN_C);
			thresholdType 为阈值类型（THRESH_BINARY / THRESH_BINARY_INV)
			blockSize 为计算阈值大小的一个像素的领域尺寸（3 、5 、7）
			C 正数，在平均值上减去用