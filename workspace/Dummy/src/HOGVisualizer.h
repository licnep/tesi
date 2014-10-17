#ifndef _HOGVISUALIZER_H
#define _HOGVISUALIZER_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Data/CImage/Images/CImageRGB8.h>

class CHOGVisualizer {
	public:
		static cv::Mat GetHOGDescriptorVisualImage(cv::Mat&,
                                                   std::vector<float>&,
                                                   cv::Size,
                                                   cv::Size,
                                                   int,
                                                   double);
		static cv::Mat GetHogDescriptorVisu(const cv::Mat&, std::vector<float>&, const cv::Size&);
		static cv::Mat CImageRGB8ToMat(cimage::CImageRGB8&);
		static void MatToCImageRGB8(cv::Mat&,cimage::CImageRGB8&);

};


#endif
