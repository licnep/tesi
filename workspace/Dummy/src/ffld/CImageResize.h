#ifndef _CIMAGERESIZE_H
#define _CIMAGERESIZE_H

#include <Data/CImage/Images/CImageRGB8.h>

//private version of the resize function, works directly on buffers, for any resolution.
//Applies bilinear interpolation
void CImageResize_(const cimage::RGB8* src, int srcWidth, int srcHeight, cimage::RGB8* dstBuffer,
					   int dstWidth, int dstHeight);

//Public version of the resize function. It resizes the image by consecutive halving, using bilinear interpolation
//then applies an extra resize if needed to reach the defined width and height
cimage::CImageRGB8 CImageResize(cimage::CImageRGB8 &srcImage, int width, int height);

#endif
