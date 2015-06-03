#include "CImageResize.h"
#include <Data/CImage/Images/CImageRGB8.h>
#include <vector>

using namespace std;


cimage::RGB8 operator*(cimage::RGB8 a, float const& n) {
	a.R *= n;
	a.G *= n;
	a.B *= n;
	return a;
}

// Bilinear interpolation coefficient
namespace FFLD
{
namespace detail
{
struct Bilinear
{
	int x0;
	int x1;
	float a;
	float b;
};
}
}

//private version of the resize function, works directly on buffers, for any resolution.
//Applies bilinear interpolation
void CImageResize_(const cimage::RGB8* src, int srcWidth, int srcHeight, cimage::RGB8* dstBuffer,
					   int dstWidth, int dstHeight)
{
	if ((srcWidth == dstWidth) && (srcHeight == dstHeight)) {
		std::cout << "RESIZE SAME SIZE, doing nothing" << std::endl;
		return;
	}

	const float xScale = static_cast<float>(srcWidth) / dstWidth;
	const float yScale = static_cast<float>(srcHeight) / dstHeight;

	// Bilinear interpolation coefficients
	std::vector<FFLD::detail::Bilinear> cols(dstWidth);

	for (int j = 0; j < dstWidth; ++j) {
		const float x = min(max((j + 0.5f) * xScale - 0.5f, 0.0f), srcWidth - 1.0f);
		cols[j].x0 = x;
		cols[j].x1 = min(cols[j].x0 + 1, srcWidth - 1);
		cols[j].a = x - cols[j].x0;
		cols[j].b = 1.0f - cols[j].a;
	}

	for (int i = 0; i < dstHeight; ++i) {
		const float y = min(max((i + 0.5f) * yScale - 0.5f, 0.0f), srcHeight - 1.0f);
		const int y0 = y;
		const int y1 = min(y0 + 1, srcHeight - 1);
		const float c = y - y0;
		const float d = 1.0f - c;

		for (int j = 0; j < dstWidth; ++j) {

			dstBuffer[i*dstWidth + j] =
					(src[(y0 * srcWidth + cols[j].x0)] * cols[j].b +
					src[(y0 * srcWidth + cols[j].x1)] * cols[j].a) * d +
					(src[(y1 * srcWidth + cols[j].x0)] * cols[j].b +
					src[(y1 * srcWidth + cols[j].x1)] * cols[j].a) * c; //+ 0.5f;

			/*

			for (int k = 0; k < depth; ++k) {

				dst[(i * dstWidth + j) * depth + k] =
					(src[(y0 * srcWidth + cols[j].x0) * depth + k] * cols[j].b +
					 src[(y0 * srcWidth + cols[j].x1) * depth + k] * cols[j].a) * d +
					(src[(y1 * srcWidth + cols[j].x0) * depth + k] * cols[j].b +
					 src[(y1 * srcWidth + cols[j].x1) * depth + k] * cols[j].a) * c + 0.5f;
			}*/
		}
	}
}

typedef boost::shared_ptr<cimage::CImageRGB8> sharedCimagePtr;

//Public version of the resize function. It resizes the image by consecutive halving, using bilinear interpolation
//then applies an extra resize if needed to reach the defined width and height
cimage::CImageRGB8 CImageResize(cimage::CImageRGB8 &srcImage, int width, int height) {
	//Empty image
	if ((width <= 0) || (height <= 0)) {
		return cimage::CImageRGB8(0,0);
		//sharedCimagePtr result(new cimage::CImageRGB8(0,0));
		//return result;
	}

	int width_ = srcImage.W();
	int height_ = srcImage.H();

	// Same dimensions
	if ((width == width_) && (height == height_)) {
		return srcImage;
	}

	//allocate a new cImage
	//boost::shared_ptr<cimage::CImageRGB8> result(new cimage::CImageRGB8(width,height));
	cimage::CImageRGB8 result(width,height);

	//Resize the image at each octave
	int srcWidth = width_;
	int srcHeight = height_;

	//vector<uint8_t> tmpSrc;
	//vector<uint8_t> tmpDst;

	std::vector<cimage::RGB8> tmpSrc;
	std::vector<cimage::RGB8> tmpDst;

	float scale = 0.5f;
	int halfWidth = width_ * scale + 0.5f;
	int halfHeight = height_ * scale + 0.5f;

	/*while ((width <= halfWidth) && (height <= halfHeight)) {
		if (tmpDst.empty())
			tmpDst.resize(halfWidth * halfHeight);

		//CImageResize_(tmpSrc.empty() ? &bits_[0] : &tmpSrc[0], srcWidth, srcHeight, &tmpDst[0], halfWidth, halfHeight, depth_);
		CImageResize_(tmpSrc.empty() ? srcImage.Buffer() : &tmpSrc[0], srcWidth, srcHeight, &tmpDst[0], halfWidth, halfHeight);

		// Dst becomes src
		tmpSrc.swap(tmpDst);
		srcWidth = halfWidth;
		srcHeight = halfHeight;

		// Next octave
		scale *= 0.5f;
		halfWidth = width_ * scale + 0.5f;
		halfHeight = height_ * scale + 0.5f;
	}*/

	CImageResize_(tmpSrc.empty() ? srcImage.Buffer() : &tmpSrc[0], srcWidth, srcHeight, result.Buffer(), width, height);

	return result;
}

