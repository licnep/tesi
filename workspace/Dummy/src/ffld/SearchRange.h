#ifndef _SEARCH_RANGE
#define _SEARCH_RANGE

#include <vector>
#include <Devices/Camera/CCamera.h>
#include <Processing/Vision/PerspectiveMapping/HomographicTransformations.h>
#include <Processing/Vision/PerspectiveMapping/ipm.h>
#include <Data/Math/Rects.h>
#include <boost/graph/graph_concepts.hpp>
#include <Data/CImage/Images/CImageRGB8.h>


class SearchRange {
public:
	std::vector<std::pair<int,int> > m_ranges;
	void setSearchRange(int width, int height, dev::CCamera *camera, cimage::CImageRGB8 & debugImage, float w0 = 0.2f, float w1 = 1.0f);
	void SearchRange::draw(cimage::CImageRGB8 & debugImage);
	bool SearchRange::isPlausibleSize(int lineFromTop, int width);
	std::pair<int,int> getUsefulLineRange(int width);

private:
	void Size(std::vector<std::pair<int,int> > & ranges, const dev::CameraParams & cameraParams, double w0, double w1, double z0, double z1, int min_width, int max_width, double max_distance, unsigned int height, double border);
};

#endif
