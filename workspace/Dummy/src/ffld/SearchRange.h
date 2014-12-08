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
	void setSearchRange(int width, int height, dev::CCamera *camera, cimage::CImageRGB8 & debugImage);
	void Size(std::vector<std::pair<int,int> > & ranges, const dev::CameraParams & cameraParams, double w0, double w1, double z0, double z1, int min_width, int max_width, double max_distance, unsigned int height, double border);
};

#endif
