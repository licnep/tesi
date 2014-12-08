/*#include <vector>
#include <Devices/Camera/CCamera.h>
#include <Processing/Vision/PerspectiveMapping/HomographicTransformations.h>
#include <Processing/Vision/PerspectiveMapping/ipm.h>
#include <Data/Math/Rects.h>
#include <boost/graph/graph_concepts.hpp>
#include <Data/CImage/Images/CImageRGB8.h>*/
#include <SearchRange.h>


void SearchRange::setSearchRange(int width, int height, dev::CCamera *camera,cimage::CImageRGB8 & debugImage) {
	std::vector<std::pair<int,int> > ranges;

	//std::string range_algo = INIFile()->Value<std::string>("RANGE MODE", "LINES"); // LINES, STATIC or SIZE

	int max_width = width/2;  //INIFile()->Value<int>("MAX WIDTH", width/2);
	int min_width =  20;  //std::max(m_cls[0]->Width(), m_cls[1]->Width() ); // INIFile()->Value<int>("MIN WIDTH", 32);

	//if (range_algo == "SIZE")

	double max_distance = 1000.0;//INIFile()->Value<double>("MAX DISTANCE", 1000.0);

	double w0, w1, z0, z1;

	w0 = 0.2;   //INIFile()->Value<double>("W0");
	w1 = 1.0; //INIFile()->Value<double>("W1");
	z0 = 0.0;   //INIFile()->Value<double>("Z0");
	z1 = 0.0; //INIFile()->Value<double>("Z1");

	/// parametri camera originali
	dev::CameraParams m_cameraParams0 = camera->Params();
	//parametri camera dopo la rettificazione
	dev::CameraParams m_cameraParams = m_cameraParams0;
	m_cameraParams.roll = 0.0;
	ht::CameraRectify(m_cameraParams, m_cameraParams0);



	//std::cout << "cameraParams::::::" << m_cameraParams0 << std::endl;
	//std::cout << "Rettificati::::::" << m_cameraParams << std::endl;

	double m_border = 0.0;
	this->Size(ranges, m_cameraParams0, w0, w1, z0,z1, min_width, max_width, max_distance, height, m_border);

	cimage::RGB8* dstBuffer = debugImage.Buffer();

	for (int i=0;i<ranges.size();i++) {
		//std::cout << "first:" << ranges[i].first << " second:" << ranges[i].second << std::endl;

		dstBuffer[i*(width)+ranges[i].first].B = 255;
		dstBuffer[i*(width)+ranges[i].second].R = 255;
	}
}


void SearchRange::Size(std::vector<std::pair<int,int> > & ranges, const dev::CameraParams & cameraParams, double w0, double w1, double z0, double z1, int min_width, int max_width, double max_distance, unsigned int height, double border)
{
	PerspectiveMapping pm(cameraParams);
    InversePerspectiveMapping ipm(cameraParams);

    double u0 = cameraParams.u0;

    std::cout << "U0: " << u0 << std::endl;

    // INITIALIZE RANGE
    for(unsigned int i=0; i<height; ++i)
        ranges.push_back(std::pair<int,int>(max_width,0));

    // FOR ANY WINDOW WIDTH
    for(int w=min_width; w<max_width; w++)
    {
        // width = real_width * (1 + 2 * alpha)
        double rw = w / (1 + 2.0 * border); // vera dimensione dell'oggetto: 2 volte bordo
        double brd = rw * border; // bordo di questa finestra, offset sulla coordinata verticale

        // devo trovare in quale riga tale width e' valida
        for(int i=0; i<(int) height; ++i)
        {
            // per ogni riga dell'immagine
            double row = i - brd; // riga vera della base dell'oggetto
            bool valid = false;
            // calcolo alla riga row, le coordinate mondo papabili data z0 e z1, al centro immagine, grande rw pixels
            math::Point3d l0 = ipm.WorldFromPixelZ(u0-rw*0.5, row, z0);
            math::Point3d r0 = ipm.WorldFromPixelZ(u0+rw*0.5, row, z0);
            double tw = l0.y - r0.y; // WIDTH in metri

            if((l0.x>0.0) && (l0.x < max_distance) && (tw >= w0) && (tw <= w1))
            {
                // siamo sotto l'"orizzonte" e davanti al veicolo. La dimensione del veicolo e' comapatibile con le soglie w0 e w1
                valid = true;
            }

            // nel caso in cui z1 != z0, eseguo il controllo anche a z1
            if(z1!=z0)
            {
                math::Point3d l1 = ipm.WorldFromPixelZ(u0-rw*0.5, row, z1);
                math::Point3d r1 = ipm.WorldFromPixelZ(u0+rw*0.5, row, z1);

                if(l1.x>0.0 && l1.x < max_distance)
                {
                    // siamo sotto l'"orizzonte"
                    double tw = l1.y - r1.y; // WIDTH in metri
                    if(tw >= w0 && tw <= w1)
                        valid = true;
                }
            }

            if(valid)
            {
                if(ranges[i].first>w)
                    ranges[i].first = w;
                if(ranges[i].second<w)
                    ranges[i].second = w;
            }

        } // <i

    } // <w

}
