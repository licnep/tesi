/**********************************************************************
 *                                                                    *
 *  This file is part of the GOLD software                            *
 *                                                                    *
 *            University of Parma, Italy   1996-2011                  *
 *                                                                    *
 *       http://www.vislab.it                                         *
 *                                                                    *
 **********************************************************************/


/**
 * \author VisLab (vislab@ce.unipr.it)
 * \date 2006-04-27
 */

#include "dummy.h"

#include <Devices/Camera/CCamera.h>
#include <Devices/Clock/CClock.h>
#include <Devices/Profiler/Profiler.h>
#include <Framework/CRecordingCtl.h>
#include <Framework/Transport.h>
#include <Processing/Vision/CImage/BasicOperations/BasicOperations.h>
#include <Processing/Vision/CImage/Conversions/CImageConversions.h>
#include <Processing/Vision/CImage/Filters/SobelFilter.h>

#define __LOCATION__ __CLASS_METHOD__
// #define __LOCATION__ __FILE_LINE__
// #define VERBOSITY_DEBUG true
#include <Libs/Logger/Log.h>

#include <boost/date_time/posix_time/posix_time.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/objdetect.hpp>
//#include <opencv2/core/cuda.hpp>
//#include <opencv2/ml/ml.hpp>
//#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/objdetect/objdetect.hpp"*/
#include "HOGVisualizer.h"

using namespace cimage;
using namespace cimage::filter;
using namespace ui::conf;
using namespace ui::var;
using namespace ui::wgt;
using namespace ui::win;
using namespace cv;
using namespace cv::ml;
using namespace std;

void CDummy::On_Initialization()
{    
    // recuperiamo l'elenco delle camere...
    CDeviceNode& cameras = Dev()["CAMERAS"];

    // ...e poi inseriamo in m_cameraNames i nomi di ognuna
    std::transform(cameras.Children().begin(), cameras.Children().end(),
                   std::back_inserter(m_cameraNames), std::mem_fun(&CDeviceNode::Name));

    Configuration config(Options());
    
    // dichiariamo una variabile di tipo stringa associata al dato membro m_inputCameraName...
    Choice<std::string> inputCameraName(&m_inputCameraName, m_cameraNames.begin(), m_cameraNames.end());
    // ...e la colleghiamo alla chiave CAMERA di Dummy.ini; se la chiave CAMERA non esiste usiamo m_cameraNames[0] come default
    config.Bind(inputCameraName, "CAMERA", m_cameraNames[0]);

    // come sopra, si caricano alcuni parametri da file di configurazione    
    Value<unsigned int> width(&m_width);
    config.Bind(width, "WIDTH", 0);
    
    Value<unsigned int> height(&m_height);
    config.Bind(height, "HEIGHT", 0);
    
    Value<bool> showInputMono(&m_showInputMono);
    config.Bind(showInputMono, "SHOW INPUT MONO", false);
    
    Value<bool> showInputRGB(&m_showInputRGB);
    config.Bind(showInputRGB, "SHOW INPUT RGB", false);
    
    Value<bool> showSobel(&m_showSobel);
    config.Bind(showSobel, "SHOW SOBEL", true);
    
    ui::var::Range<float> radius(&m_radius, 0.0f, 50.0f, 0.1f);
    config.Bind(radius, "RADIUS", 10.0f);
    
    ui::var::Range<int> value(&m_value, 0, 150, 2);
    config.Bind(value, "VALUE", 40);
    
    Value<bool> showCircle(&m_showCircle);
    config.Bind(showCircle, "SHOW CIRCLE", false);
    
    Value<bool> showBox(&m_showBox);
    config.Bind(showBox, "SHOW BOX", false);
    
    Value<bool> showText(&m_showText);
    config.Bind(showText, "SHOW TEXT", false);

    Map<int> features(&m_selectedFeature,
                      std::make_pair("First", 0),
                      std::make_pair("Second", 1),
                      std::make_pair("Third", 2));
    
    //popoliamo il pannello dell'applicazione
    panel.Label("Dummy Main Panel").Geometry(300, 150)
    (
        VSizer()
        (
            NoteBook()
            (
                Page("Basic")
                (
                    VSizer()
                    (
                        CheckBox(showInputMono, "Show input mono"),
                        CheckBox(showInputRGB, "Show input RGB"),
                        CheckBox(showSobel, "Show input Sobel")
                    )
                ),
                Page("Advanced")
                (
                    VSizer()
                    (
                        Slider(radius, "Radius"),
                        Slider(value, "Value"),
                        CheckBox(showCircle, "Show circle"),
                        CheckBox(showBox, "Show box"),
                        CheckBox(showText, "Show text"),
                        ComboBox(features, "Features")
                    )
                )
            )
        )
    );
    
    // configuriamo il Synchronizer in modo che ritorni sempre l'ultimo frame ricevuto dalla camera indicata
    m_pCam = (Dev()["CAMERAS/" + m_inputCameraName]);
    m_synchro.ConnectSync(*m_pCam);
    
    // inizializziamo alcune variabili
    m_pInputMonoWindow = NULL;
    m_pInputRGBWindow = NULL;
    m_pSobelWindow = NULL;
}

void CDummy::On_ShutDown()
{
    delete m_pInputMonoWindow;
    delete m_pInputRGBWindow;
    delete m_pSobelWindow;
}

void CDummy::On_Execute()
{
    CImage::SharedPtrConstType image;
    
    {

        // copiamo lo shared pointer al frame, che utilizzeremo in seguito per l'elaborazione
        image = m_synchro.SyncFrameFrom<dev::CCamera>(*m_pCam).Data;
        
        log_debug << " Processing frame: " << m_synchro.LastFrameFrom<dev::CCamera>(*m_pCam).TimeStamp << std::endl;
    }

    // se nel file INI non compaiono WIDTH o HEIGHT le corrispondenti variabili membro di Dummy m_width e m_height hanno assunto
    // il valore di default specificato nella Bind, cioè 0, a cui sostituiamo la risoluzione del frame corrente
    if(!m_width)
        m_width = image->W();
    if(!m_height)
        m_height = image->H();

    // per semplicità eseguiamo la Resize comunque: se m_width e m_height sono già corrette non succede nulla
    Resize(m_inputImageMono, m_width, m_height);
    Resize(m_inputImageRGB, m_width, m_height);
    Resize(m_sobelImage, m_width, m_height);

    // convertiamo il frame in una immagine a colori
    Convert(*image, m_inputImageRGB, BAYER_DECODING_SIMPLE);

    // convertiamo il frame in una immagine in bianco e nero
    Convert(*image, m_inputImageMono, BAYER_DECODING_LUMINANCE);

    // applichiamo un filtro che converte una immagine CImageMono in una CImageMono usando un kernel SobelVertical di dimensione 3x3
    SobelVertical3x3(m_inputImageMono, m_sobelImage);
    //cv::Mat img(m_width,m_height,CV_8UC1,m_inputImageMono.Buffer());
    /*for (int i=0;i<m_width*m_height/2;i++) {
		img.data[i] = img.data[i]+50;
		//m_sobelImage.Buffer()[i] = 0;
	}
	// = Mat::ones(50,50,CV_8UC1);
    //Mono8* buffer = m_sobelImage.Buffer();
	*/
    //m_sobelImage.New(50,50);

    Mat m = CHOGVisualizer::CImageRGB8ToMat(m_inputImageRGB);
    RGB8* data = m_inputImageRGB.Buffer();
    for (int i=0;i<m_width*m_height;i++) {
    	data[i].R = 0;
    	data[i].G = 0;
    	data[i].B = 0;
	}

	std::vector<float> descriptorsValues;
	std::vector<cv::Point> locations;
	resize(m, m, Size(512,256) );
	Mat img;
	cvtColor(m, img, COLOR_BGR2GRAY);
	img.convertTo(img,CV_8U);
	imwrite( "./butta.jpg", img );
	HOGDescriptor d(Size(128,64), Size(16,16), Size(8,8),Size(8,8), 9);
	d.compute(img, descriptorsValues, Size(8,8), Size(0,0), locations);
	Mat viz = CHOGVisualizer::GetHogDescriptorVisu(m,descriptorsValues,Size(128,64) /*Size(512,256)*/);
	viz.convertTo(viz,CV_16UC3);

	resize(viz,viz,Size(m_inputImageRGB.W(),m_inputImageRGB.H()));
	//resize(img,img,Size(m_inputImageRGB.W(),m_inputImageRGB.H()));
	imwrite( "./butta2.jpg", viz );
	CHOGVisualizer::MatToCImageRGB8(viz,m_inputImageRGB);
	/*
	Mat img;
	cvtColor(m, img, COLOR_BGR2GRAY);
	img.convertTo(img,CV_8U);
	cout << "type:" << img.type() << endl;
	d.compute(img, descriptorsValues, Size(8,8), Size(0,0), locations);
	Mat viz = CHOGVisualizer::GetHOGDescriptorVisualImage(img,descriptorsValues,Size(512,256),Size(8,8),4,3);

	viz = CHOGVisualizer::GetHogDescriptorVisu(img,descriptorsValues,Size(512,256));

	resize(viz,viz,Size(m_height,m_width));
	cout << "type2:" << viz.type() << endl;
	//m.convertTo(m,CV_8U);
	//cvtColor(viz,viz,COLOR_GRAY2BGR);
	viz.convertTo(viz,CV_16UC3);
	CHOGVisualizer::MatToCImageRGB8(viz,m_inputImageRGB);*/

    /*HOGDescriptor d;
    std::vector<float> descriptorsValues;
    std::vector<cv::Point> locations;
    d.compute(img, descriptorsValues, Size(8,8), Size(0,0), locations);

    cout << "HOG descriptor size is " << d.getDescriptorSize() << endl;
    cout << "img dimensions: " << img.cols << " width x " << img.rows << "height" << endl;
    cout << "Found " << descriptorsValues.size() << " descriptor values" << endl;
    cout << "Nr of locations specified : " << locations.size() << endl;
    cout << "IMG cols: " << img.cols << "  rows: " << img.rows << endl;

    Mat viz = CHOGVisualizer::GetHOGDescriptorVisualImage(img,descriptorsValues,Size(m_width,m_height),Size(8,8),1,1);
    cout << "VIZ cols: " << viz.cols << "rows: " << viz.rows << endl;
    cv::Mat img2(m_width,m_height,CV_8UC3,m_inputImageRGB.Buffer());
    Mat ma(Size(m_width,m_height),CV_8UC1);
    for (int x=0;x<m_width*m_height;x++) {
    	ma.data[x] = m_inputImageRGB.Buffer()[x].R;
    }
    rectangle(ma,Point(0,0),Point(100,50),Scalar(0,0,255));
    for (int x=0;x<m_width*m_height;x++) {
        m_inputImageRGB.Buffer()[x].R = ma.data[x];
	}*/
    //img = Scalar(0,0,0);
    //viz.convertTo(img,CV_8UC1);

    //addWeighted( viz, 1.0, img, 0.2, 0.0, img);
    //img += viz;

    // chiamiamo la funzione di disegno
    Output();
}



void CDummy::Output()
{
    // da pannello e' abilitato il disegno della finestra?
    if(m_showInputMono)
    {   
        // non e' ancora stata creata? la creiamo
        if(m_pInputMonoWindow == NULL)
            m_pInputMonoWindow = new CWindow ( "Input Mono", m_width, m_height);

        // cancelliamo le primitive di disegno
        m_pInputMonoWindow->Clear();

        // disegniamo una immagine CImageMono
        m_pInputMonoWindow->DrawImage(m_inputImageMono);

        if(m_showBox)
        {
            m_pInputMonoWindow->SetColor(255, 0, 0);
            m_pInputMonoWindow->DrawRectangle(15, 20, 50, 40);
            m_pInputMonoWindow->SetColor(255, 0, 255);
            m_pInputMonoWindow->DrawRectangle(50, 20, 90, 40);
        }

        if (m_showCircle)
        {
            m_pInputMonoWindow->SetColor(255, 255, 0);
            m_pInputMonoWindow->DrawCircle(m_value, m_value, m_radius, false);
        }

        if (m_showText)
        {
            std::ostringstream oss;
            oss << m_value;

            m_pInputMonoWindow->SetFontName("arial");
            m_pInputMonoWindow->SetFontSize(12);
            m_pInputMonoWindow->DrawText(8, 10, oss.str());
        }

        // in fondo a tutto, eseguiamo la Refresh sulla finestra
        m_pInputMonoWindow->Refresh();

        // rendiamo la finestra visibile
        if(!m_pInputMonoWindow->IsVisible())
            m_pInputMonoWindow->Show();
    } else if(m_pInputMonoWindow)
        m_pInputMonoWindow->Hide();

    if(m_showInputRGB)
    {
        if(m_pInputRGBWindow == NULL)
            m_pInputRGBWindow = new CWindow("Input RGB", m_width, m_height);

        m_pInputRGBWindow->Clear();
        m_pInputRGBWindow->DrawImage(m_inputImageRGB);
        m_pInputRGBWindow->Refresh();

        if(!m_pInputRGBWindow->IsVisible())
            m_pInputRGBWindow->Show();
    } else if(m_pInputRGBWindow)
        m_pInputRGBWindow->Hide();

    if(m_showSobel)
    {
        if(m_pSobelWindow == NULL)
            m_pSobelWindow = new CWindow("Sobel", m_width, m_height);

        m_pSobelWindow->Clear();
        m_pSobelWindow->DrawImage(m_sobelImage);
        m_pSobelWindow->Refresh();

        if(!m_pSobelWindow->IsVisible())
            m_pSobelWindow->Show();
    } else if(m_pSobelWindow)
        m_pSobelWindow->Hide();
}

#include <Framework/Application_Registration.h>

// registriamo la classe CDummy col nome Dummy
REGISTER_APPLICATION(CDummy, "Dummy");
