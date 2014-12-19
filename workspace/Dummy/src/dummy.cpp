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

//using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "HOGVisualizer.h"
#include "ffld/ffld.h"
#include "ffld/JPEGImage.h"
#include "ffld/SearchRange.h"

#include <boost/date_time/posix_time/posix_time.hpp>

#include <UI/Panel/detail/PanelTypes.h>
#include <UI/Panel/detail/WidgetCore.h>
#include <UI/Panel/detail/PanelListener.h>
#include <UI/Panel/Panel.h>

using namespace cimage;
using namespace cimage::filter;
using namespace ui::conf;
using namespace ui::var;
using namespace ui::wgt;
using namespace ui::win;
using namespace cv;
//using namespace cv::ml;
using namespace std;

class CMyPanelListener : public ui::detail::PanelListener {
public:
	CMyPanelListener(CDummy * dummy) {
		mDummy = dummy;
	};

	/// Quando viene aggiunto al pannello parente un nuovo widget
	void On_Widget_Add(ui::detail::widget_guid_t parent_guid, ui::detail::widget_guid_t guid, const ui::detail::WidgetCore& widget) {
		cout << "ON WIDGET ADD aaaaaaaaaaaaaaaaaaa" << endl;
	};
	/// Quando un widget cambia geometria
	void On_Widget_Changed(ui::detail::widget_guid_t guid, const ui::detail::WidgetCore& widget) {
		cout << "ON WIDGET CHANGED aaaaaaaaaaaaaaaaaaa" << endl;
	};
	/// Quando un widget viene rimosso
	void On_Widget_Remove(ui::detail::widget_guid_t guid) {};
	/// Quando un widget cambia valore *
	void On_Value_Changed(ui::detail::widget_guid_t guid, const ui::detail::WidgetCore& widget) {
		cout << "Value changed! =" << endl;
		try {
			mDummy->On_Execute();
		} catch (...) {
			cout << "EXCEPTION! load a frame before changing params" << endl;
		}
	};
	/// Quando un widget cambia attributi nella parte del dato
	void On_Data_Changed(ui::detail::widget_guid_t guid, const ui::detail::WidgetCore& widget) {
		cout << "ON DATA CHANGED aaaaaaaaaaaaaaaaaaa" << endl;
	};

	/// Quando un pannello diventa visible
	void On_Panel_Opened(ui::detail::widget_guid_t guid, const ui::detail::WidgetCore& widget) {
		cout << "ON PANEL OPENED aaaaaaaaaaaaaaaaaaa" << endl;
	};
	/// Quando un widget viene nascosto
	void On_Panel_Close(ui::detail::widget_guid_t guid) {
		cout << "ON PANEL CLOSE aaaaaaaaaaaaaaaaaaa" << endl;
	};

private:
	CDummy * mDummy;
};

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
    
    Value<bool> showInputHOG(&m_showInputHOG);
    config.Bind(showInputHOG, "SHOW INPUT HOG", false);
    
    Value<bool> showDetected(&m_showDetected);
    config.Bind(showDetected, "SHOW DETECTED", false);
    
    ui::var::Range<float> threshold(&m_threshold, -2.0f, 0.5f, 0.1f);
    config.Bind(threshold, "THRESHOLD", -0.4f);
    
    ui::var::Range<float> sliderW0(&m_W0, 0.0f, 2.0f, 0.01f);
    config.Bind(sliderW0, "W0", 0.2f);

    ui::var::Range<float> sliderW1(&m_W1, 0.0f, 2.0f, 0.01f);
	config.Bind(sliderW1, "W1", 1.0f);

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
    
    Slider thresholdSlider(threshold, "Threshold");

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
                        CheckBox(showInputHOG, "Show input HOG"),
                        CheckBox(showDetected, "Show detected")
                    )
                ),
                Page("Advanced")
                (
                    VSizer()
                    (
                        thresholdSlider, //Slider(threshold, "Threshold"),
						Slider(sliderW0, "W0"),
						Slider(sliderW1, "W1"),
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

    ui::detail::listener_id_t listenerId = ui::detail::PanelManagerSingleton::Instance().Register(new CMyPanelListener(this));
    ui::detail::PanelManagerSingleton::Instance().Register(panel.GUID(),listenerId);

    // configuriamo il Synchronizer in modo che ritorni sempre l'ultimo frame ricevuto dalla camera indicata
    m_pCam = (Dev()["CAMERAS/" + m_inputCameraName]);
    m_synchro.ConnectSync(*m_pCam);
    
    // inizializziamo alcune variabili
    m_pInputMonoWindow = NULL;
    m_pInputHOGWindow = NULL;
    m_pDetectedWindow = NULL;

    //inzializziamo i timer per il profiler:
	dev::CProfiler & profiler  = static_cast<dev::CProfiler&>(Dev()["Profiler"]);
	m_cvChronometer = vl::chrono::CChronometer("openCV",vl::chrono::CChronometer::REAL_TIME_CLOCK);
	profiler.Connect(m_cvChronometer);
	vl::chrono::CChronometer ffldChronometer = vl::chrono::CChronometer("ffldHOG",vl::chrono::CChronometer::REAL_TIME_CLOCK);
	profiler.Connect(ffldChronometer);

	//inizializzo ffld (carica il modello)
	ffld.init("/home/alox/Tesi/workspace/Dummy/src/ffld/models/person_final2007.txt");
}

void CDummy::On_ShutDown()
{
    delete m_pInputMonoWindow;
    delete m_pInputHOGWindow;
    delete m_pDetectedWindow;
}

void CDummy::On_Execute()
{
	int frameNumber = 0;

    CImage::SharedPtrConstType image;
    
    {

        // copiamo lo shared pointer al frame, che utilizzeremo in seguito per l'elaborazione
        image = m_synchro.SyncFrameFrom<dev::CCamera>(*m_pCam).Data;
        frameNumber = m_synchro.SyncFrameFrom<dev::CCamera>(*m_pCam).Number;
        
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
    Resize(m_detectedImage, m_width, m_height);
    Resize(m_srcImageRGB, m_width, m_height);

    // convertiamo il frame in una immagine a colori
    Convert(*image, m_inputImageRGB, BAYER_DECODING_SIMPLE);
    Convert(*image, m_srcImageRGB, BAYER_DECODING_SIMPLE);

    // convertiamo il frame in una immagine in bianco e nero
    Convert(*image, m_inputImageMono, BAYER_DECODING_LUMINANCE);

    // applichiamo un filtro che converte una immagine CImageMono in una CImageMono usando un kernel SobelVertical di dimensione 3x3
    //SobelVertical3x3(m_inputImageMono, m_sobelImage);

    if (m_showInputHOG || m_showDetected) {
		Mat m = CHOGVisualizer::CImageRGB8ToMat(m_inputImageRGB);

		std::vector<float> descriptorsValues;
		std::vector<cv::Point> locations;
		//resize(m, m, Size(512,256) );
		Mat img;
		resize(m, m, Size(512,256) ); //la dimensione su cui calcolo le feature HOG
		cvtColor(m, img, COLOR_BGR2GRAY);
		img.convertTo(img,CV_8U); //converto in scala di grigi perche' openCV vuole scala di grigi (di solito si usano tutti i canali e si prende il gradiente maggiore)
		//imwrite( "./butta.jpg", img );
		//cv::gpu::HOGDescriptor d(Size(512,256), Size(16,16), Size(8,8),Size(8,8), 9);
		HOGDescriptor d(Size(512,256), Size(16,16), Size(8,8),Size(8,8), 9);

		m_cvChronometer.Start();
		d.compute(img, descriptorsValues, Size(8,8), Size(0,0), locations);
		m_cvChronometer.Stop();

		Mat viz = CHOGVisualizer::GetHogDescriptorVisu(m,descriptorsValues,Size(512,256));
		viz.convertTo(viz,CV_16UC3);

		resize(viz,viz,Size(m_inputImageRGB.W(),m_inputImageRGB.H()));
		//resize(img,img,Size(m_inputImageRGB.W(),m_inputImageRGB.H()));
		//imwrite( "./butta2.jpg", viz );
		CHOGVisualizer::MatToCImageRGB8(viz,m_inputImageRGB);
		if(m_showDetected)
		{

			std::vector<String> filenames;
			filenames.push_back("/home/alox/Downloads/person.xml");
			LatentSvmDetector detector(filenames);
			if( detector.empty() )
			{
				cout << "Models cann't be loaded" << endl;
				exit(-1);
			}
			vector<LatentSvmDetector::ObjectDetection> detections;
			cout << "Detecting..." << endl;
			detector.detect( m, detections, 0.2f, 8);

			for( size_t i = 0; i < detections.size(); i++ )
			{
				const LatentSvmDetector::ObjectDetection& od = detections[i];
				cout << "confidence:" << od.score << endl;
				if (od.score > 0.1f) rectangle( m, od.rect, Scalar(255,0,255*od.score), 2 );
			}
			resize(m,m,Size(m_inputImageRGB.W(),m_inputImageRGB.H()));
			CHOGVisualizer::MatToCImageRGB8(m,m_detectedImage);
		}
    }

	//-m models/person_final2007.txt -i ./result/ -r ./result/result.txt -t=-0.1 004963.jpg
	//-m /home/alox/Tesi/workspace/Dummy/src/ffld/models/person_final2007.txt -i / -r  -t=-0.1 /home/alox/Tesi/ffld/pedoni.jpg
	char arg0[] = "ffld";
	char arg1[] = "-m";
	char arg2[] = "/home/alox/Tesi/workspace/Dummy/src/ffld/models/person_final2007.txt";
	char arg3[] = "-i";
	char arg4[] = "/home/alox/Tesi/workspace/Dummy/src/ffld/result/";
	char arg5[] = "-r";
	char arg6[] = "/home/alox/Tesi/workspace/Dummy/src/ffld/result/result.txt";
	char arg7[] = "-t=-0.1";
	//char arg8[] = "/home/alox/Tesi/ffld/pedoni.jpg";
	char arg8[] = "./butta.jpg";
	char* argv[] = { &arg0[0], &arg1[0], &arg2[0], &arg3[0], &arg4[0], &arg5[0], &arg6[0], &arg7[0], &arg8[0], NULL };
	FFLD::JPEGImage immmg();
	int argc = sizeof(argv) / sizeof(char*) - 1;
	//main_ffld(argc,argv,m_srcImageRGB);
	SearchRange r;
	r.setSearchRange(image->W(),image->H(),m_pCam,m_inputImageMono, m_W0, m_W1);

	vector<Detection> detections;
	ffld.dpmDetect("/home/alox/Tesi/workspace/Dummy/src/ffld/models/person_final2007.txt",m_srcImageRGB, m_threshold,r,detections);

	//print detections to file using KITTI format for evaluation
	ofstream myfile;
	char filename[100];
	sprintf(filename, "/home/alox/Tesi/detections/%06d.txt",frameNumber);
	myfile.open(filename);

	for (int i = 0; i < detections.size(); ++i) {
		bool plausible = r.isPlausibleSize(detections[i].bottom(),detections[i].width());
		if(plausible) {
		myfile << "Pedestrian -1 -1 -10 "
				<< detections[i].left() << " "
				<< detections[i].top() << " "
				<< detections[i].right() << " "
				<< detections[i].bottom() << " "
				<< "-1 -1 -1 -1000 -1000 -1000 -10 " << (plausible ? detections[i].score : (detections[i].score - 1))
				<< endl;
		}
	}
	myfile.close();

	//Mat detectionResult = imread("/home/alox/Tesi/workspace/Dummy/src/ffld/result/butta.jpg");
	//detectionResult.convertTo(detectionResult,CV_16UC3);
	//CHOGVisualizer::MatToCImageRGB8(detectionResult,m_inputImageMono);
	m_inputImageMono = m_srcImageRGB;
	r.draw(m_inputImageMono);

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
            m_pInputMonoWindow->DrawCircle(m_value, m_value, m_threshold, false);
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

    if(m_showInputHOG)
    {
        if(m_pInputHOGWindow == NULL)
            m_pInputHOGWindow = new CWindow("Input HOG", m_width, m_height);

        m_pInputHOGWindow->Clear();
        m_pInputHOGWindow->DrawImage(m_inputImageRGB);
        m_pInputHOGWindow->Refresh();

        if(!m_pInputHOGWindow->IsVisible())
            m_pInputHOGWindow->Show();
    } else if(m_pInputHOGWindow)
        m_pInputHOGWindow->Hide();

    if(m_showDetected)
    {
        if(m_pDetectedWindow == NULL)
            m_pDetectedWindow = new CWindow("Detected", m_width, m_height);

        m_pDetectedWindow->Clear();
        m_pDetectedWindow->DrawImage(m_detectedImage);
        m_pDetectedWindow->Refresh();

        if(!m_pDetectedWindow->IsVisible())
            m_pDetectedWindow->Show();
    } else if(m_pDetectedWindow)
        m_pDetectedWindow->Hide();
}

#include <Framework/Application_Registration.h>

// registriamo la classe CDummy col nome Dummy
REGISTER_APPLICATION(CDummy, "Dummy");
