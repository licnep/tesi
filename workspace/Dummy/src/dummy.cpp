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
#include <Processing/Vision/CImage/Draw/Brushes.h>
#include <Processing/Vision/CImage/Draw/Box.h>
#include <Data/CImage/IO/CImageIO.h>

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
#include <opencv2/latentsvm.hpp>
//#include "../../../../Downloads/opencv-3.0.0-alpha/opencv_contrib/modules/latentsvm/src/_lsvmc_latentsvm.h"
#include "latentsvm/cascadeDetector.h"
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

#define OPENCV3 true

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
//#pragma omp critical(execute)
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
    config.Bind(showInputMono, "SHOW INPUT AND DETECTIONS", false);
    
    Value<bool> showInputHOG(&m_showInputHOG);
    config.Bind(showInputHOG, "SHOW INPUT HOG", false);
    
    Value<bool> showDetected(&m_showDetected);
    config.Bind(showDetected, "SHOW OPENCV DETECTIONS", false);
    
    Value<string> modelPath(&m_modelPath);
    config.Bind(modelPath, "MODEL PATH", "/home/alox/Tesi/ffld/models/kitti1000.txt");

    ui::var::Range<float> threshold(&m_threshold, -3.0f, 5.0f, 0.1f);
    config.Bind(threshold, "THRESHOLD", -0.4f);
    
    ui::var::Range<int> interval(&m_interval, 1, 10, 1);
    config.Bind(interval, "PYRAMID INTERVAL", 4);

	ui::var::Range<float> sliderScale(&m_scale, 0.5f, 2.0f, 0.1f);
	config.Bind(sliderScale, "SCALE", 1.0f);

    ui::var::Range<int> value(&m_value, 0, 150, 2);
    config.Bind(value, "VALUE", 40);
    
    Value<bool> enableSearchRanges(&m_enableSearchRanges);
	config.Bind(enableSearchRanges, "ENABLE SEARCH RANGES", true);

	Value<bool> showGroundTruth(&m_showGroundTruth);
	config.Bind(showGroundTruth, "SHOW GROUND TRUTH", true);

    ui::var::Range<float> sliderW0(&m_W0, 0.0f, 2.0f, 0.01f);
    config.Bind(sliderW0, "W0", 0.2f);

    ui::var::Range<float> sliderW1(&m_W1, 0.0f, 2.0f, 0.01f);
	config.Bind(sliderW1, "W1", 1.0f);

    Value<bool> showCircle(&m_showCircle);
    config.Bind(showCircle, "SHOW CIRCLE", false);
    
    Value<bool> showBox(&m_showBox);
    config.Bind(showBox, "SHOW BOX", false);
    
    Value<bool> showText(&m_showText);
    config.Bind(showText, "SHOW TEXT", false);
    
    ui::var::Range<float> overlap(&m_overlap, 0.1f, 1.0f, 0.1f);
	config.Bind(overlap, "OVERLAP", 0.5f);

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
                        CheckBox(showInputMono, "Show input and detections"),
                        CheckBox(showInputHOG, "Show HOG"),
                        CheckBox(showDetected, "Show OpenCV detections")
                    )
                ),
                Page("Advanced")
                (
                    VSizer()
                    (
						Slider(interval, "Pyramid interval"),
                        thresholdSlider, //Slider(threshold, "Threshold"),
						CheckBox(enableSearchRanges,"Use search ranges"),
						Slider(sliderW0, "W0"),
						Slider(sliderW1, "W1"),
						Slider(overlap, "Overlap"),
						Slider(sliderScale, "Scale"),
                        CheckBox(showGroundTruth, "Show Ground Truth")
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

	//inizializzo ffld (carica il modello)
	ffld.init(m_modelPath);
}

void CDummy::On_ShutDown()
{
	remove("wisdowm.fftw");
    delete m_pInputMonoWindow;
    delete m_pInputHOGWindow;
    delete m_pDetectedWindow;
}

void printWidthStats() {
	vector<Detection> groundTruth;
	char filename[256];
	for (int i=0;i<7480;i++) {
		sprintf(filename, "/home/alox/Tesi/Sequenze/Kitti/training/label_2/%06d.txt",i);
		FILE *fp = fopen(filename,"r");
		if (!fp) {
			  cout << "No ground truth data available at " << filename << endl;
			return;
		}
		while (!feof(fp)) {
			double trash, x1,y1,x2,y2;
			char str[255];
			if (fscanf(fp, "%s %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
						   str, &trash, &trash, &trash,
						   &x1,   &y1,     &x2,    &y2,
						   &trash,      &trash,        &trash,       &trash,
						   &trash,      &trash,        &trash )==15) {
				//Detection(HOGPyramid::Scalar score, int l, int x, int y, FFLD::Rectangle bndbox)
				FFLD::Rectangle bndbox(x1,y1,x2-x1,y2-y1);
			  if (strcmp(str,"Pedestrian")==0) {
				  groundTruth.push_back( Detection(100,0,x1,y1,bndbox) );
			  }
			}
		}
		fclose(fp);
	}
	//here detections has been populated with all pedestrian ground thrut detections, now we calculate, for each row, the min and max width
	int MAX_ROWS=370;
	int mins[MAX_ROWS], maxs[MAX_ROWS];
	for (int i=0;i<MAX_ROWS;i++) {mins[i] = 9999; maxs[i] = 0;}

	ofstream myfile;
	myfile.open("ranges.txt");
	for (int r=0; r<groundTruth.size();r++) {
		int row = groundTruth[r].bottom();
		int width = groundTruth[r].width();
		if (width < mins[row]) mins[row] = width;
		if (width > maxs[row]) maxs[row] = width;
		myfile << row << "  \t" << width << endl;
	}
	myfile.close();
	for (int i=0;i<MAX_ROWS;i++) {
		cout << i << "  \t" << mins[i] << "  \t" << maxs[i] << endl;
	}
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

    FFLD::Globals::GLOBAL_SCALE = m_scale;
    FFLD::Globals::SEARCH_RANGES_ENABLED = m_enableSearchRanges;

    // larghezza e altezza possono variare da un frame all'altro (e' cosi' nel dataset KITTI)
    m_width = image->W();
    m_height = image->H();

    // eseguiamo la Resize comunque: se m_width e m_height sono già corrette non succede nulla
    Resize(m_inputImageMono, m_width, m_height);
    Resize(m_inputImageRGB, m_width, m_height);
    Resize(m_detectedImage, m_width, m_height);
    Resize(m_srcImageRGB, m_width, m_height);

    // convertiamo il frame in una immagine a colori
    Convert(*image, m_inputImageRGB, BAYER_DECODING_SIMPLE);
    Convert(*image, m_srcImageRGB, BAYER_DECODING_SIMPLE);

    // convertiamo il frame in una immagine in bianco e nero
    Convert(*image, m_inputImageMono, BAYER_DECODING_LUMINANCE);

    // [LEGACY] mostriamo le feature di HOG e detection calcolate con OpenCV se l'utente lo richiede
    if (m_showInputHOG || m_showDetected) {
		Mat m = CHOGVisualizer::CImageRGB8ToMat(m_inputImageRGB);

		std::vector<float> descriptorsValues;
		std::vector<cv::Point> locations;
		//resize(m, m, Size(512,256) );
		Mat img;
		resize(m, m, Size(512,256) ); //la dimensione su cui calcolo le feature HOG
		cvtColor(m, img, COLOR_BGR2GRAY);
		img.convertTo(img,CV_8U); //converto in scala di grigi perche' openCV vuole scala di grigi (di solito si usano tutti i canali e si prende il gradiente maggiore)

//#ifndef OPENCV3
		HOGDescriptor d(Size(512,256), Size(16,16), Size(8,8),Size(8,8), 9);

		//m_cvChronometer.Start();
		d.compute(img, descriptorsValues, Size(8,8), Size(0,0), locations);
		//m_cvChronometer.Stop();
//#endif
		Mat viz = CHOGVisualizer::GetHogDescriptorVisu(m,descriptorsValues,Size(512,256));
		viz.convertTo(viz,CV_16UC3);

		resize(viz,viz,Size(m_inputImageRGB.W(),m_inputImageRGB.H()));
		//resize(img,img,Size(m_inputImageRGB.W(),m_inputImageRGB.H()));
		//imwrite( "./butta2.jpg", viz );
		CHOGVisualizer::MatToCImageRGB8(viz,m_inputImageRGB);
		if(m_showDetected)
		{
			m_cvChronometer.Start();

			std::vector<std::string> filenames;

#ifndef OPENCV3
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
#else
			//cv::Ptr<lsvm::LSVMDetector> detector1 = lsvm::LSVMDetector::create(std::vector<std::string>(1,"/home/alox/Tesi/workspace/Dummy/src/latentsvm/testdata/latentsvm/models_VOC2007_cascade/person.xml"));
			filenames.push_back("/home/alox/Tesi/workspace/Dummy/src/latentsvm/testdata/latentsvm/models_VOC2007_cascade/person.xml");

			cv::lsvm::LSVMDetectorImpl detector(filenames);
			//vector<LatentSvmDetector::ObjectDetection> detections;
			vector<cv::lsvm::LSVMDetector::ObjectDetection> detections;
			cout << "Detecting..." << endl;
			detector.detect( m, detections, 0.5f);
			for( size_t i = 0; i < detections.size(); i++ )
			{
				const cv::lsvm::LSVMDetector::ObjectDetection& od = detections[i];
				cout << "confidence:" << od.score << endl;
				if (od.score > 0.1f) rectangle( m, od.rect, Scalar(255,0,255*od.score), 2 );
				rectangle( m, od.rect, Scalar(255,0,255*od.score), 2 );
			}
			resize(m,m,Size(m_inputImageRGB.W(),m_inputImageRGB.H()));
			CHOGVisualizer::MatToCImageRGB8(m,m_detectedImage);

#endif
		}
		m_cvChronometer.Stop();
    }

    // creo la classe SearchRange che puo' variare da fotogramma a fotogramma (dato che le immagini variano di dimensioni)
	SearchRange r;
	r.setSearchRange(m_srcImageRGB.W(),m_srcImageRGB.H(),m_pCam,m_srcImageRGB, m_W0, m_W1);

	vector<Detection> detections;

	FFLD::Globals::PYRAMID_INTERVAL = m_interval;
	FFLD::Globals::OVERLAP = m_overlap;

	// qui succede la magia
	ffld.dpmDetect(m_modelPath,m_srcImageRGB, m_threshold,r,detections);

	// carica i label corretti se presenti
	if (m_showGroundTruth)
		loadGroundTruth(frameNumber,m_srcImageRGB,detections);

	// print detections to file using KITTI format for evaluation
	ofstream myfile;
	char filename[100];
	sprintf(filename, "/home/alox/Tesi/detections/%06d.txt",frameNumber);
	myfile.open(filename);

	for (int i = 0; i < detections.size(); ++i) {
		myfile << "Pedestrian -1 -1 -10 "
				<< detections[i].left() << " "
				<< detections[i].top() << " "
				<< detections[i].right() << " "
				<< detections[i].bottom() << " "
				<< "-1 -1 -1 -1000 -1000 -1000 -10 " << detections[i].score << " "
				<< detections[i].l
				<< endl;
	}
	myfile.close();


	m_inputImageMono = m_srcImageRGB;
	// disegno il search range sull'immagine per renderlo piu' comprensibile
	r.draw(m_inputImageMono);

    // chiamiamo la funzione di disegno
    Output();
}

void CDummy::loadGroundTruth(int frameNumber, cimage::CImageRGB8 &srcImage,vector<Detection> detections) {
	// Try to open the label file
	char filename[256];
	sprintf(filename, "/home/alox/Tesi/Sequenze/Kitti/training/label_2/%06d.txt",frameNumber);
	ifstream in(filename, ios::binary);

	if (!in.is_open()) {
		cout << "No ground truth data available at " << filename << endl;
		return;
	}


	vector<Detection> groundTruth;
	// holds all ground truth (ignored ground truth is indicated by an index vector
	//vector<tGroundtruth> groundtruth;
	FILE *fp = fopen(filename,"r");
	if (!fp) {
		  cout << "No ground truth data available at " << filename << endl;
		return;
	}
	while (!feof(fp)) {
	    double trash, x1,y1,x2,y2;
	    char str[255];
	    if (fscanf(fp, "%s %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
	                   str, &trash, &trash, &trash,
	                   &x1,   &y1,     &x2,    &y2,
	                   &trash,      &trash,        &trash,       &trash,
	                   &trash,      &trash,        &trash )==15) {
	    	//Detection(HOGPyramid::Scalar score, int l, int x, int y, FFLD::Rectangle bndbox)
	    	FFLD::Rectangle bndbox(x1,y1,x2-x1,y2-y1);
	      if (strcmp(str,"Pedestrian")==0) {
	    	  Detection d(100,0,x1,y1,bndbox);
	    	  //d.scale(FFLD::Globals::GLOBAL_SCALE);
	    	  groundTruth.push_back( d );
	      }
	    }
	}
	fclose(fp);

	ofstream detectedFile,undetectedFile;
	detectedFile.open("detected.txt", ios::out | ios::app);
	undetectedFile.open("undetected.txt", ios::out | ios::app);
	ofstream allUndetecteds;
	char filename2[100];
	sprintf(filename2, "/home/alox/Tesi/undetections/%06d.txt",frameNumber);
	allUndetecteds.open(filename2);

	//draw ground truth, dark green if it was detected, light green if it wasn't
	for (int i=0;i<groundTruth.size();i++) {
		bool detected = false;
		Intersector intersect(groundTruth[i], 0.5f, false);
		for (int j=0;j<detections.size();j++) {
			if (intersect(detections[j])) {
				detected = true;
			}
		}
		math::Rect2i r(groundTruth[i].left(),groundTruth[i].top(),groundTruth[i].right(),groundTruth[i].bottom());
		draw::Opaque<cimage::RGB8> brush(srcImage, detected ? cimage::RGB8(0,150,0) : cimage::RGB8(0,255,0) );
		draw::Rectangle(brush,r);
		if (detected) {
			detectedFile << groundTruth[i].bottom() << "\t" << groundTruth[i].width() << endl;
		} else {
			undetectedFile << groundTruth[i].bottom() << "\t" << groundTruth[i].width() << "\t" << frameNumber << endl;
			//print undetected to file using KITTI format if we want to train on them
			allUndetecteds << "Pedestrian -1 -1 -10 "
					<< groundTruth[i].left() << " "
					<< groundTruth[i].top() << " "
					<< groundTruth[i].right() << " "
					<< groundTruth[i].bottom() << " "
					<< "-1 -1 -1 -1000 -1000 -1000 -10 " << 999
					<< endl;
			//string percorso = "/home/alox/Tesi/undetections/" + boost::lexical_cast<std::string>(frameNumber) +".jpg";;
			//cimage::Save(percorso,srcImage);
		}
	}

	detectedFile.close();
	undetectedFile.close();
	allUndetecteds.close();
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
