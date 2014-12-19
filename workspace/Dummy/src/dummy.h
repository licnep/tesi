#ifndef _DUMMY_H
#define _DUMMY_H

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
 * \file dummy.h
 * \author VisLab (vislab@ce.unipr.it)
 * \date 2006-04-27
 */

#include "Application.h"

#include <Devices/Camera/CCamera.h>
#include <Data/CImage/Images/CImageBayer.h>
#include <Data/CImage/Images/CImageMono8.h>
#include <Data/CImage/Images/CImageRGB8.h>
#include <Framework/Synchronizers.h>
#include <UI/CWindows/CWindow.h>
#include <UI/Panel/Panel.h>

#include <opencv2/opencv.hpp>
#include "ffld/ffld.h"

#include <string>
#include <vector>

/**
 * \class CDummy
 * \brief Sample functionality class
 *
 * This class contains some sample code demonstrating:
 * - images acquisition
 * - visualization
 * - performance measures
 * - user interface usage
 */
class CDummy : public CApplication
{
    public:

        DECL_METHODS;

        virtual void On_Initialization();

        virtual void On_ShutDown();

        virtual void On_Execute();

        virtual usr::Synchronizer_Basic * Synchro(){
        	return &m_synchro;
        }

    private:

        void Output();

        std::vector<std::string> m_cameraNames;
        std::string m_inputCameraName;

        //cimage::CImageMono m_inputImageMono;
        cimage::CImageRGB8 m_inputImageMono;
        cimage::CImageRGB8 m_detectedImage;
        cimage::CImageRGB8 m_inputImageRGB;
        cimage::CImageRGB8 m_srcImageRGB;

        dev::CCamera * m_pCam;
	Synchronizer_Basic m_synchro;

		vl::chrono::CChronometer m_cvChronometer;
		CFfld ffld;
        
        unsigned int m_width, m_height;

        ui::win::CWindow* m_pInputMonoWindow;
        ui::win::CWindow* m_pInputHOGWindow;
        ui::win::CWindow* m_pDetectedWindow;

        float m_threshold, m_W0, m_W1;
        int m_value;

        bool m_showInputMono;
        bool m_showInputHOG;
        bool m_showDetected;
        bool m_showBox;
        bool m_showCircle;
        bool m_showText;
        bool m_showTimer;
        int m_selectedFeature;
};

#endif
