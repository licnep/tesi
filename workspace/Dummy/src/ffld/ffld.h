//--------------------------------------------------------------------------------------------------
// Implementation of the paper "Exact Acceleration of Linear Object Detectors", 12th European
// Conference on Computer Vision, 2012.
//
// Copyright (c) 2012 Idiap Research Institute, <http://www.idiap.ch/>
// Written by Charles Dubout <charles.dubout@idiap.ch>
//
// This file is part of FFLD (the Fast Fourier Linear Detector)
//
// FFLD is free software: you can redistribute it and/or modify it under the terms of the GNU
// General Public License version 3 as published by the Free Software Foundation.
//
// FFLD is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
// Public License for more details.
//
// You should have received a copy of the GNU General Public License along with FFLD. If not, see
// <http://www.gnu.org/licenses/>.
//--------------------------------------------------------------------------------------------------
#pragma once
#include "ffld_export.h"

#include "SimpleOpt.h"

#include "Intersector.h"
#include "Mixture.h"
#include "Scene.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <Data/CImage/CImage.h>
#include <Data/CImage/Images/CImageRGB8.h>

#include <SearchRange.h>

using namespace FFLD;
using namespace std;

static vl::chrono::CChronometer ffldChronometer;

struct Detection : public FFLD::Rectangle
{
	HOGPyramid::Scalar score;
	int l;
	int x;
	int y;

	Detection() : score(0), l(0), x(0), y(0)
	{
	}

	Detection(HOGPyramid::Scalar score, int l, int x, int y, FFLD::Rectangle bndbox) :
	FFLD::Rectangle(bndbox), score(score), l(l), x(x), y(y)
	{
	}

	bool operator<(const Detection & detection) const
	{
		return score > detection.score;
	}
};


class CFfld {
public:
	//loads the model
	int init(std::string model_path);

	//draws the detected people in the rectangle
	int dpmDetect(std::string model_path, cimage::CImageRGB8 & srcImage, double threshold, SearchRange r,vector<Detection> &detections);
private:
	Mixture mMixture;
};


// Test a mixture model (compute a ROC curve)
int main_ffld(int argc, char * argv[], cimage::CImageRGB8 &srcImage);
