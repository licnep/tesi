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

#include "Model.h"

#include <algorithm>
#include <cmath>
#include <Data/CImage/IO/CImageIO.h>

using namespace Eigen;
using namespace FFLD;
using namespace std;

Model::Model() : parts_(1), bias_(0.0)
{
	parts_[0].offset.setZero();
	parts_[0].deformation.setZero();
}

bool Model::empty() const
{
	return !parts_[0].filter.size() && !nbParts();
}

pair<int, int> Model::rootSize() const
{
	return pair<int, int>(parts_[0].filter.rows(), parts_[0].filter.cols());
}

int Model::nbParts() const
{
	return parts_.size() - 1;
}

pair<int, int> Model::partSize() const
{
	if (nbParts())
		return pair<int, int>(parts_[1].filter.rows(), parts_[1].filter.cols());
	else
		return pair<int, int>(0, 0);
}

Model::Scalar Model::bias() const
{
	return bias_;
}

void Model::convolve(const HOGPyramid & pyramid, vector<HOGPyramid::Matrix> & scores,
					 vector<vector<Positions> > * positions) const
{
	// Invalid parameters
	if (empty() || pyramid.empty())
		return;
	
	// The only necessary constant
	const int nbFilters = parts_.size();
	
	// Convolve the pyramid with all the filters
	vector<vector<HOGPyramid::Matrix> > convolutions(nbFilters);
	
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < nbFilters; ++i)
		pyramid.convolve(parts_[i].filter, convolutions[i]);
	
	convolve(pyramid, convolutions, scores, positions);
}

Model Model::flip() const
{
	Model model;
	
	if (!empty()) {
		model.parts_.resize(parts_.size());
		
		// Flip the root
		model.parts_[0].filter = HOGPyramid::Flip(parts_[0].filter);
		model.parts_[0].offset = parts_[0].offset;
		model.parts_[0].deformation = parts_[0].deformation;
		
		// Flip the parts
		for (int i = 1; i < parts_.size(); ++i) {
			model.parts_[i].filter = HOGPyramid::Flip(parts_[i].filter);
			model.parts_[i].offset(0) = parts_[0].filter.cols() * 2 - parts_[i].filter.cols() -
										parts_[i].offset(0);
			model.parts_[i].offset(1) = parts_[i].offset(1);
			model.parts_[i].deformation = parts_[i].deformation;
			model.parts_[i].deformation(1) = -model.parts_[i].deformation(1);
		}
	}
	
	model.bias_ = bias_;
	
	return model;
}

void savePlane(string percorso, HOGPyramid::Matrix plane) {
	int cols = plane.cols();
	int rows = plane.rows();
	cimage::CImageRGB8 img(cols,rows);
	cimage::RGB8* dstBuffer = img.Buffer();
	float max = 0, min = 255;
	for (int i=0; i<cols;i++) {
		for (int j=0; j<rows;j++) {
			float val = plane(j, i);
			dstBuffer[j*cols+i].R = 0;dstBuffer[j*cols+i].B = 0;
			if (val>0&&val<4)
				dstBuffer[j*cols+i].G = 255*(val/4.0f);
			else if (val>=4)
				dstBuffer[j*cols+i].G = 255;
			else
				dstBuffer[j*cols+i].G = 0;
		}
	}
	cimage::Save(percorso,img);
}

void Model::printToFile(std::string filename) {
	HOGPyramid::saveLevel(filename,parts_[0].filter,15.0f);
}

void Model::convolve(const HOGPyramid & pyramid, vector<vector<HOGPyramid::Matrix> > & convolutions,
					 vector<HOGPyramid::Matrix> & scores,
					 vector<vector<Positions> > * positions) const
{
	// Invalid parameters
	if (empty() || pyramid.empty() || (convolutions.size() != parts_.size()))
		return;

	// All the constants relative to the model and the pyramid
	const int nbFilters = parts_.size();
	const int nbParts = nbFilters - 1;
	const int padx = pyramid.padx();
	const int pady = pyramid.pady();
	const int interval = pyramid.interval();
	const std::vector<std::pair<int,int> > offsets = pyramid.offsets();
	const int nbLevels = pyramid.levels().size();
	
	// Resize the positions
	if (positions && nbParts) {
		positions->resize(nbParts);
		
		for (int i = 0; i < nbParts; ++i)
			(*positions)[i].resize(nbLevels);
	}
	
	// Temporary data needed by the distance transforms
	vector<Scalar> tmp(pyramid.levels()[0].size());

	// uncomment for DEBUGGING purposes, save model levels to file
	/*for (int j = 0; j < nbLevels - interval; ++j) {
		string percorso = "/home/alox/ROOTlevel"+ boost::lexical_cast<std::string>(j) + ".png";
		savePlane(percorso,convolutions[0][j+interval]);
	}*/

	//pre-calculate the offsets for faster computation of cell position one octave below
	//for each level used for parts (0 to nbLevels-interval) we store how much to subtract and add to calculate position one octave below.
	int srcOffsets[nbLevels-interval], dstOffsets[nbLevels-interval];
	std::vector<std::pair<int,int> > offsets_ = pyramid.offsets();
	for (int i=0; i< nbLevels-interval; i++) {
		int src_level = i+interval;
		srcOffsets[i] = offsets_[src_level].first/8;
		dstOffsets[i] = (src_level>=interval*2) ? offsets_[src_level-interval].first/8 : offsets_[src_level-interval].first/4;
	}


	// For each part
	for (int i = 0; i < nbParts; ++i) {
		// For each part level (the root is interval higher in the pyramid)
		for (int j = 0; j < nbLevels - interval; ++j) {
			//A for each level except the smallest ones (only used for root filters), calculate the part optimal positions and deformation costs given any root position
			DT2D(convolutions[i + 1][j], parts_[i + 1], &tmp[0],
				 positions ? &(*positions)[i][j] : 0);
			//For DEBUGGING purposses:
			//string percorso = "/home/alox/partIlevel"+ boost::lexical_cast<std::string>(j) + ".png";
			//savePlane(percorso,convolutions[0][j+interval]);


			// Add the distance transforms of the part one octave below
			for (int y = 0; y < convolutions[0][j + interval].rows(); ++y) {
				for (int x = 0; x < convolutions[0][j + interval].cols(); ++x) {
					// The position of the root one octave below
					const int x2 = x * 2 - padx;
					const int y2 = (y+srcOffsets[j]) * 2 - pady - dstOffsets[j];
					
					// Nearest-neighbor interpolation
					if ((x2 >= 0) && (y2 >= 0) && (x2 < convolutions[i + 1][j].cols()) &&
						(y2 < convolutions[i + 1][j].rows()))
						convolutions[0][j + interval](y, x) += convolutions[i + 1][j](y2, x2);
					else
						convolutions[0][j + interval](y, x) =- numeric_limits<Scalar>::infinity();
				}
			}
		}
	}

	//Uncomment for DEBUGGING purposes:
	/*
#pragma omp critical
	for (int j = interval; j < nbLevels; ++j) {
		string percorso = "/home/alox/partIlevel"+ boost::lexical_cast<std::string>(j) + ".png";
		savePlane(percorso,convolutions[0][j]);
	}*/

	scores.swap(convolutions[0]);
	
	// Add the bias if necessary
	if (bias_) {
		int i;
#pragma omp parallel for private(i)
		for (i = 0; i < nbLevels; ++i)
			scores[i].array() += bias_;
	}
}

void Model::DT1D(const Scalar * x, int n, Scalar a, Scalar b, Scalar * z, int * v, Scalar * y,
				 int * m, int offset, const Scalar * t, int incx, int incy, int incm)
{
	// Early return in case any of the parameters is invalid
	if (!x || (n <= 0) || (a >= 0) || !z || !v || !y || !incx || !incy || !incm)
		return;
	
	z[0] =-numeric_limits<Scalar>::infinity();
	z[1] = numeric_limits<Scalar>::infinity();
	v[0] = 0;
	
	Scalar xvk = x[0];
	
	// Same version of the algorithm except that the first version uses a lookup table to replace
	// the division by (a * (i - v[k]))
	if (t) {
		int k = 0;
		
		for (int i = 1; i < n;) {
			const Scalar s = (x[i * incx] - xvk) * t[i - v[k]] + (i + v[k]) - b / a;
			
			if (s <= z[k]) {
				--k;
				xvk = x[v[k] * incx];
			}
			else {
				++k;
				v[k] = i;
				z[k] = s;
				xvk = x[i * incx];
				++i;
			}
		}
		
		z[k + 1] = numeric_limits<Scalar>::infinity();
	}
	else {
		int k = 0;
		
		for (int i = 1; i < n;) {
			const Scalar s = (x[i * incx] - xvk) / (a * (i - v[k])) + (i + v[k]) - b / a;
			
			if (s <= z[k]) {
				--k;
				xvk = x[v[k] * incx];
			}
			else {
				++k;
				v[k] = i;
				z[k] = s;
				xvk = x[i * incx];
				++i;
			}
		}
		
		z[k + 1] = numeric_limits<Scalar>::infinity();
	}
	
	for (int i = 0, k = 0; i < n; ++i) {
		while (z[k + 1] < (i + offset) * 2)
			++k;
		
		y[i * incy] = (a * (i + offset - v[k]) + b) * (i + offset - v[k]) + x[v[k] * incx];
		
		if (m)
			m[i * incm] = v[k];
	}
}

void Model::DT2D(HOGPyramid::Matrix & matrix, const Model::Part & part, Scalar * tmp,
				 Positions * positions)
{
	// Nothing to do if the matrix is empty
	if (!matrix.size())
		return;
	
	const int rows = matrix.rows();
	const int cols = matrix.cols();
	
	if (positions)
		positions->resize(rows, cols);

	// Temporary vectors
	vector<Scalar> z(max(rows, cols) + 1);
	vector<int> v(max(rows, cols) + 1);
	vector<Scalar> t(max(rows, cols));
	
	t[0] = numeric_limits<Scalar>::infinity();
	
	for (int x = 1; x < cols; ++x)
		t[x] = 1 / (part.deformation(0) * x);
	
	// Filter the rows in tmp
	for (int y = 0; y < rows; ++y)
		DT1D(matrix.row(y).data(), cols, part.deformation(0), part.deformation(1), &z[0], &v[0],
			 tmp + y * cols, positions ? positions->row(y).data()->data() : 0, part.offset(0),
			 &t[0], 1, 1, 2);
	
	for (int y = 1; y < rows; ++y)
		t[y] = 1 / (part.deformation(2) * y);
	
	// Filter the columns back to the original matrix
	for (int x = 0; x < cols; x += 2)
		DT1D(tmp + x, rows, part.deformation(2), part.deformation(3), &z[0], &v[0],
			 matrix.data() + x, positions ? ((positions->data() + x)->data() + 1) : 0,
			 part.offset(1), &t[0], cols, cols, cols * 2);
	
	// Re-index the best x positions now that the best y changed
	if (positions) {
		Map<MatrixXi> tmp2(reinterpret_cast<int *>(tmp), rows, cols);
		
		for (int y = 0; y < rows; ++y)
			for (int x = 0; x < cols; x += 2)
				tmp2(y, x) = (*positions)(y, x)(0);
		
		for (int y = 0; y < rows; ++y)
			for (int x = 0; x < cols; x += 2)
				(*positions)(y, x)(0) = tmp2((*positions)(y, x)(1), x);
	}
}

ostream & FFLD::operator<<(ostream & os, const Model & model)
{
	// Save the number of parts and the bias
	os << model.parts_.size() << ' ' << model.bias_ << endl;
	
	// Save the parts themselves
	for (int i = 0; i < model.parts_.size(); ++i) {
		os << model.parts_[i].filter.rows() << ' ' << model.parts_[i].filter.cols() << ' '
		   << HOGPyramid::NbFeatures << ' ' << model.parts_[i].offset(0) << ' '
		   << model.parts_[i].offset(1) << ' ' << model.parts_[i].deformation(0) << ' '
		   << model.parts_[i].deformation(1) << ' ' << model.parts_[i].deformation(2) << ' '
		   << model.parts_[i].deformation(3) << endl;
		
		for (int y = 0; y < model.parts_[i].filter.rows(); ++y) {
			for (int x = 0; x < model.parts_[i].filter.cols(); ++x)
				for (int j = 0; j < HOGPyramid::NbFeatures; ++j)
					os << model.parts_[i].filter(y, x)(j) << ' ';
			
			os << endl;
		}
	}
	
	return os;
}

istream & FFLD::operator>>(istream & is, Model & model)
{
	int nbParts;
	Model::Scalar bias;
	is >> nbParts >> bias;
	
	if (!is) {
		model = Model();
		return is;
	}
	
	model.parts_.resize(nbParts);
	model.bias_ = bias;
	
	for (int i = 0; i < nbParts; ++i) {
		int rows, cols, nbFeatures;
		
		is >> rows >> cols >> nbFeatures >> model.parts_[i].offset(0) >> model.parts_[i].offset(1)
		   >> model.parts_[i].deformation(0) >> model.parts_[i].deformation(1)
		   >> model.parts_[i].deformation(2) >> model.parts_[i].deformation(3);
		
		// Always set the deformation of the root to zero
		if (!i)
			model.parts_[0].deformation.setZero();
		
		model.parts_[i].filter.resize(rows, cols);
		
		for (int y = 0; y < rows; ++y) {
			for (int x = 0; x < cols; ++x) {
				for (int j = 0; j < nbFeatures; ++j) {
					Model::Scalar f;
					is >> f;
					
					if (j < HOGPyramid::NbFeatures)
						model.parts_[i].filter(y, x)(j) = f;
				}
			}
		}
	}
	
	if (!is)
		model = Model();
	
	return is;
}
