#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <vector>
#include <algorithm>
#include "io_png.h"
using namespace std;
#include <iostream>
#define DTYPE int
#define H 5
#define W 5
#define K2 2
#define N2 2
#define SIZE_H_PAD  (H+2*N2)
#define SIZE_W_PAD  (W+2*N2)
#define SQRT2     1.414213562373095
#define SQRT2_INV 0.7071067811865475

void symetrize(
	const std::vector<float>& img
	, std::vector<float>& img_sym
	, const unsigned width
	, const unsigned height
	, const unsigned chnls
	, const unsigned N
) {
	//! Declaration
	const unsigned w = width + 2 * N;
	const unsigned h = height + 2 * N;

	if (img_sym.size() != w * h * chnls)
		img_sym.resize(w * h * chnls);

	for (unsigned c = 0; c < chnls; c++)
	{
		unsigned dc = c * width * height;
		unsigned dc_2 = c * w * h + N * w + N;

		//! Center of the image
		for (unsigned i = 0; i < height; i++)
			for (unsigned j = 0; j < width; j++, dc++)
				img_sym[dc_2 + i * w + j] = img[dc];

		//! Top and bottom
		dc_2 = c * w * h;
		for (unsigned j = 0; j < w; j++, dc_2++)
			for (unsigned i = 0; i < N; i++)
			{
				img_sym[dc_2 + i * w] = img_sym[dc_2 + (2 * N - i - 1) * w];
				img_sym[dc_2 + (h - i - 1) * w] = img_sym[dc_2 + (h - 2 * N + i) * w];
			}

		//! Right and left
		dc_2 = c * w * h;
		for (unsigned i = 0; i < h; i++)
		{
			const unsigned di = dc_2 + i * w;
			for (unsigned j = 0; j < N; j++)
			{
				img_sym[di + j] = img_sym[di + 2 * N - j - 1];
				img_sym[di + w - j - 1] = img_sym[di + w - 2 * N + j];
			}
		}
	}

	return;
}
/**
 * @brief Look for the closest power of 2 number
 *
 * @param n: number
 *
 * @return the closest power of 2 lower or equal to n
 **/
int closest_power_of_2(
	const unsigned n
) {
	unsigned r = 1;
	while (r * 2 <= n)
		r *= 2;

	return r;
}

bool ComparaisonFirst(pair<float, unsigned> pair1, pair<float, unsigned> pair2)
{
	return pair1.first < pair2.first;
}
/**
 * @brief Load image, check the number of channels
 *
 * @param name : name of the image to read
 * @param img : vector which will contain the image : R, G and B concatenated
 * @param width, height, chnls : size of the image
 *
 * @return EXIT_SUCCESS if the image has been loaded, EXIT_FAILURE otherwise
 **/
int load_image(
	char* name
	, vector<float> & img
	, unsigned* width
	, unsigned* height
	, unsigned* chnls
) {
	//! read input image
	cout << endl << "Read input image...";
	size_t h, w, c;
	float* tmp = NULL;
	tmp = read_png_f32(name, &w, &h, &c);
	if (!tmp)
	{
		cout << "error :: " << name << " not found or not a correct png image" << endl;
		return EXIT_FAILURE;
	}
	cout << "done." << endl;

	//! test if image is really a color image and exclude the alpha channel
	if (c > 2)
	{
		unsigned k = 0;
		while (k < w * h && tmp[k] == tmp[w * h + k] && tmp[k] == tmp[2 * w * h + k])
			k++;
		c = (k == w * h ? 1 : 3);
	}

	//! Some image informations
	cout << "image size :" << endl;
	cout << " - width          = " << w << endl;
	cout << " - height         = " << h << endl;
	cout << " - nb of channels = " << c << endl;

	//! Initializations
	*width = w;
	*height = h;
	*chnls = c;
	img.resize(w * h * c);
	for (unsigned k = 0; k < w * h * c; k++)
		img[k] = tmp[k];

	return EXIT_SUCCESS;
}

void ind_initialize(
	vector<unsigned>& ind_set
	, const unsigned max_size
	, const unsigned N
	, const unsigned step
) {
	ind_set.clear();
	unsigned ind = N;
	while (ind < max_size - N)
	{
		ind_set.push_back(ind);
		ind += step;
	}
	if (ind_set.back() < max_size - N - 1)
		ind_set.push_back(max_size - N - 1);
}

/**
 * @brief Precompute Bloc Matching (distance inter-patches)
 *
 * @param patch_table: for each patch in the image, will contain
 * all coordonnate of its similar patches
 * @param img: noisy image on which the distance is computed
 * @param width, height: size of img
 * @param kHW: size of patch
 * @param NHW: maximum similar patches wanted
 * @param nHW: size of the boundary of img
 * @param tauMatch: threshold used to determinate similarity between
 *        patches
 *
 * @return none.
 **/
void precompute_BM(
	vector<vector<unsigned> >& patch_table
	, const vector<float>& img
	, const unsigned width
	, const unsigned height
	, const unsigned kHW
	, const unsigned NHW
	, const unsigned nHW
	, const unsigned pHW
	, const float    tauMatch
) {
	//! Declarations
	const unsigned Ns = 2 * nHW + 1;
	//const float threshold = tauMatch * kHW * kHW;
	const float threshold = 40000;
	cout << threshold;
	vector<float> diff_table(width * height);
	vector<vector<float> > sum_table((nHW + 1) * Ns, vector<float>(width * height, 2 * threshold));
	if (patch_table.size() != width * height)
		patch_table.resize(width * height);
	vector<unsigned> row_ind;
	ind_initialize(row_ind, height - kHW + 1, nHW, pHW);
	vector<unsigned> column_ind;
	ind_initialize(column_ind, width - kHW + 1, nHW, pHW);

	for (vector<unsigned>::iterator iter = row_ind.begin(); iter != row_ind.end(); iter++)
	{
		cout << (*iter) << "	row_ind"  << endl;
	}
	for (vector<unsigned>::iterator iter = column_ind.begin(); iter != column_ind.end(); iter++)
	{
		cout << (*iter) << "	column_ind" << endl;
	}

	//! For each possible distance, precompute inter-patches distance
	for (unsigned di = 0; di <= nHW; di++)
		for (unsigned dj = 0; dj < Ns; dj++)
		{
			const int dk = (int)(di * width + dj) - (int)nHW;
			
			const unsigned ddk = di * Ns + dj;
			
			//! Process the image containing the square distance between pixels
			for (unsigned i = nHW; i < height - nHW; i++)
			{
				unsigned k = i * width + nHW;
				
				for (unsigned j = nHW; j < width - nHW; j++, k++)
					
					diff_table[k] = (img[k + dk] - img[k]) * (img[k + dk] - img[k]);
			
			}
			

			//! Compute the sum for each patches, using the method of the integral images
			const unsigned dn = nHW * width + nHW;
			//! 1st patch, top left corner
			float value = 0.0f;
			for (unsigned p = 0; p < kHW; p++)
			{
				unsigned pq = p * width + dn;
				
				for (unsigned q = 0; q < kHW; q++, pq++)
					value += diff_table[pq];
			}
			
			sum_table[ddk][dn] = value;

			

			//! 1st row, top
			for (unsigned j = nHW + 1; j < width - nHW; j++)
			{
				const unsigned ind = nHW * width + j - 1;
				float sum = sum_table[ddk][ind];
				for (unsigned p = 0; p < kHW; p++)
					sum += diff_table[ind + p * width + kHW] - diff_table[ind + p * width];
				sum_table[ddk][ind + 1] = sum;
				//cout <<"ind:" <<ind+1 << " " <<sum_table[ddk][ind + 1]<<" ";
			}

			//! General case
			for (unsigned i = nHW + 1; i < height - nHW; i++)
			{
				const unsigned ind = (i - 1) * width + nHW;
				float sum = sum_table[ddk][ind];
				//! 1st column, left
				for (unsigned q = 0; q < kHW; q++)
					sum += diff_table[ind + kHW * width + q] - diff_table[ind + q];
				sum_table[ddk][ind + width] = sum;

				//! Other columns
				unsigned k = i * width + nHW + 1;
				unsigned pq = (i + kHW - 1) * width + kHW - 1 + nHW + 1;
				for (unsigned j = nHW + 1; j < width - nHW; j++, k++, pq++)
				{
					sum_table[ddk][k] =
						sum_table[ddk][k - 1]
						+ sum_table[ddk][k - width]
						- sum_table[ddk][k - 1 - width]
						+ diff_table[pq]
						- diff_table[pq - kHW]
						- diff_table[pq - kHW * width]
						+ diff_table[pq - kHW - kHW * width];
					//cout << "ind:" << k << " " << sum_table[ddk][k] << " ";
				}
				
			}
		}
	/*for (vector<vector<float>>::iterator it = sum_table.begin(); it != sum_table.end(); ++it) {
		cout << (*it)[138]<<" ";
	}*/
	cout << "duandian";
	//! Precompute Bloc Matching
	vector<pair<float, unsigned> > table_distance;
	//! To avoid reallocation
	table_distance.reserve(Ns * Ns);


	for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
	{
		for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
		{
			//! Initialization
			const unsigned k_r = row_ind[ind_i] * width + column_ind[ind_j];
			
			table_distance.clear();
			patch_table[k_r].clear();

			//! Threshold distances in order to keep similar patches
			for (int dj = -(int)nHW; dj <= (int)nHW; dj++)
			{
				for (int di = 0; di <= (int)nHW; di++){
					if (sum_table[dj + nHW + di * Ns][k_r] < threshold)
						table_distance.push_back(make_pair(
							sum_table[dj + nHW + di * Ns][k_r]
							, k_r + di * width + dj));
					//cout << k_r << ":" << sum_table[dj + nHW + di * Ns][k_r] << " " << "k_r + di * width + dj" << (k_r + di * width + dj)<< " ;";
				}

				for (int di = -(int)nHW; di < 0; di++)
					if (sum_table[-dj + nHW + (-di) * Ns][k_r] < threshold)
						table_distance.push_back(make_pair(
							sum_table[-dj + nHW + (-di) * Ns][k_r + di * width + dj]
							, k_r + di * width + dj));
			}

			//! We need a power of 2 for the number of similar patches,
			//! because of the Welsh-Hadamard transform on the third dimension.
			//! We assume that NHW is already a power of 2
			const unsigned nSx_r = (NHW > table_distance.size() ?
				closest_power_of_2(table_distance.size()) : NHW);
			//cout << " nSx_r" << nSx_r << endl;

			//! To avoid problem
			if (nSx_r == 1 && table_distance.size() == 0)
			{
				cout << "problem size" << endl;
				table_distance.push_back(make_pair(0, k_r));
			}

			//! Sort patches according to their distance to the reference one
			partial_sort(table_distance.begin(), table_distance.begin() + nSx_r,
				table_distance.end(), ComparaisonFirst);

			//! Keep a maximum of NHW similar patches
			for (unsigned n = 0; n < nSx_r; n++) {
				patch_table[k_r].push_back(table_distance[n].second); //第k_r个中心点 加入对应 相似块中心坐标
				cout << "K_r" << k_r << " " << table_distance[n].second << ",value: " << table_distance[n].first << "   ";
			}
			//! To avoid problem
			if (nSx_r == 1)
				patch_table[k_r].push_back(table_distance[0].second);
		}
	
	}
	//for (vector<vector<unsigned>>::iterator it = patch_table.begin(); it != patch_table.end(); ++it) {
	//	int kk = 0;
	//	kk++;
	//	//cout << "k:" << kk<<endl;
	//	for (int i = 0; i < (*it).size(); ++i)
	//		cout << "i:" << i << " " << (*it)[i] << " ";
	//};
}



int main(int argc, char** argv) {
	//! Declarations
	float sigma = 10.0f;
	vector<float> img,img_33;
	vector<float> img_basic;
	vector<float> img_sym_test;
	
	unsigned w, h, chnls;


	load_image(argv[1], img, &w, &h, &chnls);
	for (int i = 0; i < 10; ++i)
		for (int j = 0; j < 10; j++)
			img_33.push_back(img[i*w+j]);

	const unsigned nHard = 6; //! Half size of the search window
	const unsigned nWien = 6; //! Half size of the search window
	const unsigned kHard = 5; //! Must be a power of 2 if tau_2D_hard == BIOR
	const unsigned kWien = 5; //! Must be a power of 2 if tau_2D_wien == BIOR
	const unsigned NHard = 16; //! Must be a power of 2
	const unsigned NWien = 32; //! Must be a power of 2
	const unsigned pHard = 3;
	const unsigned pWien = 3;
	const float    lambdaHard3D = 2.7f;            //! Threshold for Hard Thresholding
	const float    tauMatch = (chnls == 1 ? 3.f : 1.f) * (sigma < 35.0f ? 2500 : 5000); //! threshold used to determinate similarity between patches

	

	const unsigned h_b = (unsigned)10 + 2 * nHard;
	const unsigned w_b = (unsigned)10 + 2 * nHard;
	vector<float> img_sym_noisy, img_sym_basic, img_sym_denoised;
	symetrize(img_33, img_sym_noisy, 10, 10, 1, 6);
	
	 vector<vector<unsigned> > patch_table;
	if (img_basic.size() != img.size())
		img_basic.resize(img.size());


	precompute_BM(patch_table, img_sym_noisy, w_b, h_b, kWien, NHard, nWien, pWien, tauMatch);

	/*vector<float> testim = { 1,2,3,4,5,6,7, 1,2,3,4,5,6,7, 1,2,3,4,5,6,7, 1,2,3,4,5,6,7, 1,2,3,4,5,6,7, 1,2,3,4,5,6,7, 1,2,3,4,5,6,7 };
	symetrize(testim, img_sym_test, 7, 7, 1, 5);*/
	
	//
	//for (vector<float>::iterator iter = img_sym_test.begin(); iter != img_sym_test.end(); iter++)
	//{
	//	cout << (*iter)<< "   ";
	//}
	//cout << img_sym_test[53]<<"size";

	/*for (vector<float>::iterator iter = img_sym_noisy.begin(); iter != img_sym_noisy.end(); iter++)
	{
		cout << (*iter) << "   ";
	}*/
	
	
	
	

		




	cout << "end";
}
