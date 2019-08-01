#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <vector>
#include <algorithm>
#include "io_png.h"
using namespace std;
#include <iostream>
#define DTYPE int
#define H 384
#define W 512
#define K2 4
#define N2 16
#define nb_threads 1
#define SIZE_H_PAD  (H+2*N2)
#define SIZE_W_PAD  (W+2*N2)
#define SQRT2     1.414213562373095
#define SQRT2_INV 0.7071067811865475

void symetrize(
	const std::vector<float> & img
	, std::vector<float> & img_sym
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

/**
 * @brief For convenience. Estimate the size of the ind_set vector built
 *        with the function ind_initialize().
 *
 * @return size of ind_set vector built in ind_initialize().
 **/
unsigned ind_size(
	const unsigned max_size
	, const unsigned N
	, const unsigned step
) {
	unsigned ind = N;
	unsigned k = 0;
	while (ind < max_size - N)
	{
		k++;
		ind += step;
	}
	if (ind - step < max_size - N - 1)
		k++;

	return k;
}

/**
 * @brief Initialize a 2D fftwf_plan with some parameters
 *
 * @param plan: fftwf_plan to allocate;
 * @param N: size of the patch to apply the 2D transform;
 * @param kind: forward or backward;
 * @param nb: number of 2D transform which will be processed.
 *
 * @return none.
 **/
void allocate_plan_2d(
	fftwf_plan* plan
	, const unsigned N
	, const fftwf_r2r_kind kind
	, const unsigned nb
) {
	int            nb_table[2] = { N, N };
	int            nembed[2] = { N, N };
	fftwf_r2r_kind kind_table[2] = { kind, kind };

	float* vec = (float*)fftwf_malloc(N * N * nb * sizeof(float));
	(*plan) = fftwf_plan_many_r2r(2, nb_table, nb, vec, nembed, 1, N * N, vec,
		nembed, 1, N * N, kind_table, FFTW_ESTIMATE);

	fftwf_free(vec);
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
	, vector<float>& img
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
	//cout << threshold;
	vector<float> diff_table(width * height);
	vector<vector<float> > sum_table((nHW + 1) * Ns, vector<float>(width * height, 2 * threshold));
	if (patch_table.size() != width * height)
		patch_table.resize(width * height);
	vector<unsigned> row_ind;
	ind_initialize(row_ind, height - kHW + 1, nHW, pHW);
	vector<unsigned> column_ind;
	ind_initialize(column_ind, width - kHW + 1, nHW, pHW);

	//cout << row_ind.size() << " " << column_ind.size()<< " ";
	//for (vector<unsigned>::iterator iter = row_ind.begin(); iter != row_ind.end(); iter++)
	//{
	//	cout << (*iter) << "	row_ind" << endl;
	//}
	//for (vector<unsigned>::iterator iter = column_ind.begin(); iter != column_ind.end(); iter++)
	//{
	//	cout << (*iter) << "	column_ind" << endl;
	//}

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

	//! Precompute Bloc Matching
	vector<pair<float, unsigned> > table_distance;
	//! To avoid reallocation
	table_distance.reserve(Ns * Ns);


	for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
	//for (unsigned ind_i = 0; ind_i < 1; ind_i++)
	{
		for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
		//for (unsigned ind_j = 0; ind_j < 1; ind_j++)
		{
			//! Initialization
			const unsigned k_r = row_ind[ind_i] * width + column_ind[ind_j];

			table_distance.clear();
			patch_table[k_r].clear();

			//! Threshold distances in order to keep similar patches
			for (int dj = -(int)nHW; dj <= (int)nHW; dj++)
			{
				for (int di = 0; di <= (int)nHW; di++) {
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
				//cout << "K_r" << k_r << " " << table_distance[n].second << ",value: " << table_distance[n].first << "   ";
			}
			//! To avoid problem
			if (nSx_r == 1)
				patch_table[k_r].push_back(table_distance[0].second);
		}

	}
}



void preProcess(
	vector<float>& kaiserWindow
	, vector<float>& coef_norm
	, vector<float>& coef_norm_inv
	, const unsigned kHW
) {
	//! Kaiser Window coefficients
	if (kHW == 8)
	{
		//! First quarter of the matrix
		kaiserWindow[0 + kHW * 0] = 0.1924f; kaiserWindow[0 + kHW * 1] = 0.2989f; kaiserWindow[0 + kHW * 2] = 0.3846f; kaiserWindow[0 + kHW * 3] = 0.4325f;
		kaiserWindow[1 + kHW * 0] = 0.2989f; kaiserWindow[1 + kHW * 1] = 0.4642f; kaiserWindow[1 + kHW * 2] = 0.5974f; kaiserWindow[1 + kHW * 3] = 0.6717f;
		kaiserWindow[2 + kHW * 0] = 0.3846f; kaiserWindow[2 + kHW * 1] = 0.5974f; kaiserWindow[2 + kHW * 2] = 0.7688f; kaiserWindow[2 + kHW * 3] = 0.8644f;
		kaiserWindow[3 + kHW * 0] = 0.4325f; kaiserWindow[3 + kHW * 1] = 0.6717f; kaiserWindow[3 + kHW * 2] = 0.8644f; kaiserWindow[3 + kHW * 3] = 0.9718f;

		//! Completing the rest of the matrix by symmetry
		for (unsigned i = 0; i < kHW / 2; i++)
			for (unsigned j = kHW / 2; j < kHW; j++)
				kaiserWindow[i + kHW * j] = kaiserWindow[i + kHW * (kHW - j - 1)];

		for (unsigned i = kHW / 2; i < kHW; i++)
			for (unsigned j = 0; j < kHW; j++)
				kaiserWindow[i + kHW * j] = kaiserWindow[kHW - i - 1 + kHW * j];
	}
	else if (kHW == 12)
	{
		//! First quarter of the matrix
		kaiserWindow[0 + kHW * 0] = 0.1924f; kaiserWindow[0 + kHW * 1] = 0.2615f; kaiserWindow[0 + kHW * 2] = 0.3251f; kaiserWindow[0 + kHW * 3] = 0.3782f;  kaiserWindow[0 + kHW * 4] = 0.4163f;  kaiserWindow[0 + kHW * 5] = 0.4362f;
		kaiserWindow[1 + kHW * 0] = 0.2615f; kaiserWindow[1 + kHW * 1] = 0.3554f; kaiserWindow[1 + kHW * 2] = 0.4419f; kaiserWindow[1 + kHW * 3] = 0.5139f;  kaiserWindow[1 + kHW * 4] = 0.5657f;  kaiserWindow[1 + kHW * 5] = 0.5927f;
		kaiserWindow[2 + kHW * 0] = 0.3251f; kaiserWindow[2 + kHW * 1] = 0.4419f; kaiserWindow[2 + kHW * 2] = 0.5494f; kaiserWindow[2 + kHW * 3] = 0.6390f;  kaiserWindow[2 + kHW * 4] = 0.7033f;  kaiserWindow[2 + kHW * 5] = 0.7369f;
		kaiserWindow[3 + kHW * 0] = 0.3782f; kaiserWindow[3 + kHW * 1] = 0.5139f; kaiserWindow[3 + kHW * 2] = 0.6390f; kaiserWindow[3 + kHW * 3] = 0.7433f;  kaiserWindow[3 + kHW * 4] = 0.8181f;  kaiserWindow[3 + kHW * 5] = 0.8572f;
		kaiserWindow[4 + kHW * 0] = 0.4163f; kaiserWindow[4 + kHW * 1] = 0.5657f; kaiserWindow[4 + kHW * 2] = 0.7033f; kaiserWindow[4 + kHW * 3] = 0.8181f;  kaiserWindow[4 + kHW * 4] = 0.9005f;  kaiserWindow[4 + kHW * 5] = 0.9435f;
		kaiserWindow[5 + kHW * 0] = 0.4362f; kaiserWindow[5 + kHW * 1] = 0.5927f; kaiserWindow[5 + kHW * 2] = 0.7369f; kaiserWindow[5 + kHW * 3] = 0.8572f;  kaiserWindow[5 + kHW * 4] = 0.9435f;  kaiserWindow[5 + kHW * 5] = 0.9885f;

		//! Completing the rest of the matrix by symmetry
		for (unsigned i = 0; i < kHW / 2; i++)
			for (unsigned j = kHW / 2; j < kHW; j++)
				kaiserWindow[i + kHW * j] = kaiserWindow[i + kHW * (kHW - j - 1)];

		for (unsigned i = kHW / 2; i < kHW; i++)
			for (unsigned j = 0; j < kHW; j++)
				kaiserWindow[i + kHW * j] = kaiserWindow[kHW - i - 1 + kHW * j];
	}
	else
		for (unsigned k = 0; k < kHW * kHW; k++)
			kaiserWindow[k] = 1.0f;

	//! Coefficient of normalization for DCT II and DCT II inverse
	const float coef = 0.5f / ((float)(kHW));
	for (unsigned i = 0; i < kHW; i++)
		for (unsigned j = 0; j < kHW; j++)
		{
			if (i == 0 && j == 0)
			{
				coef_norm[i * kHW + j] = 0.5f * coef;
				coef_norm_inv[i * kHW + j] = 2.0f;
			}
			else if (i * j == 0)
			{
				coef_norm[i * kHW + j] = SQRT2_INV * coef;
				coef_norm_inv[i * kHW + j] = SQRT2;
			}
			else
			{
				coef_norm[i * kHW + j] = 1.0f * coef;
				coef_norm_inv[i * kHW + j] = 1.0f;
			}
		}
}

/**
 * @brief Precompute a 2D DCT transform on all patches contained in
 *        a part of the image.
 *
 * @param DCT_table_2D : will contain the 2d DCT transform for all
 *        chosen patches;
 * @param img : image on which the 2d DCT will be processed;
 * @param plan_1, plan_2 : for convenience. Used by fftw;
 * @param nHW : size of the boundary around img;
 * @param width, height, chnls: size of img;
 * @param kHW : size of patches (kHW x kHW);
 * @param i_r: current index of the reference patches;
 * @param step: space in pixels between two references patches;
 * @param coef_norm : normalization coefficients of the 2D DCT;
 * @param i_min (resp. i_max) : minimum (resp. maximum) value
 *        for i_r. In this case the whole 2d transform is applied
 *        on every patches. Otherwise the precomputed 2d DCT is re-used
 *        without processing it.
 **/
void dct_2d_process(
	vector<float>& DCT_table_2D
	, vector<float> const& img
	, fftwf_plan* plan_1
	, fftwf_plan* plan_2
	, const unsigned nHW
	, const unsigned width
	, const unsigned height
	, const unsigned chnls
	, const unsigned kHW
	, const unsigned i_r
	, const unsigned step
	, vector<float> const& coef_norm
	, const unsigned i_min
	, const unsigned i_max
) {
	// !Declarations
	const unsigned kHW_2 = kHW * kHW;
	const unsigned size = chnls * kHW_2 * width * (2 * nHW + 1);

	//! If i_r == ns, then we have to process all DCT
	if (i_r == i_min || i_r == i_max)
	{
		//! Allocating Memory
		float* vec = (float*)fftwf_malloc(size * sizeof(float));
		float* dct = (float*)fftwf_malloc(size * sizeof(float));

		for (unsigned c = 0; c < chnls; c++)
		{
			const unsigned dc = c * width * height;
			const unsigned dc_p = c * kHW_2 * width * (2 * nHW + 1);
			for (unsigned i = 0; i < 2 * nHW + 1; i++)
				for (unsigned j = 0; j < width - kHW; j++)
					for (unsigned p = 0; p < kHW; p++)
						for (unsigned q = 0; q < kHW; q++)
							vec[p * kHW + q + dc_p + (i * width + j) * kHW_2] =
							img[dc + (i_r + i - nHW + p) * width + j + q];
		}

		//! Process of all DCTs
		fftwf_execute_r2r(*plan_1, vec, dct);
		fftwf_free(vec);

		//! Getting the result
		for (unsigned c = 0; c < chnls; c++)
		{
			const unsigned dc = c * kHW_2 * width * (2 * nHW + 1);
			const unsigned dc_p = c * kHW_2 * width * (2 * nHW + 1);
			for (unsigned i = 0; i < 2 * nHW + 1; i++)
				for (unsigned j = 0; j < width - kHW; j++)
					for (unsigned k = 0; k < kHW_2; k++)
						DCT_table_2D[dc + (i * width + j) * kHW_2 + k] =
						dct[dc_p + (i * width + j) * kHW_2 + k] * coef_norm[k];
		}
		fftwf_free(dct);
	}
	else
	{
		const unsigned ds = step * width * kHW_2;

		//! Re-use of DCT already processed
		for (unsigned c = 0; c < chnls; c++)
		{
			unsigned dc = c * width * (2 * nHW + 1) * kHW_2;
			for (unsigned i = 0; i < 2 * nHW + 1 - step; i++)
				for (unsigned j = 0; j < width - kHW; j++)
					for (unsigned k = 0; k < kHW_2; k++)
						DCT_table_2D[k + (i * width + j) * kHW_2 + dc] =
						DCT_table_2D[k + (i * width + j) * kHW_2 + dc + ds];
		}

		//! Compute the new DCT
		float* vec = (float*)fftwf_malloc(chnls * kHW_2 * step * width * sizeof(float));
		float* dct = (float*)fftwf_malloc(chnls * kHW_2 * step * width * sizeof(float));

		for (unsigned c = 0; c < chnls; c++)
		{
			const unsigned dc = c * width * height;
			const unsigned dc_p = c * kHW_2 * width * step;
			for (unsigned i = 0; i < step; i++)
				for (unsigned j = 0; j < width - kHW; j++)
					for (unsigned p = 0; p < kHW; p++)
						for (unsigned q = 0; q < kHW; q++)
							vec[p * kHW + q + dc_p + (i * width + j) * kHW_2] =
							img[(p + i + 2 * nHW + 1 - step + i_r - nHW)
							* width + j + q + dc];
		}

		//! Process of all DCTs
		fftwf_execute_r2r(*plan_2, vec, dct);
		fftwf_free(vec);

		//! Getting the result
		for (unsigned c = 0; c < chnls; c++)
		{
			const unsigned dc = c * kHW_2 * width * (2 * nHW + 1);
			const unsigned dc_p = c * kHW_2 * width * step;
			for (unsigned i = 0; i < step; i++)
				for (unsigned j = 0; j < width - kHW; j++)
					for (unsigned k = 0; k < kHW_2; k++)
						DCT_table_2D[dc + ((i + 2 * nHW + 1 - step) * width + j) * kHW_2 + k] =
						dct[dc_p + (i * width + j) * kHW_2 + k] * coef_norm[k];
		}
		fftwf_free(dct);
	}

}


void dct_2d_inverse(
	vector<float>& group_3D_table
	, const unsigned kHW
	, const unsigned N
	, vector<float> const& coef_norm_inv
	, fftwf_plan* plan
) {
	//! Declarations
	const unsigned kHW_2 = kHW * kHW;
	const unsigned size = kHW_2 * N;
	const unsigned Ns = group_3D_table.size() / kHW_2;

	//! Allocate Memory
	float* vec = (float*)fftwf_malloc(size * sizeof(float));
	float* dct = (float*)fftwf_malloc(size * sizeof(float));

	//! Normalization
	for (unsigned n = 0; n < Ns; n++)
		for (unsigned k = 0; k < kHW_2; k++)
			dct[k + n * kHW_2] = group_3D_table[k + n * kHW_2] * coef_norm_inv[k];

	//! 2D dct inverse
	fftwf_execute_r2r(*plan, dct, vec);
	fftwf_free(dct);

	//! Getting the result + normalization
	const float coef = 1.0f / (float)(kHW * 2);
	for (unsigned k = 0; k < group_3D_table.size(); k++)
		group_3D_table[k] = coef * vec[k];

	//! Free Memory
	fftwf_free(vec);
}


void per_ext_ind(
	vector<unsigned>& ind_per
	, const unsigned N
	, const unsigned L
) {
	for (unsigned k = 0; k < N; k++)
		ind_per[k + L] = k;

	int ind1 = (N - L);
	while (ind1 < 0)
		ind1 += N;
	unsigned ind2 = 0;
	unsigned k = 0;
	while (k < L)
	{
		ind_per[k] = (unsigned)ind1;
		ind_per[k + L + N] = ind2;
		ind1 = ((unsigned)ind1 < N - 1 ? (unsigned)ind1 + 1 : 0);
		ind2 = (ind2 < N - 1 ? ind2 + 1 : 0);
		k++;
	}
}


void bior_2d_forward(
	vector<float> const& input
	, vector<float>& output
	, const unsigned N
	, const unsigned d_i
	, const unsigned r_i
	, const unsigned d_o
	, vector<float> const& lpd
	, vector<float> const& hpd
) {
	//! Initializing output
	for (unsigned i = 0; i < N; i++)
		for (unsigned j = 0; j < N; j++)
			output[i * N + j + d_o] = input[i * r_i + j + d_i];

	const unsigned iter_max = log2(N);
	unsigned N_1 = N;
	unsigned N_2 = N / 2;
	const unsigned S_1 = lpd.size();
	const unsigned S_2 = S_1 / 2 - 1;

	for (unsigned iter = 0; iter < iter_max; iter++)
	{
		//! Periodic extension index initialization
		vector<float> tmp(N_1 + 2 * S_2);
		vector<unsigned> ind_per(N_1 + 2 * S_2);
		per_ext_ind(ind_per, N_1, S_2);

		//! Implementing row filtering
		for (unsigned i = 0; i < N_1; i++)
		{
			//! Periodic extension of the signal in row
			for (unsigned j = 0; j < tmp.size(); j++)
				tmp[j] = output[d_o + i * N + ind_per[j]];

			//! Low and High frequencies filtering
			for (unsigned j = 0; j < N_2; j++)
			{
				float v_l = 0.0f, v_h = 0.0f;
				for (unsigned k = 0; k < S_1; k++)
				{
					v_l += tmp[k + j * 2] * lpd[k];
					v_h += tmp[k + j * 2] * hpd[k];
				}
				output[d_o + i * N + j] = v_l;
				output[d_o + i * N + j + N_2] = v_h;
			}
		}

		//! Implementing column filtering
		for (unsigned j = 0; j < N_1; j++)
		{
			//! Periodic extension of the signal in column
			for (unsigned i = 0; i < tmp.size(); i++)
				tmp[i] = output[d_o + j + ind_per[i] * N];

			//! Low and High frequencies filtering
			for (unsigned i = 0; i < N_2; i++)
			{
				float v_l = 0.0f, v_h = 0.0f;
				for (unsigned k = 0; k < S_1; k++)
				{
					v_l += tmp[k + i * 2] * lpd[k];
					v_h += tmp[k + i * 2] * hpd[k];
				}
				output[d_o + j + i * N] = v_l;
				output[d_o + j + (i + N_2) * N] = v_h;
			}
		}

		//! Sizes update
		N_1 /= 2;
		N_2 /= 2;
	}
}

void bior_2d_forward_test(
	vector<float> const& input
	, vector<float>& output
	, const unsigned N
	, const unsigned d_i
	, const unsigned r_i
	, const unsigned d_o
	, vector<float> const& lpd
	, vector<float> const& hpd
	, vector<float>& tmp
	, vector<unsigned>& ind_per
) {
	//! Initializing output
	for (unsigned i = 0; i < N; i++)
		for (unsigned j = 0; j < N; j++)
			output[i * N + j + d_o] = input[i * r_i + j + d_i];

	const unsigned iter_max = log2(N);
	unsigned N_1 = N;
	unsigned N_2 = N / 2;
	const unsigned S_1 = lpd.size();
	const unsigned S_2 = S_1 / 2 - 1;

	for (unsigned iter = 0; iter < iter_max; iter++)
	{
		//! Periodic extension index initialization
//        vector<float> tmp(N_1 + 2 * S_2);
//        vector<unsigned> ind_per(N_1 + 2 * S_2);
		per_ext_ind(ind_per, N_1, S_2);

		//! Implementing row filtering
		for (unsigned i = 0; i < N_1; i++)
		{
			//! Periodic extension of the signal in row
			for (unsigned j = 0; j < tmp.size(); j++)
				tmp[j] = output[d_o + i * N + ind_per[j]];

			//! Low and High frequencies filtering
			for (unsigned j = 0; j < N_2; j++)
			{
				float v_l = 0.0f, v_h = 0.0f;
				for (unsigned k = 0; k < S_1; k++)
				{
					v_l += tmp[k + j * 2] * lpd[k];
					v_h += tmp[k + j * 2] * hpd[k];
				}
				output[d_o + i * N + j] = v_l;
				output[d_o + i * N + j + N_2] = v_h;
				//                output[d_o + i * N + j] = inner_product(tmp.begin() + j * 2, tmp.begin() + j * 2 + S_1, lpd.begin(), 0.f);
				//                output[d_o + i * N + j + N_2] = inner_product(tmp.begin() + j * 2, tmp.begin() + j * 2 + S_1, hpd.begin(), 0.f);
			}
		}

		//! Implementing column filtering
		for (unsigned j = 0; j < N_1; j++)
		{
			//! Periodic extension of the signal in column
			for (unsigned i = 0; i < tmp.size(); i++)
				tmp[i] = output[d_o + j + ind_per[i] * N];

			//! Low and High frequencies filtering
			for (unsigned i = 0; i < N_2; i++)
			{
				float v_l = 0.0f, v_h = 0.0f;
				for (unsigned k = 0; k < S_1; k++)
				{
					v_l += tmp[k + i * 2] * lpd[k];
					v_h += tmp[k + i * 2] * hpd[k];
				}
				output[d_o + j + i * N] = v_l;
				output[d_o + j + (i + N_2) * N] = v_h;
				//                output[d_o + j + i * N] = inner_product(tmp.begin() + i * 2, tmp.begin() + i * 2 + S_1, lpd.begin(), 0.f);
				//                output[d_o + j + (i + N_2) * N] = inner_product(tmp.begin() + i * 2, tmp.begin() + i * 2 + S_1, hpd.begin(), 0.f);
			}
		}

		//! Sizes update
		N_1 /= 2;
		N_2 /= 2;
	}
}

/**
 * @brief Compute a full 2D Bior 1.5 spline wavelet inverse (normalized)
 *
 * @param signal: vector on which the transform will be applied; It
 *                will contain the result at the end;
 * @param N: size of the 2D patch (N x N) on which the 2D transform
 *           is applied. Must be a power of 2;
 * @param d_s: for convenience. Shift for signal to access to the patch;
 * @param lpr: low frequencies coefficients for the inverse Bior 1.5;
 * @param hpr: high frequencies coefficients for the inverse Bior 1.5.
 *
 * @return none.
 **/
void bior_2d_inverse(
	vector<float>& signal
	, const unsigned N
	, const unsigned d_s
	, vector<float> const& lpr
	, vector<float> const& hpr
) {
	//! Initialization
	const unsigned iter_max = log2(N);
	unsigned N_1 = 2;
	unsigned N_2 = 1;
	const unsigned S_1 = lpr.size();
	const unsigned S_2 = S_1 / 2 - 1;

	for (unsigned iter = 0; iter < iter_max; iter++)
	{

		vector<float> tmp(N_1 + S_2 * N_1);
		vector<unsigned> ind_per(N_1 + 2 * S_2 * N_2);
		per_ext_ind(ind_per, N_1, S_2 * N_2);

		//! Implementing column filtering
		for (unsigned j = 0; j < N_1; j++)
		{
			//! Periodic extension of the signal in column
			for (unsigned i = 0; i < tmp.size(); i++)
				tmp[i] = signal[d_s + j + ind_per[i] * N];

			//! Low and High frequencies filtering
			for (unsigned i = 0; i < N_2; i++)
			{
				float v_l = 0.0f, v_h = 0.0f;
				for (unsigned k = 0; k < S_1; k++)
				{
					v_l += lpr[k] * tmp[k * N_2 + i];
					v_h += hpr[k] * tmp[k * N_2 + i];
				}

				signal[d_s + i * 2 * N + j] = v_h;
				signal[d_s + (i * 2 + 1) * N + j] = v_l;
			}
		}

		//! Implementing row filtering
		for (unsigned i = 0; i < N_1; i++)
		{
			//! Periodic extension of the signal in row
			for (unsigned j = 0; j < tmp.size(); j++)
				tmp[j] = signal[d_s + i * N + ind_per[j]];

			//! Low and High frequencies filtering
			for (unsigned j = 0; j < N_2; j++)
			{
				float v_l = 0.0f, v_h = 0.0f;
				for (unsigned k = 0; k < S_1; k++)
				{
					v_l += lpr[k] * tmp[k * N_2 + j];
					v_h += hpr[k] * tmp[k * N_2 + j];
				}

				signal[d_s + i * N + j * 2] = v_h;
				signal[d_s + i * N + j * 2 + 1] = v_l;
			}
		}

		//! Sizes update
		N_1 *= 2;
		N_2 *= 2;
	}
}

/**
 * @brief Initialize forward and backward low and high filter
 *        for a Bior1.5 spline wavelet.
 *
 * @param lp1: low frequencies forward filter;
 * @param hp1: high frequencies forward filter;
 * @param lp2: low frequencies backward filter;
 * @param hp2: high frequencies backward filter.
 **/
void bior15_coef(
	vector<float>& lp1
	, vector<float>& hp1
	, vector<float>& lp2
	, vector<float>& hp2
) {
	const float coef_norm = 1.f / (sqrtf(2.f) * 128.f);
	const float sqrt2_inv = 1.f / sqrtf(2.f);

	lp1.resize(10);
	lp1[0] = 3.f;
	lp1[1] = -3.f;
	lp1[2] = -22.f;
	lp1[3] = 22.f;
	lp1[4] = 128.f;
	lp1[5] = 128.f;
	lp1[6] = 22.f;
	lp1[7] = -22.f;
	lp1[8] = -3.f;
	lp1[9] = 3.f;

	hp1.resize(10);
	hp1[0] = 0.f;
	hp1[1] = 0.f;
	hp1[2] = 0.f;
	hp1[3] = 0.f;
	hp1[4] = -sqrt2_inv;
	hp1[5] = sqrt2_inv;
	hp1[6] = 0.f;
	hp1[7] = 0.f;
	hp1[8] = 0.f;
	hp1[9] = 0.f;

	lp2.resize(10);
	lp2[0] = 0.f;
	lp2[1] = 0.f;
	lp2[2] = 0.f;
	lp2[3] = 0.f;
	lp2[4] = sqrt2_inv;
	lp2[5] = sqrt2_inv;
	lp2[6] = 0.f;
	lp2[7] = 0.f;
	lp2[8] = 0.f;
	lp2[9] = 0.f;

	hp2.resize(10);
	hp2[0] = 3.f;
	hp2[1] = 3.f;
	hp2[2] = -22.f;
	hp2[3] = -22.f;
	hp2[4] = 128.f;
	hp2[5] = -128.f;
	hp2[6] = 22.f;
	hp2[7] = 22.f;
	hp2[8] = -3.f;
	hp2[9] = -3.f;

	for (unsigned k = 0; k < 10; k++)
	{
		lp1[k] *= coef_norm;
		hp2[k] *= coef_norm;
	}
}



/**
 * @brief Apply Welsh-Hadamard transform on vec (non normalized !!)
 *
 * @param vec: vector on which a Hadamard transform will be applied.
 *        It will contain the transform at the end;
 * @param tmp: must have the same size as vec. Used for convenience;
 * @param N, d: the Hadamard transform will be applied on vec[d] -> vec[d + N].
 *        N must be a power of 2!!!!
 *
 * @return None.
 **/
void hadamard_transform(
	vector<float>& vec
	, vector<float>& tmp
	, const unsigned N
	, const unsigned D
) {
	if (N == 1)
		return;
	else if (N == 2)
	{
		const float a = vec[D + 0];
		const float b = vec[D + 1];
		vec[D + 0] = a + b;
		vec[D + 1] = a - b;
	}
	else
	{
		const unsigned n = N / 2;
		for (unsigned k = 0; k < n; k++)
		{
			const float a = vec[D + 2 * k];
			const float b = vec[D + 2 * k + 1];
			vec[D + k] = a + b;
			tmp[k] = a - b;
		}
		for (unsigned k = 0; k < n; k++)
			vec[D + n + k] = tmp[k];

		hadamard_transform(vec, tmp, n, D);
		hadamard_transform(vec, tmp, n, D + n);
	}
}

void ht_filtering_hadamard(
	vector<float>& group_3D
	, vector<float>& tmp
	, const unsigned nSx_r
	, const unsigned kHard
	, const unsigned chnls
	, vector<float> const& sigma_table
	, const float lambdaHard3D
	, vector<float>& weight_table
	, const bool doWeight
) {
	//! Declarations
	const unsigned kHard_2 = kHard * kHard;
	for (unsigned c = 0; c < chnls; c++)
		weight_table[c] = 0.0f;
	const float coef_norm = sqrtf((float)nSx_r);
	const float coef = 1.0f / (float)nSx_r;

	//! Process the Welsh-Hadamard transform on the 3rd dimension
	for (unsigned n = 0; n < kHard_2 * chnls; n++)
		hadamard_transform(group_3D, tmp, nSx_r, n * nSx_r);

	//! Hard Thresholding
	for (unsigned c = 0; c < chnls; c++)
	{
		const unsigned dc = c * nSx_r * kHard_2;
		const float T = lambdaHard3D * sigma_table[c] * coef_norm;
		for (unsigned k = 0; k < kHard_2 * nSx_r; k++)
		{
			if (fabs(group_3D[k + dc]) > T)
				weight_table[c]++;
			else
				group_3D[k + dc] = 0.0f;
		}
	}

	//! Process of the Welsh-Hadamard inverse transform
	for (unsigned n = 0; n < kHard_2 * chnls; n++)
		hadamard_transform(group_3D, tmp, nSx_r, n * nSx_r);

	for (unsigned k = 0; k < group_3D.size(); k++)
		group_3D[k] *= coef;

	//! Weight for aggregation
	if (doWeight)
		for (unsigned c = 0; c < chnls; c++)
			weight_table[c] = (weight_table[c] > 0.0f ? 1.0f / (float)
			(sigma_table[c] * sigma_table[c] * weight_table[c]) : 1.0f);
}

/**
 * @brief Compute PSNR and RMSE between img_1 and img_2
 *
 * @param img_1 : pointer to an allocated array of pixels.
 * @param img_2 : pointer to an allocated array of pixels.
 * @param psnr  : will contain the PSNR
 * @param rmse  : will contain the RMSE
 *
 * @return EXIT_FAILURE if both images haven't the same size.
 **/
int compute_psnr(
	const vector<float>& img_1
	, const vector<float>& img_2
	, float* psnr
	, float* rmse
)
{
	if (img_1.size() != img_2.size())
	{
		cout << "Can't compute PSNR & RMSE: images have different sizes: " << endl;
		cout << "img_1 : " << img_1.size() << endl;
		cout << "img_2 : " << img_2.size() << endl;
		return EXIT_FAILURE;
	}

	float tmp = 0.0f;
	for (unsigned k = 0; k < img_1.size(); k++)
		tmp += (img_1[k] - img_2[k]) * (img_1[k] - img_2[k]);

	(*rmse) = sqrtf(tmp / (float)img_1.size());
	(*psnr) = 20.0f * log10f(255.0f / (*rmse));

	return EXIT_SUCCESS;
}

int save_image(
	char* name
	, std::vector<float>& img
	, const unsigned width
	, const unsigned height
	, const unsigned chnls
) {
	//! Allocate Memory
	float* tmp = new float[width * height * chnls];

	//! Check for boundary problems
	for (unsigned k = 0; k < width * height * chnls; k++)
		tmp[k] = (img[k] > 255.0f ? 255.0f : (img[k] < 0.0f ? 0.0f : img[k]));

	if (write_png_f32(name, tmp, width, height, chnls) != 0)
	{
		cout << "... failed to save png image " << name << endl;
		return EXIT_FAILURE;
	}

	//! Free Memory
	delete[] tmp;

	return EXIT_SUCCESS;
}


int main(int argc, char** argv) {
	//! Declarations
	float sigma = 10.0f;
	vector<float> sigma_table(1);
	sigma_table[0] = sigma;
	vector<float> img, img_out;
	vector<float> img_basic;
	vector<float> img_sym_test;

	unsigned w, h, chnls;
	float psnr_basic, rmse_basic, psnr_basic_bias, rmse_basic_bias;
	float psnr, rmse, psnr_bias, rmse_bias;
	load_image(argv[1], img, &w, &h, &chnls);

	const bool     useSD = true;


	const unsigned nHard = 16; //! Half size of the search window
	const unsigned nWien = 16; //! Half size of the search window
	const unsigned kHard = 8; //! Must be a power of 2 if tau_2D_hard == BIOR
	const unsigned kWien = 8; //! Must be a power of 2 if tau_2D_wien == BIOR
	const unsigned NHard = 16; //! Must be a power of 2
	const unsigned NWien = 32; //! Must be a power of 2
	const unsigned pHard = 3;
	const unsigned pWien = 3;
	const float    lambdaHard3D = 2.7f;            //! Threshold for Hard Thresholding
	const float    tauMatch = (chnls == 1 ? 3.f : 1.f) * (sigma < 35.0f ? 2500 : 5000); //! threshold used to determinate similarity between patches


	const unsigned kHard_2 = kHard * kHard;
	const unsigned h_b = (unsigned)H + 2 * nHard;
	const unsigned w_b = (unsigned)W + 2 * nHard;
	vector<float> img_sym_noisy, img_sym_basic, img_sym_denoised;
	symetrize(img, img_sym_noisy, W, H, 1, nHard);

	vector<vector<unsigned> > patch_table;
	if (img_basic.size() != img_sym_noisy.size())
		img_basic.resize(img_sym_noisy.size());
	vector<float> denominator(w_b * h_b * chnls, 0.0f);
	vector<float> numerator(w_b * h_b * chnls, 0.0f);

	precompute_BM(patch_table, img_sym_noisy, w_b, h_b, kWien, NHard, nWien, pWien, tauMatch);

	//! table_2D[p * N + q + (i * width + j) * kHard_2 + c * (2 * nHard + 1) * width * kHard_2]
	vector<float> table_2D((2 * nHard + 1) * w_b * chnls * kHard_2, 0.0f);

	//! Initialization for convenience
	vector<unsigned> row_ind;
	ind_initialize(row_ind, h_b - kHard + 1, nHard, pHard);
	vector<unsigned> column_ind;
	ind_initialize(column_ind, w_b - kHard + 1, nHard, pHard);

	//! Allocate plan for FFTW library
	fftwf_plan* plan_2d_for_1 = new fftwf_plan[nb_threads];
	fftwf_plan* plan_2d_for_2 = new fftwf_plan[nb_threads];
	fftwf_plan* plan_2d_inv = new fftwf_plan[nb_threads];
	const unsigned nb_cols = ind_size(w_b - kHard + 1, nHard, pHard);
	allocate_plan_2d(&plan_2d_for_1[0], kHard, FFTW_REDFT10,
		w_b * (2 * nHard + 1) * chnls);
	allocate_plan_2d(&plan_2d_for_2[0], kHard, FFTW_REDFT10,
		w_b * pHard * chnls);
	allocate_plan_2d(&plan_2d_inv[0], kHard, FFTW_REDFT01,
		NHard * nb_cols * chnls);


	vector<float> kaiser_window(kHard_2);
	vector<float> coef_norm(kHard_2);
	vector<float> coef_norm_inv(kHard_2);
	preProcess(kaiser_window, coef_norm, coef_norm_inv, kHard);


	vector<float> group_3D_table(chnls * kHard_2 * NHard * column_ind.size());
	vector<float> wx_r_table;
	wx_r_table.reserve(chnls * column_ind.size());
	vector<float> hadamard_tmp(NHard);

	//! Loop on i_r
	for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
	//for (unsigned ind_i = 0; ind_i < 1; ind_i++)
	{
		const unsigned i_r = row_ind[ind_i];

		//! Update of table_2D

		dct_2d_process(table_2D, img_sym_noisy, plan_2d_for_1, plan_2d_for_2, nHard,
			w_b, h_b, chnls, kHard, i_r, pHard, coef_norm,
			row_ind[0], row_ind.back());

		wx_r_table.clear();
		group_3D_table.clear();


		//! Loop on j_r
		for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
		//for (unsigned ind_j = 0; ind_j < 1; ind_j++)
		{
			//! Initialization
			const unsigned j_r = column_ind[ind_j];
			const unsigned k_r = i_r * w_b + j_r;
			
			//! Number of similar patches
			const unsigned nSx_r = patch_table[k_r].size();
			//cout<<"nSx_r " << nSx_r<<" ";
			//! Build of the 3D group
			vector<float> group_3D(chnls * nSx_r * kHard_2, 0.0f);
			for (unsigned c = 0; c < chnls; c++)
				for (unsigned n = 0; n < nSx_r; n++)
	/*			for (unsigned n = 0; n < 1; n++)*/
				{
					const unsigned ind = patch_table[k_r][n] + (nHard - i_r) * w_b;
					for (unsigned k = 0; k < kHard_2; k++){
						group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] = //注意此处group_3D排列顺序，是按块排列，group_3D前nSx_r行是所有8*8块左上角第一个像素的值
						table_2D[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * w_b];
						//cout<< "("<< k_r << ")" <<group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] <<"  ";
					}
				}

			//////! HT filtering of the 3D group
			vector<float> weight_table(chnls);
			ht_filtering_hadamard(group_3D, hadamard_tmp, nSx_r, kHard, chnls, sigma_table,
				lambdaHard3D, weight_table, !useSD);

			//for (unsigned n = 0; n < nSx_r; n++)
			//{
			//	for (unsigned k = 0; k < kHard_2; k++) {
			//		cout << group_3D[n + k * nSx_r ] <<" ";
			//	}
			//	cout << endl;
			//}

			//! 3D weighting using Standard Deviation
			/*if (useSD)
				sd_weighting(group_3D, nSx_r, kHard, chnls, weight_table);*/

				//! Save the 3D group. The DCT 2D inverse will be done after.
			for (unsigned c = 0; c < chnls; c++)
				for (unsigned n = 0; n < nSx_r; n++)
					for (unsigned k = 0; k < kHard_2; k++)
						group_3D_table.push_back(group_3D[n + k * nSx_r +
							c * kHard_2 * nSx_r]);




			//! Save weighting
			for (unsigned c = 0; c < chnls; c++)
				wx_r_table.push_back(weight_table[c]);
			//cout << "j_r1 " << j_r << " ;";

		} //! End of loop on j_r
		//!  Apply 2D inverse transform
			dct_2d_inverse(group_3D_table, kHard, NHard* chnls* column_ind.size(),
				coef_norm_inv, plan_2d_inv);
			//int i = 0;
			//for (vector<float>::iterator it = group_3D_table.begin(); it != group_3D_table.end(); ++it,i++) {
			//	cout << (*it)<<" ";
			//	if (i == 63) {
			//		i = 0;
			//		cout << " fenhang "<<endl;
			//	}
			//}
			 //! Registration of the weighted estimation
			//cout << "done1" << endl;
			unsigned dec = 0;
			for (unsigned ind_j = 0; ind_j <column_ind.size(); ind_j++)
			{
				const unsigned j_r = column_ind[ind_j];
				const unsigned k_r = i_r * w_b + j_r;
				//cout <<"k_r"<< k_r;
				const unsigned nSx_r = patch_table[k_r].size();
				for (unsigned c = 0; c < chnls; c++)
				{
					for (unsigned n = 0; n < nSx_r; n++)
					{
						const unsigned k = patch_table[k_r][n] + c * w_b * h_b;
						for (unsigned p = 0; p < kHard; p++)
							for (unsigned q = 0; q < kHard; q++)
							{
								const unsigned ind = k + p * w_b + q;
								numerator[ind] += kaiser_window[p * kHard + q]
									* wx_r_table[c + ind_j * chnls]
									* group_3D_table[p * kHard + q + n * kHard_2
									+ c * kHard_2 * nSx_r + dec];
								denominator[ind] += kaiser_window[p * kHard + q]
									* wx_r_table[c + ind_j * chnls];
							}
					}
				}
				dec += nSx_r * chnls * kHard_2;
				//cout << "j_r2 " << j_r << " ;";
			}
			//cout << "i_r " << i_r << " ;";
	} //! End of loop on i_r
	//cout << "size: " << img_basic.size()<<endl;
	//! Final reconstruction
	for (unsigned k = 0; k < w_b * h_b; k++)
		img_basic[k] = numerator[k] / denominator[k];


	if (img_out.size() != img.size())
		img_out.resize(img.size());


	

		const unsigned dc_b =  nHard * w_b + nHard;
		for (unsigned i = 0; i < h; i++)
			for (unsigned j = 0; j < w; j++)
				img_out[i*w+j] = img_basic[dc_b + i * w_b + j];

	
	if (compute_psnr(img, img_out, &psnr_basic, &rmse_basic) != EXIT_SUCCESS)
		return EXIT_FAILURE;
	cout << "(basic image) :" << endl;
	cout << "PSNR: " << psnr_basic << endl;
	cout << "RMSE: " << rmse_basic << endl << endl;

	if (save_image(argv[2], img_out,w, h, chnls) != EXIT_SUCCESS)
		return EXIT_FAILURE;
}
