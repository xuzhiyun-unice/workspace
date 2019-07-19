/*
BM
*/

// #define __GECOS_TYPE_EXPL__

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "utils.h"
#include <fftw3.h>

#ifndef __GECOS_TYPE_EXPL__
#include "io_png.c"
#endif


#define K 8
#define N 33

//K2 and N2 are floor(K/2) and floor(N/2)
#define K2 4
#define N2 16
#define KK 64
// CROP = N2 - K2
#define CROP 4
#define M_PI       3.14159265358979323846   // pi
#define H 384
#define W 512
//HW=H*W
#define HW (H*W)
//size_h_pad = H+2*N2; 
#define SIZE_H_PAD  (H+2*N2)
//size_w_pad = W+2*N2; 522
#define SIZE_W_PAD  (W+2*N2)
//size_h_ref = H + N2 - K2; 
#define SIZE_H_REF (H + N2 - K2)
//size_w_ref = H + N2 - K2; 
#define SIZE_W_REF (W + N2 - K2)

// Parameter
#define DTYPE float
#define DTYPE_DOUBLE double
#define SIZE_T int

#define SQRT2     1.414213562373095
#define SQRT2_INV 0.7071067811865475
#define YUV       0
#define YCBCR     1
#define OPP       2
#define RGB       3
#define DCT       4
#define BIOR      5
#define HADAMARD  6




#ifndef max
#define max(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
#define min(a,b) ((a) < (b) ? (a) : (b))
#endif

/**
int main(int argc, char** argv) {

	if (argc < 3) {
		printf("usage: nlm image denoised \n");
		exit(-1);
	}

	treat_one_image(argv[1], argv[2]);
}
 **/


//BM3D
/** ----------------- **/
/** - Main function - **/
/** ----------------- **/
/**
 * @brief run BM3D process. Depending on if OpenMP is used or not,
 *        and on the number of available threads, it divides the noisy
 *        image in sub_images, to process them in parallel.
 *
 * @param sigma: value of assumed noise of the noisy image;
 * @param img_noisy: noisy image;
 * @param img_basic: will be the basic estimation after the 1st step
 * @param img_denoised: will be the denoised final image;
 * @param width, height, chnls: size of the image;
 * @param useSD_h (resp. useSD_w): if true, use weight based
 *        on the standard variation of the 3D group for the
 *        first (resp. second) step, otherwise use the number
 *        of non-zero coefficients after Hard Thresholding
 *        (resp. the norm of Wiener coefficients);
 * @param tau_2D_hard (resp. tau_2D_wien): 2D transform to apply
 *        on every 3D group for the first (resp. second) part.
 *        Allowed values are DCT and BIOR;
 * @param color_space: Transformation from RGB to YUV. Allowed
 *        values are RGB (do nothing), YUV, YCBCR and OPP.
 *
 * @return EXIT_FAILURE if color_space has not expected
 *         type, otherwise return EXIT_SUCCESS.
 **/
int run_bm3d(
	const float sigma
	, float  img_noisy[H][W]
	, float  img_basic[H][W]
	, float  img_denoised[H][W]
	, const unsigned width
	, const unsigned height
	, const unsigned chnls
	, const bool useSD_h
	, const bool useSD_w
	, const unsigned tau_2D_hard
	, const unsigned tau_2D_wien
	, const unsigned color_space
) {
	//! Parameters
	const unsigned nHard = 16; //! Half size of the search window
	const unsigned nWien = 16; //! Half size of the search window
	const unsigned kHard = (tau_2D_hard == BIOR || sigma < 40.f ? 8 : 12); //! Must be a power of 2 if tau_2D_hard == BIOR
	const unsigned kWien = (tau_2D_wien == BIOR || sigma < 40.f ? 8 : 12); //! Must be a power of 2 if tau_2D_wien == BIOR
	const unsigned NHard = 16; //! Must be a power of 2
	const unsigned NWien = 32; //! Must be a power of 2
	const unsigned pHard = 3;  //step
	const unsigned pWien = 3;
	float mat_in_pad[SIZE_H_PAD][SIZE_W_PAD];
	float img_sym_basic[SIZE_H_PAD][SIZE_W_PAD];
	//! Check memory allocation (check size)




	//! Allocate plan for FFTW library  可能会有问题 
	fftwf_plan* plan_2d_for_1;
	fftwf_plan* plan_2d_for_2;
	fftwf_plan* plan_2d_inv;


	//! Add boundaries and symetrize them //continue...
	const unsigned h_b = height + 2 * nHard; //SIZE_H_PAD
	const unsigned w_b = width + 2 * nHard; //SIZE_W_PAD
	pad_matrix_replicate(img_noisy, mat_in_pad);

	//! Allocating Plan for FFTW process
	if (tau_2D_hard == DCT)
	{
		const unsigned nb_cols = ind_size(w_b - kHard + 1, nHard, pHard);
		allocate_plan_2d(plan_2d_for_1, kHard, FFTW_REDFT10,
			w_b * (2 * nHard + 1) * chnls);
		allocate_plan_2d(&plan_2d_for_2, kHard, FFTW_REDFT10,
			w_b * pHard * chnls);
		allocate_plan_2d(&plan_2d_inv, kHard, FFTW_REDFT01,
			NHard * nb_cols * chnls);
	}

	//! Denoising, 1st Step
	printf("step 1...");
	bm3d_1st_step(sigma, mat_in_pad, img_sym_basic, w_b, h_b, chnls, nHard,
		kHard, NHard, pHard, useSD_h, color_space, tau_2D_hard,
		plan_2d_for_1, &plan_2d_for_2, &plan_2d_inv);
	printf("done.");

	//! To avoid boundaries problem


	//! Allocating Plan for FFTW process
	if (tau_2D_wien == DCT)
	{
		const unsigned nb_cols = ind_size(w_b - kWien + 1, nWien, pWien);
		allocate_plan_2d(&plan_2d_for_1[0], kWien, FFTW_REDFT10,
			w_b * (2 * nWien + 1) * chnls);
		allocate_plan_2d(&plan_2d_for_2[0], kWien, FFTW_REDFT10,
			w_b * pWien * chnls);
		allocate_plan_2d(&plan_2d_inv[0], kWien, FFTW_REDFT01,
			NWien * nb_cols * chnls);
	}

	//! Denoising, 2nd Step
	printf("step 2... \n");
	bm3d_2nd_step(sigma, img_sym_noisy, img_sym_basic, img_sym_denoised,
		w_b, h_b, chnls, nWien, kWien, NWien, pWien, useSD_w, color_space,
		tau_2D_wien, &plan_2d_for_1[0], &plan_2d_for_2[0], &plan_2d_inv[0]);
	printf("done \n" );

	//! Obtention of img_denoised
	for (unsigned c = 0; c < chnls; c++)
	{
		const unsigned dc_b = c * w_b * h_b + nWien * w_b + nWien;
		unsigned dc = c * width * height;
		for (unsigned i = 0; i < height; i++)
			for (unsigned j = 0; j < width; j++, dc++)
				img_denoised[dc] = img_sym_denoised[dc_b + i * w_b + j];
	}

	if (tau_2D_hard == DCT || tau_2D_wien == DCT)
		for (unsigned n = 0; n < nb_threads; n++)
		{
			fftwf_destroy_plan(plan_2d_for_1[n]);
			fftwf_destroy_plan(plan_2d_for_2[n]);
			fftwf_destroy_plan(plan_2d_inv[n]);
		}
	fftwf_cleanup();

	return EXIT_SUCCESS;
}


/**
 * @brief Run the basic process of BM3D (1st step). The result
 *        is contained in img_basic. The image has boundary, which
 *        are here only for block-matching and doesn't need to be
 *        denoised.
 *
 * @param sigma: value of assumed noise of the image to denoise;
 * @param img_noisy: noisy image;
 * @param img_basic: will contain the denoised image after the 1st step;
 * @param width, height, chnls : size of img_noisy;
 * @param nHard: size of the boundary around img_noisy;
 * @param useSD: if true, use weight based on the standard variation
 *        of the 3D group for the first step, otherwise use the number
 *        of non-zero coefficients after Hard-thresholding;
 * @param tau_2D: DCT or BIOR;
 * @param plan_2d_for_1, plan_2d_for_2, plan_2d_inv : for convenience. Used
 *        by fftw.
 *
 * @return none.
 **/

void bm3d_1st_step(
	const float sigma
	, float mat_in_pad[H][W]
	,float img_basic[SIZE_H_PAD][SIZE_W_PAD]
	, const unsigned width
	, const unsigned height
	, const unsigned chnls
	, const unsigned nHard
	, const unsigned kHard
	, const unsigned NHard
	, const unsigned pHard
	, const bool     useSD
	, const unsigned color_space
	, const unsigned tau_2D
	, fftwf_plan* plan_2d_for_1
	, fftwf_plan* plan_2d_for_2
	, fftwf_plan* plan_2d_inv
) {
	

	////! Estimatation of sigma on each channel
	float sigma_table[3] = {sigma,sigma,sigma};
	//if (estimate_sigma(sigma, sigma_table, chnls, color_space) != EXIT_SUCCESS)
	//	return;

	//! Parameters initialization
	const float    lambdaHard3D = 2.7f;            //! Threshold for Hard Thresholding
	const float    tauMatch = (chnls == 1 ? 3.f : 1.f) * (sigma < 35.0f ? 2500 : 5000); //! threshold used to determinate similarity between patches

	//! Initialization for convenience
	/*vector<unsigned> row_ind;
	ind_initialize(row_ind, height - kHard + 1, nHard, pHard);
	vector<unsigned> column_ind;
	ind_initialize(column_ind, width - kHard + 1, nHard, pHard);
	const unsigned kHard_2 = kHard * kHard;*/

	//vector<float> group_3D_table(chnls * kHard_2 * NHard * column_ind.size());
	float group_3D_table[KK][KK];

	//vector<float> wx_r_table;
	//wx_r_table.reserve(chnls * column_ind.size());

	//vector<float> hadamard_tmp(NHard);
	float hadamard_tmp[K];

	//! Check allocation memory
	/*if (img_basic.size() != img_noisy.size())
		img_basic.resize(img_noisy.size());*/

	//! Preprocessing (KaiserWindow, Threshold, DCT normalization, ...)
	/*vector<float> kaiser_window(kHard_2);
	vector<float> coef_norm(kHard_2);
	vector<float> coef_norm_inv(kHard_2);*/
	
	//! Kaiser Window coefficients
	float kaiserWindow[K][K] = 
	 { { 0.1924f, 0.2989f, 0.3846f, 0.4325f, 0.4325f, 0.3846f, 0.2989f, 0.1924f},
	   { 0.2989f, 0.4642f, 0.5974f, 0.6717f, 0.6717f, 0.5974f, 0.4642f, 0.2989f },
	   { 0.3846f, 0.5974f, 0.7688f, 0.8644f,0.8644f, 0.7688f, 0.5974f, 0.3846f },
	   { 0.4325f, 0.6717f, 0.8644f, 0.9718f, 0.9718f, 0.8644f,0.6717f, 0.4325f },
	   { 0.4325f, 0.6717f, 0.8644f, 0.9718f, 0.9718f, 0.8644f, 0.6717f, 0.4325f },
	   { 0.3846f, 0.5974f, 0.7688f, 0.8644f, 0.8644f, 0.7688f, 0.5974f, 0.3846f },
	   { 0.2989f, 0.4642f,0.5974f, 0.6717f, 0.6717f, 0.5974f, 0.4642f, 0.2989f },
	{0.1924f, 0.2989f, 0.3846f, 0.4325f, 0.4325f, 0.3846f, 0.2989f, 0.1924f } };
	

	//! Coefficient of normalization for DCT II and DCT II inverse

	float coef_norm[K][K];
	float coef_norm_inv[K][K];
	const float coef = 0.5f / ((float)(K));
	for (unsigned i = 0; i < K; i++)
		for (unsigned j = 0; j < K; j++)
		{
			if (i == 0 && j == 0)
			{
				coef_norm[i][j] = 0.5f * coef;
				coef_norm_inv[i][j] = 2.0f;
			}
			else if (i * j == 0)
			{
				coef_norm[i][j] = SQRT2_INV * coef;
				coef_norm_inv[i][j] = SQRT2;
			}
			else
			{
				coef_norm[i][j] = 1.0f * coef;
				coef_norm_inv[i][j] = 1.0f;
			}
		}
	

	//! Preprocessing of Bior table
	float lpd[10], hpd[10], lpr[10], hpr[10];
	bior15_coef(lpd, hpd, lpr, hpr);

	

	//! For aggregation part
	//vector float denominator(width * height * chnls, 0.0f);
	//vector<float> numerator(width * height * chnls, 0.0f);
	float denominator[H][W];
	float numerator[H][W];

	//! Precompute Bloc-Matching
	//vector<vector<unsigned> > patch_table;
	float patch_table[10][H][W];
	precompute_BM(patch_table, img_noisy, width, height, kHard, NHard, nHard, pHard, tauMatch);

	//! table_2D[p * N + q + (i * width + j) * kHard_2 + c * (2 * nHard + 1) * width * kHard_2]
	vector<float> table_2D((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);

	//! Loop on i_r
	for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
	{
		const unsigned i_r = row_ind[ind_i];

		//! Update of table_2D
		if (tau_2D == DCT)
			dct_2d_process(table_2D, img_noisy, plan_2d_for_1, plan_2d_for_2, nHard,
				width, height, chnls, kHard, i_r, pHard, coef_norm,
				row_ind[0], row_ind.back());
		else if (tau_2D == BIOR)
			bior_2d_process(table_2D, img_noisy, nHard, width, height, chnls,
				kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);

		wx_r_table.clear();
		group_3D_table.clear();

		//! Loop on j_r
		for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
		{
			//! Initialization
			const unsigned j_r = column_ind[ind_j];
			const unsigned k_r = i_r * width + j_r;

			//! Number of similar patches
			const unsigned nSx_r = patch_table[k_r].size();

			//! Build of the 3D group
			vector<float> group_3D(chnls * nSx_r * kHard_2, 0.0f);
			for (unsigned c = 0; c < chnls; c++)
				for (unsigned n = 0; n < nSx_r; n++)
				{
					const unsigned ind = patch_table[k_r][n] + (nHard - i_r) * width;
					for (unsigned k = 0; k < kHard_2; k++)
						group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
						table_2D[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
				}

			//! HT filtering of the 3D group
			vector<float> weight_table(chnls);
			ht_filtering_hadamard(group_3D, hadamard_tmp, nSx_r, kHard, chnls, sigma,
				lambdaHard3D, weight_table, !useSD);

			//! 3D weighting using Standard Deviation
			if (useSD)
				sd_weighting(group_3D, nSx_r, kHard, chnls, weight_table);

			//! Save the 3D group. The DCT 2D inverse will be done after.
			for (unsigned c = 0; c < chnls; c++)
				for (unsigned n = 0; n < nSx_r; n++)
					for (unsigned k = 0; k < kHard_2; k++)
						group_3D_table.push_back(group_3D[n + k * nSx_r +
							c * kHard_2 * nSx_r]);

			//! Save weighting
			for (unsigned c = 0; c < chnls; c++)
				wx_r_table.push_back(weight_table[c]);

		} //! End of loop on j_r

		//!  Apply 2D inverse transform
		if (tau_2D == DCT)
			dct_2d_inverse(group_3D_table, kHard, NHard * chnls * column_ind.size(),
				coef_norm_inv, plan_2d_inv);
		else if (tau_2D == BIOR)
			bior_2d_inverse(group_3D_table, kHard, lpr, hpr);

		//! Registration of the weighted estimation
		unsigned dec = 0;
		for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
		{
			const unsigned j_r = column_ind[ind_j];
			const unsigned k_r = i_r * width + j_r;
			const unsigned nSx_r = patch_table[k_r].size();
			for (unsigned c = 0; c < chnls; c++)
			{
				for (unsigned n = 0; n < nSx_r; n++)
				{
					const unsigned k = patch_table[k_r][n] + c * width * height;
					for (unsigned p = 0; p < kHard; p++)
						for (unsigned q = 0; q < kHard; q++)
						{
							const unsigned ind = k + p * width + q;
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
		}

	} //! End of loop on i_r

	//! Final reconstruction
	for (unsigned k = 0; k < width * height * chnls; k++)
		img_basic[k] = numerator[k] / denominator[k];
}


// 变换函数 

/*Add boundaries by replicate*/
void pad_matrix_replicate(float orig[H][W], float padded[SIZE_H_PAD][SIZE_W_PAD]) {
	int i, j;
	for (i = -N2; i < H + N2; i++)
	{
		for (j = -N2; j < W + N2; j++)
		{
			padded[(i + N2)][j + N2] = orig[min(max(i, 0), H - 1)][min(max(j, 0), W - 1)];
		}
	}
}

/**
 * @brief Add boundaries by symetry
 *
 * @param orig :original image to symetrize
 * @param padded : will contain img with symetrized boundaries
 *
 * @return none.
 **/
void pad_matrix_symetry(float orig[H][W], float padded[SIZE_H_PAD][SIZE_W_PAD]) {
	// just for chls=1;
	int w = SIZE_W_PAD;
	int h = SIZE_H_PAD;

	//! Center of the image(original img)
	for (unsigned i = 0; i < H; i++)
		for (unsigned j = 0; j < W; j++)
			padded[N2 + i][N2 + j] = orig[i][j];

	//! Top and bottom
	for (unsigned j = 0; j < w; j++)
		for (unsigned i = 0; i < N2; i++)
		{
			padded[i][j] = padded[2 * N2 - i - 1][j];
			padded[h - i - 1][j] = padded[h - 2 * N2 + i][j];
		}
	//! Right and left

	for (unsigned i = 0; i < h; i++)
	{
		for (unsigned j = 0; j < N2; j++)
		{
			padded[i][j] = padded[i][2 * N2 - j - 1];
			padded[i][w - j - 1] = padded[i][w - 2 * N2 + j];
		}
	}
}

//小波系数
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
	float lp1[10]
	, float hp1[10]
	, float lp2[10]
	, float hp2[10]
) {
	const float coef_norm = 1.f / (sqrtf(2.f) * 128.f);
	const float sqrt2_inv = 1.f / sqrtf(2.f);

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
 * @brief Initialize a set of indices.
 *
 * @param ind_set: will contain the set of indices;
 * @param max_size: indices can't go over this size;
 * @param N : boundary;
 * @param step: step between two indices.
 *
 * @return none.
 **/
void ind_initialize(
	unsigned ind_set[H][W]
	, const unsigned max_size
	, const unsigned M
	, const unsigned step
) {


}

/**
 * @brief For convenience. Estimate the size of the ind_set vector built
 *        with the function ind_initialize().
 *
 * @return size of ind_set vector built in ind_initialize().
 **/
unsigned ind_size(
	const unsigned max_size
	, const unsigned M
	, const unsigned step
) {
	unsigned ind = M;
	unsigned k = 0;
	while (ind < max_size - M)
	{
		k++;
		ind += step;
	}
	if (ind - step < max_size - M - 1)
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
	, const unsigned M
	, const fftwf_r2r_kind kind
	, const unsigned nb
) {
	int            nb_table[2] = { M, M };
	int            nembed[2] = { M, M };
	fftwf_r2r_kind kind_table[2] = { kind, kind };

	typedef float type_vec2d;
	type_vec2d* vec = (type_vec2d*)fftwf_malloc(M * M * nb * sizeof(type_vec2d));//malloc hanshu 1.7.2
	(*plan) = fftwf_plan_many_r2r(2, nb_table, nb, vec, nembed, 1, M * M, vec,
		nembed, 1, M * M, kind_table, FFTW_ESTIMATE);

	fftwf_free(vec);
}

/**
 * @brief Initialize a set of indices.
 *
 * @param ind_set: will contain the set of indices;
 * @param max_size: indices can't go over this size;
 * @param boundary : boundary;
 * @param step: step between two indices.
 *	int row_ind[127] = { 0 };
	int column_ind[169] = { 0 };
 * @return none.
 **/
void ind_initialize(
	int* ind_set
	, const int max_size
	, const int boundary
	, const int step
) {
	int ind = boundary;
	int count = 0;
	while (ind < max_size - boundary)
	{
		ind_set[count] = ind;
		ind += step;
		count++;
	}
	if (ind_set[count - 1] < max_size - boundary - 1) {
		ind_set[count] = (max_size - boundary - 1);
	}
}

/**
 * @brief Initialize a 1D fftwf_plan with some parameters
 *
 * @param plan: fftwf_plan to allocate;
 * @param M: size of the vector to apply the 1D transform;
 * @param kind: forward or backward;
 * @param nb: number of 1D transform which will be processed.
 *
 * @return none.
 **/
void allocate_plan_1d(
	fftwf_plan* plan
	, const unsigned M
	, const fftwf_r2r_kind kind
	, const unsigned nb
) {
	int nb_table[1] = { M };
	int nembed[1] = { M * nb };
	fftwf_r2r_kind kind_table[1] = { kind };

	typedef float type_ve1d;
	type_ve1d* vec = (type_ve1d*)fftwf_malloc(M * nb * sizeof(type_ve1d));//malloc hanshu 1.7.2
	(*plan) = fftwf_plan_many_r2r(1, nb_table, nb, vec, nembed, 1, M, vec,
		nembed, 1, M, kind_table, FFTW_ESTIMATE);
	fftwf_free(vec);
}

/**
 * @brief tabulated values of log2(M), where M = 2 ^ n.
 *
 * @param M : must be a power of 2 smaller than 64
 *
 * @return n = log2(M)
 **/
unsigned ind_log2(
	const unsigned M
) {
	return (M == 1 ? 0 :
		(M == 2 ? 1 :
		(M == 4 ? 2 :
			(M == 8 ? 3 :
			(M == 16 ? 4 :
				(M == 32 ? 5 : 6))))));
}

/**
 * @brief tabulated values of log2(M), where M = 2 ^ n.
 *
 * @param M : must be a power of 2 smaller than 64
 *
 * @return n = 2 ^ M
 **/
unsigned ind_pow2(
	const unsigned M
) {
	return (M == 0 ? 1 :
		(M == 1 ? 2 :
		(M == 2 ? 4 :
			(M == 3 ? 8 :
			(M == 4 ? 16 :
				(M == 5 ? 32 : 64))))));
}

/**
 * @brief matrix_multiplication, only for matrix with dim = k*k 
 *
 * @param A : dim must be K*K
 * @param B : dim must be K*K
 * @param R : result R = A*B
 * 
 **/
void matrix_multiplication(float A[K][K], float B[K][K], float R[K][K])
{
	int i, j, n;
	float tmp;
	for (i = 0; i < K; ++i)
		for (n = 0; n < K; ++n) {
			tmp = A[i][n];
			for (j = 0; j < K; ++j)
				R[i][j] += tmp * B[n][j];
		}
}

/**
 * @brief 2d DCT transform, only for matrix with dim = k*k
 *
 * @param A : dim must be K*K
   @param flag : flag==0 dect_2d; flag==1 dec_2d_inv
 *
 **/
void dct_2d(float A[K][K],int flag) 
{ 
	// If A is an n x n matrix, then the following are true:
	//   T*A    == dct(A),  T'*A   == idct(A)
	//   T*A*T' == dct2(A), T'*A*T == idct2(A)
	float T[K][K] =
	{ { 0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f},
	{ 0.490393f, 0.415735f, 0.277785f, 0.097545f, -0.097545f, -0.277785f, -0.415735f, -0.490393f },
	{ 0.461940f, 0.191342f, -0.191342f, -0.461940f, -0.461940f, -0.191342f, 0.191342f, 0.461940f },
	{ 0.415735f, -0.097545f, -0.490393f, -0.277785f, 0.277785f, 0.490393f, 0.097545f, -0.415735f },
	{ 0.353553f, -0.353553f, -0.353553f, 0.353553f, 0.353553f, -0.353553f, -0.353553f, 0.353553f },
	{ 0.277785f, -0.490393f, 0.097545f, 0.415735f, -0.415735f, -0.097545f, 0.490393f, -0.277785f },
	{ 0.191342f, -0.461940f, 0.461940f, -0.191342f, -0.191342f, 0.461940f, -0.461940f, 0.191342f },
	{ 0.097545f, -0.277785f, 0.415735f, -0.490393f, 0.490393f, -0.415735f, 0.277785f, -0.097545f } };

	float Tinv[K][K] =
	{ { 0.353553f, 0.490393f, 0.461940f, 0.415735f, 0.353553f, 0.277785f, 0.191342f, 0.097545f},
	{ 0.353553f, 0.415735f, 0.191342f,-0.097545f,-0.353553f,-0.490393f,-0.461940f, -0.277785f },
	{ 0.353553f, 0.277785f,-0.191342f,-0.490393f,-0.353553f, 0.097545f, 0.461940f,  0.415735f },
	{ 0.353553f, 0.097545f,-0.461940f,-0.277785f, 0.353553f, 0.415735f,-0.191342f, -0.490393f },
	{ 0.353553f,-0.097545f,-0.461940f, 0.277785f, 0.353553f,-0.415735f,-0.191342f,  0.490393f },
	{ 0.353553f,-0.277785f,-0.191342f, 0.490393f,-0.353553f,-0.097545f, 0.461940f, -0.415735f },
	{ 0.353553f,-0.415735f, 0.191342f, 0.097545f,-0.353553f, 0.490393f,-0.461940f,  0.277785f },
	{ 0.353553f,-0.490393f, 0.461940f,-0.415735f, 0.353553f,-0.277785f, 0.191342f, -0.097545f } };

	float res1[K][K] = { 0 };
	float res2[K][K] = { 0 };
	
	if (flag==0)
	{
		matrix_multiplication(T, A, res1);
		matrix_multiplication(res1, Tinv, res2); //T*A*T' == dct2(A)
	}
	if (flag==1)
	{
		matrix_multiplication(Tinv, A, res1);
		matrix_multiplication(res1,T,res2); //T'*A*T == idct2(A)
	}
	int i, j;
	for (i = 0; i < K; ++i)
		for (j = 0; j < K; ++j) {
			A[i][j] = res2[i][j];
		}
}void pad_matrix_symetry(int orig[H][W], int padded[SIZE_H_PAD][SIZE_W_PAD]) {
	int chnls = 1;
	int w = SIZE_W_PAD;
	int h = SIZE_H_PAD;

		//! Center of the image
		for (unsigned i = 0; i < H; i++)
			for (unsigned j = 0; j < W; j++)
				padded[N2 + i ][N2+j] = orig[i][j];

		//! Top and bottom
		for (unsigned j = 0; j < w; j++)
			for (unsigned i = 0; i < N2; i++)
			{
				padded[i][j] = padded[2*N2-i-1][j];
				padded[h-i-1][j] = padded[h-2*N2+i][j];
			}

		//! Right and left

		for (unsigned i = 0; i < h; i++)
		{
			for (unsigned j = 0; j < N2; j++)
			{
				padded[i][j] = padded[i][2 * N2 - j - 1];
				padded[i][w - j - 1] = padded[i][w - 2 * N2 + j];
			}

		}
}
/**
 * @brief Precompute Bloc Matching (distance inter-patches)
 *
 * @param patch_table: for each patch in the image, will contain
 * all coordinnate of its similar patches
 * @param img: noisy image on which the distance is computed
 * @param width, height: size of img
 * @param kHW: size of patch  kHard
 * @param NHW: maximum similar patches wanted  NHard
 * @param nHW: size of the boundary of img
 * @param tauMatch: threshold used to determinate similarity between
 *        patches
 *patch_table, img_noisy, width, height, kHard, NHard, nHard, pHard, tauMatch
 * @return none.
 **/
void precompute_BM(
	vector<vector<unsigned> >& patch_table // H*W 的向量 保存相似快坐标
	, const vector<float>& img  //噪音图像
	, const unsigned width
	, const unsigned height
	, const unsigned kHW  //patch直径
	, const unsigned NHW // patch最大数量
	, const unsigned nHW //搜索窗半径
	, const unsigned pHW  //步长
	, const float    tauMatch
) {
	//! Declarations
	const unsigned Ns = 2 * nHW + 1; //搜索窗长度
	const float threshold = tauMatch * kHW * kHW; // 2500*8*8
	vector<float> diff_table(width * height); //距离值
	vector<vector<float> > sum_table((nHW + 1) * Ns, vector<float>(width * height, 2 * threshold));
	if (patch_table.size() != width * height)
		patch_table.resize(width * height);
	vector<unsigned> row_ind;
	ind_initialize(row_ind, height - kHW + 1, nHW, pHW);
	vector<unsigned> column_ind;
	ind_initialize(column_ind, width - kHW + 1, nHW, pHW);

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
				}

			}
		}

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
				for (int di = 0; di <= (int)nHW; di++)
					if (sum_table[dj + nHW + di * Ns][k_r] < threshold)
						table_distance.push_back(make_pair(
							sum_table[dj + nHW + di * Ns][k_r]
							, k_r + di * width + dj));

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
			for (unsigned n = 0; n < nSx_r; n++)
				patch_table[k_r].push_back(table_distance[n].second);

			//! To avoid problem
			if (nSx_r == 1)
				patch_table[k_r].push_back(table_distance[0].second);
		}
	}
}
double psnr(double* start1_psnr, double* start2_psnr, int height_psnr, int width_psnr)
{
	int psnr_i, psnr_j;
	double sum_psnr = 0.0, output_psnr;
	for (psnr_i = 0; psnr_i < height_psnr; psnr_i++)
		for (psnr_j = 0; psnr_j < width_psnr; psnr_j++)
			sum_psnr += ((*(start1_psnr + psnr_i * width_psnr + psnr_j)) - (*(start2_psnr + psnr_i * width_psnr + psnr_j))) * ((*(start1_psnr + psnr_i * width_psnr + psnr_j)) - (*(start2_psnr + psnr_i * width_psnr + psnr_j)));
	output_psnr = 10 * (2 * log10(255.0) + log10((double)height_psnr * width_psnr) - log10(sum_psnr));
	return output_psnr;
}
