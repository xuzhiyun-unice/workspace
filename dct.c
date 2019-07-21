#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <math.h>
#include<time.h>
#include "io_png.h"
#define DTYPE int
#define DCT       4
#define BIOR      5
#define H 10
#define W 10
#define K 5
#define N 13
#define K2 2
#define N2 6
#define row_index_size 5
#define column_index_size 5
#define kHard 8   //kHard = (tau_2D_hard == BIOR || sigma < 40.f ? 8 : 12); //! Must be a power of 2 if tau_2D_hard == BIOR
#define kHard_2 (kHard*kHard)
#define kWien 8  // kWien = (tau_2D_wien == BIOR || sigma < 40.f ? 8 : 12); //! Must be a power of 2 if tau_2D_wien == BIOR
#define kWien_2 (kWien*kWien)
#define NHard 16 //MAX NUMBLE BLOCK step1 , Must be a power of 2
#define NWien 32 //MAX NUMBLE BLOCK step2 , Must be a power of 2
#define KK (K*K)

#define CROP (N2-K2)
#define PATCH_SIZE 24
#define SIZE_H_PAD  (H+2*N2)
#define SIZE_W_PAD  (W+2*N2)
//size_h_ref = H + N2 - K2
#define SIZE_H_REF (H + N2 - K2)  
//size_w_ref = W + N2 - K2
#define SIZE_W_REF (W + N2 - K2) 
#define Height 384
#define Width 512
#ifndef max
#define max(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
#define min(a,b) ((a) < (b) ? (a) : (b))
#endif
const int dx_patch[24] = { -4,-2,0,2,4,-4,-2,0,2,4,-4,-2,2,4,-4,-2,0,2,4,-4,-2,0,2,4 };
const int dy_patch[24] = { -4,-4,-4,-4,-4,-2,-2,-2,-2,-2,0,0,0,0,2,2,2,2,2,4,4,4,4,4 };
const float threshold = 3 * 2500 * KK;

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
//void do_loop(float mat_out[H][W], float mat_in_pad[SIZE_H_PAD][SIZE_W_PAD], float mat_sqdiff[SIZE_H_REF][SIZE_W_REF])
//{
//	int i, j, x, y;
//	float sqdiff_current;
//	//#pragma EXPLORE_CONSTRAINT SAME = sqdiff_current
//	float sqdiff_prev;
//	//#pragma EXPLORE_CONSTRAINT SAME = sqdiff_current
//	float sqdiff_save;
//	//#pragma EXPLORE_CONSTRAINT SAME = mat_sqdiff
//	float tmp_dist;
//	float pre_exp;
//	//#pragma EXPLORE_FIX W={8} I={2}
//	float weight;
//	//#pragma EXPLORE_CONSTRAINT SAME = mat_in_pad
//	float denoised;
//	float dis[SIZE_H_PAD][SIZE_W_PAD] = { 0.0f };
//	float mat_out_nlm[H][W];
//	float mat_acc_weight[H][W];
//	float sigma2 = 900.0f;
//	int count = 0;
//	// Initialization
//	for (i = 0; i < H; i++)
//	{
//		for (j = 0; j < W; j++)
//		{
//			mat_out_nlm[i][j] = 0;
//			mat_acc_weight[i][j] = 0;
//		}
//	}
//	for (x = -4; x <= -4; x ++)
//		for (y = -4; y <=- 4; y ++)
//
//			if (!(x == 0 && y == 0)) {
//
//				for (i = CROP; i <5 + CROP; i++)
//				{
//					for (j = CROP; j < 5 + CROP; j++)
//					{
//						// Calculate square difference (and its prefix sum)
//						sqdiff_current = (float)((mat_in_pad[i][j] - mat_in_pad[i + y][j + x]) * (mat_in_pad[i][j] - mat_in_pad[i + y][j + x]));
//						if (i == CROP && j == CROP) {
//							sqdiff_save = sqdiff_current;
//							sqdiff_prev = sqdiff_current;
//						}
//						else if (i == CROP && j > CROP) {
//							sqdiff_save = sqdiff_current + sqdiff_prev;
//							sqdiff_prev = sqdiff_current + sqdiff_prev;
//						}
//						else if (i > CROP && j == CROP) {
//							sqdiff_save = sqdiff_current + mat_sqdiff[i - CROP - 1][j - CROP];
//							sqdiff_prev = sqdiff_current;
//						}
//						else if (i > CROP && j > CROP) {
//							sqdiff_save = sqdiff_current + sqdiff_prev + mat_sqdiff[i - CROP - 1][j - CROP];
//							sqdiff_prev = sqdiff_current + sqdiff_prev;
//						}
//						mat_sqdiff[i - CROP][j - CROP] = sqdiff_save;
//
//						// Calculate mat_denoised and mat_acc_weight
//						if ((i >= CROP + K - 1) && (j >= CROP + K - 1))
//						{
//							if (i == CROP + K - 1 && j == CROP + K - 1) {
//								tmp_dist = mat_sqdiff[i - CROP][j - CROP];
//								printf("(%d,%d): %f; ", (i), (j), tmp_dist);
//							}
//							else if (i > CROP + K - 1 && j == CROP + K - 1) {
//								tmp_dist = mat_sqdiff[i - CROP][j - CROP] - mat_sqdiff[i - K - CROP][j - CROP];
//							}
//							else if (i == CROP + K - 1 && j > CROP + K - 1) {
//								tmp_dist = mat_sqdiff[i - CROP][j - CROP] - mat_sqdiff[i - CROP][j - K - CROP];
//							}
//							else if (i > CROP + K - 1 && j > CROP + K - 1) {
//								tmp_dist = mat_sqdiff[i - CROP][j - CROP] - mat_sqdiff[i - K - CROP][j - CROP] - mat_sqdiff[i - CROP][j - K - CROP] + mat_sqdiff[i - K - CROP][j - K - CROP];
//							}
//							
//							//dis[i - K2 + y][j - K2 + x] = tmp_dist;
//							//printf("(%d,%d):%f; ", (i - K2 + y), (j - K2 + x), tmp_dist);
//							/*pre_exp = tmp_dist - 2 * sigma2 * KK;
//							if (pre_exp <= 0)
//								weight = 1;
//							else
//								weight = 0;*/
//
//								// Compute and accumulate denoised pixels
//								//denoised = weight * mat_in_pad[i - K2 + y][j - K2 + x];
//								//mat_out_nlm[i - CROP - K + 1][j - CROP - K + 1] += denoised;
//								// Update accumulated weights
//								//mat_acc_weight[i - CROP - K + 1][j - CROP - K + 1] += weight;
//								//printf("(%d,%d): %f; ", (i - CROP - K + 1), (j - CROP - K + 1), mat_acc_weight[i - CROP - K + 1][j - CROP - K + 1]);
//							/*for (int i = 0; i < SIZE_H_REF; i++)
//								for (int j = 0; j < SIZE_W_REF; j++)
//									printf("(%d,%d): %f; ", (i), (j), mat_sqdiff[i][j]);*/
//
//
//						}
//					}
//				}
//			}
//
//
//	// Initialization
//	//for (i = 0; i < H; i++)
//	//{
//	//	for (j = 0; j < W; j++)
//	//	{
//	//		mat_out_nlm[i][j] = 0;
//	//		mat_acc_weight[i][j] = 0;
//	//	}
//	//}
//
//	//for (int k = 0; k < PATCH_SIZE; k++)
//	//{
//	//	tmp_dist = 0;
//	//	sqdiff_prev = 0;
//	//	sqdiff_save = 0;
//	//	for (int i = CROP; i < SIZE_H_REF + CROP; i++)
//	//	{
//	//		for (int j = CROP; j < SIZE_W_REF + CROP; j++)
//	//		{
//	//			sqdiff_current = (double)((mat_in_pad[i][j] - mat_in_pad[i + dy_patch[k]][j + dx_patch[k]]) * (mat_in_pad[i][j] - mat_in_pad[i + dy_patch[k]][j + dx_patch[k]]));
//
//	//			if (i == CROP && j == CROP) {
//	//				sqdiff_save = sqdiff_current;
//	//				sqdiff_prev = sqdiff_current;
//	//			}
//	//			else if (i == CROP && j > CROP) {
//	//				sqdiff_save = sqdiff_current + sqdiff_prev;
//	//				sqdiff_prev = sqdiff_current + sqdiff_prev;
//	//			}
//	//			else if (i > CROP && j == CROP) {
//	//				sqdiff_save = sqdiff_current + mat_sqdiff[i - CROP - 1][j - CROP];
//	//				sqdiff_prev = sqdiff_current;
//	//			}
//	//			else if (i > CROP && j > CROP) {
//	//				sqdiff_save = sqdiff_current + sqdiff_prev + mat_sqdiff[i - CROP - 1][j - CROP];
//	//				sqdiff_prev = sqdiff_current + sqdiff_prev;
//	//			}
//	//			mat_sqdiff[i - CROP][j - CROP] = sqdiff_save;
//
//	//			/// Calculate mat_denoised and mat_acc_weight
//	//			if ((i >= CROP + K - 1) && (j >= CROP + K - 1))
//	//			{
//	//				if (i == CROP + K - 1 && j == CROP + K - 1) {
//	//					tmp_dist = mat_sqdiff[i - CROP][j - CROP];
//	//				}
//	//				else if (i > CROP + K - 1 && j == CROP + K - 1) {
//	//					tmp_dist = mat_sqdiff[i - CROP][j - CROP] - mat_sqdiff[i - K - CROP][j - CROP];
//	//				}
//	//				else if (i == CROP + K - 1 && j > CROP + K - 1) {
//	//					tmp_dist = mat_sqdiff[i - CROP][j - CROP] - mat_sqdiff[i - CROP][j - K - CROP];
//	//				}
//	//				else if (i > CROP + K - 1 && j > CROP + K - 1) {
//	//					tmp_dist = mat_sqdiff[i - CROP][j - CROP] - mat_sqdiff[i - K - CROP][j - CROP] - mat_sqdiff[i - CROP][j - K - CROP] + mat_sqdiff[i - K - CROP][j - K - CROP];
//	//				}
//	//				// Compute and accumulate denoised pixels
//	//				pre_exp = tmp_dist - 2 * sigma2 * KK;
//	//				if (pre_exp <= 0)
//	//					weight = 1;
//	//				else
//	//					weight = 0;
//	//				denoised = weight * mat_in_pad[i - K2 + dy_patch[k]][j - K2 + dx_patch[k]];
//	//				mat_out_nlm[i - CROP - K + 1][j - CROP - K + 1] = mat_out_nlm[i - CROP - K + 1][j - CROP - K + 1] + denoised;
//	//				printf("(%d,%d): %f; ", (i - CROP - K + 1), (i - CROP - K + 1), mat_out_nlm[i - CROP - K + 1][j - CROP - K + 1]);
//	//			}
//
//	//		}
//	//	}
//	//}
//}

void pad_matrix_symetry(float orig[H][W], float padded[SIZE_H_PAD][SIZE_W_PAD]) {
	// just for chls=1;
	int w = SIZE_W_PAD;
	int h = SIZE_H_PAD;

	//! Center of the image
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

void precompute_BM(
	float patch_table[NHard][SIZE_H_PAD][SIZE_W_PAD],
	float img[SIZE_H_PAD][SIZE_W_PAD] //H_pad*W_pad个元素，每个元素（vector）中包含k个（小于NHW最大数量）相似块的索引值（中心坐标）, 
	, const int width  //pad 后尺寸
	, const int height
	, const int kHW  //K
	, const int NHW   //最大相似块
	, const int nHW   //N
	, const int pHW   //步长
	,const unsigned tau_2D
	, const float    tauMatch
) {
	int i, j, n, k,di,dj,p,q;
	const int Ns = N;
	//const float threshold = tauMatch * kHW * kHW;
	const float threshold = 40000.0f;
	float diff_table[SIZE_H_PAD][SIZE_W_PAD] = { 0.0f};
	float  sum_table[SIZE_H_PAD*SIZE_W_PAD][N2 + 1][N];
	int row_ind[5] = { 0 };
	int column_ind[5] = { 0 };
	ind_initialize(row_ind, SIZE_H_PAD - kHW + 1, nHW, pHW);
	ind_initialize(column_ind, SIZE_W_PAD - kHW + 1, nHW, pHW);
	printf("%d,%d,%d,", row_ind[0], row_ind[1], row_ind[2]);





	float group_3D_table[kHard_2*NHard*column_index_size];   //  vector<float> group_3D_table(chnls * kHard_2 * NHard * column_ind.size());
	float  wx_r_table[column_index_size];	 //    vector<float> wx_r_table;  wx_r_table.reserve(chnls * column_ind.size());
	float hadamard_tmp[NHard];	 //    vector<float> hadamard_tmp(NHard);
		//! For aggregation part
	float denominator[SIZE_H_PAD][SIZE_W_PAD] = { 0.0f };	// vector<float> denominator(width * height * chnls, 0.0f);
	float numerator[SIZE_H_PAD][SIZE_W_PAD] = { 0.0f };	// vector<float> numerator(width * height * chnls, 0.0f);
	   //! table_2D[p * N + q + (i * width + j) * kHard_2 + c * (2 * nHard + 1) * width * kHard_2]
	float	table_2D[2 * N2 + 1][kHard_2] = { 0.0f };	// vector<float> table_2D((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);


	//! Loop on i_r
	for (unsigned ind_i = 0; ind_i < row_index_size; ind_i++)
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
		}
	}


	//Initialization
	for (i = 0; i < (SIZE_H_PAD * SIZE_W_PAD); i++)
		for (j = 0; j < N2 + 1; j++)
		{
			for (int n = 0; n < N; n++)
			{
				sum_table[i][j][n] = 2 * threshold;
				//printf("%f; ",sum_table[i][j][n]);
			}
		}

	for (int di = 0; di <= nHW; di++)
		for (int dj = 0; dj < Ns; dj++)
		{
			const int dk = (int)(di * width + dj) - (int)nHW;
			const int ddk = di * Ns + dj;

			//! Process the image containing the square distance between pixels
			for (i = nHW; i < height - nHW; i++)
			{

				for (k = nHW; k < width - nHW; k++)
					diff_table[i][k] = (img[i + di][k + dj - nHW] - img[i][k]) * (img[i + di][k + dj - nHW] - img[i][k]);

			}

			//! Compute the sum for each patches, using the method of the integral images
			const int dn = nHW * width + nHW;
			//! 1st patch, top left corner
			float value = 0.0f;
			for (int p = 0; p < kHW; p++)
			{
				int pq = p * width + dn;
				for (int q = 0; q < kHW; q++)
					value += diff_table[p + nHW][q + nHW];
			}
			sum_table[dn][di][dj] = value;

			//! 1st row, top
			for (int j = nHW + 1; j < width - nHW; j++)
			{
				const int ind = nHW * width + j - 1;
				float sum = sum_table[ind][di][dj];
				for (int p = 0; p < kHW; p++)
					sum += diff_table[nHW + p][j - 1 + kHW] - diff_table[nHW + p][j - 1];
				sum_table[ind + 1][di][dj] = sum;
				//printf("(%d,%d,%d): %f; ", (ind + 1), (di), (dj), sum_table[ind + 1][di][dj]);
			}

			//! General case
			for (int i = nHW + 1; i < height - nHW; i++)
			{
				const int ind = (i - 1) * width + nHW;
				float sum = sum_table[ind][di][dj];
				//! 1st column, left
				for (int q = 0; q < kHW; q++)
					sum += diff_table[i - 1 + kHW][nHW + q] - diff_table[i - 1][nHW + q];
				sum_table[ind + width][di][dj] = sum;

				//! Other columns
				int k = i * width + nHW + 1;
				int pq = (i + kHW - 1) * width + kHW - 1 + nHW + 1;
				for (int j = nHW + 1; j < width - nHW; j++, k++, pq++)
				{
					sum_table[k][di][dj] =
						sum_table[k - 1][di][dj]
						+ sum_table[k - width][di][dj]
						- sum_table[k - 1 - width][di][dj]
						+ diff_table[i + kHW - 1][kHW + j - 1]
						- diff_table[i + kHW - 1][j - 1]
						- diff_table[i - 1][j + kHW - 1]
						+ diff_table[i - 1][j - 1];
					//printf("(%d,%d,%d): %f; ", (k), (di), (dj), sum_table[k][di][dj]);
				}
				;
			}
		}

	//! Precompute Bloc Matching


	//vector<pair<float, unsigned> > table_distance;
	////! To avoid reallocation
	//table_distance.reserve(Ns * Ns);
	int BlockCount = 1;
	for (unsigned ind_i = 0; ind_i < row_index_size; ind_i++)
	{
		for (unsigned ind_j = 0; ind_j < column_index_size; ind_j++)
		{
			//! Initialization
			const unsigned k_r = row_ind[ind_i] * width + column_ind[ind_j];
			int BlockCount = 0;
			float table_distance[NHard] = { 0 };//搜索窗每个像素的 相似块pair(距离差，相似块索引）
			int index_w[NHard] = { 0 };
			int index_h[NHard] = { 0 };
			//table_distance.clear();
			//patch_table[k_r].clear();

			//! Threshold distances in order to keep similar patches
			for (int dj = -(int)nHW; dj <= (int)nHW; dj++)
			{
					for (int di = 0; di <= (int)nHW; di++){
						if (sum_table[k_r][di][dj + nHW] < threshold) {
							if (BlockCount < NHard)
							{
								table_distance[BlockCount] = sum_table[k_r][di][dj + nHW];
								index_h[BlockCount] = row_ind[ind_i] + di;
								index_w[BlockCount] = column_ind[ind_j] + dj;
								BlockCount++;
							}
							else if (BlockCount == NHard)//sort block by value of distance
							{
								float tmp;
								int ind_tmp1;
								int ind_tmp2;
								for(int p = 0; p < BlockCount-1; p++){
									for (int q = 0; q < BlockCount - 1; q++)
									{
										if (table_distance[q]> table_distance[q+1])
										{
											tmp = table_distance[q];
											table_distance[q] = table_distance[q + 1];
											table_distance[q + 1] = tmp;
											ind_tmp1 = index_h[q];
											index_h[q] = index_h[q + 1];
											index_h[q + 1] = ind_tmp1;
											ind_tmp2 = index_w[q];
											index_w[q] = index_w[q + 1];
											index_w[q + 1] = ind_tmp2;
										}
									}
								}
								{		if ((sum_table[k_r][di][dj + nHW])< table_distance[BlockCount-1])
										{
											table_distance[BlockCount - 1] = sum_table[k_r][di][dj + nHW];
											index_h[BlockCount - 1] = row_ind[ind_i] + di;
											index_w[BlockCount - 1] = column_ind[ind_j] + dj;
										}
									}
							}
							
						}
					}
					for (int di = -(int)nHW; di < 0; di++)
						if (sum_table[k_r][-di][-dj + nHW] < threshold) {
							if (BlockCount < NHard)
							{
								table_distance[BlockCount] = sum_table[k_r + di * width + dj][-di][-dj + nHW];
								index_h[BlockCount] = row_ind[ind_i] + di;
								index_w[BlockCount] = column_ind[ind_j] + dj;
								BlockCount++;
							}
							else if (BlockCount == NHard)
							{
								float tmp;
								int ind_tmp1;
								int ind_tmp2;
								for (int p = 0; p < BlockCount - 1; p++){
									for (int q = 0; q < BlockCount - 1; q++)
									{
										if (table_distance[q] > table_distance[q + 1])
										{
											tmp = table_distance[q];
											table_distance[q] = table_distance[q + 1];
											table_distance[q + 1] = tmp;
											ind_tmp1 = index_h[q];
											index_h[q] = index_h[q + 1];
											index_h[q + 1] = ind_tmp1;
											ind_tmp2 = index_w[q];
											index_w[q] = index_w[q + 1];
											index_w[q + 1] = ind_tmp2;
										}
									}
								}
								if ((sum_table[k_r + di * width + dj][-di][-dj + nHW])< table_distance[BlockCount - 1])
									{
										table_distance[BlockCount - 1] = sum_table[k_r + di * width + dj][-di][-dj + nHW];
										index_h[BlockCount - 1] = row_ind[ind_i] + di;
										index_w[BlockCount - 1] = column_ind[ind_j] + dj;
									}
							}
						}
					//table_distance.push_back(make_pair(
					//	sum_table[-dj + nHW + (-di) * Ns][k_r + di * width + dj]
					//	, k_r + di * width + dj));
			}
			for (int p = 0; p < BlockCount; p++)
			{
				patch_table[p][row_ind[ind_i]][column_ind[ind_j]] = table_distance[p];
				printf("k_r:%d,index:%d,value%f: ",k_r, (index_h[p]*width+ index_w[p]),patch_table[p][row_ind[ind_i]][column_ind[ind_j]]  );
			}
		}
	}
}
int main()
{
	float Image[Height][Width];
	float img[H][W];
	float mat_sqdiff[SIZE_H_REF][SIZE_W_REF];
	float pad[SIZE_H_PAD][SIZE_W_PAD];
	float patch_table[NHard][SIZE_H_PAD][SIZE_W_PAD] = { 0 };
	char input_img[32];
	char output_img[32];
	sprintf(input_img, "test.png");

	size_t nx, ny, nc;
	float* mat_in = NULL;
	mat_in = read_png_f32(input_img, &nx, &ny, &nc);
	if (!mat_in) {
		printf("error :: %s not found  or not a correct png image \n", input_img);
		exit(-1);
	}

	int i, j;
	for (i = 0; i < Height; i++)
	{
		for (j = 0; j < Width; j++)
		{
			Image[i][j] = mat_in[j + i * Width];
		}
	}
	for (i = 0; i < H; i++)
	{
		for (j = 0; j < W; j++)
		{
			img[i][j] = Image[i][j];
			//printf("%f ", img[i][j]
		}
	}

	pad_matrix_symetry(img, pad);
	//do_loop(img, pad, mat_sqdiff);
	precompute_BM(patch_table, pad, SIZE_W_PAD, SIZE_H_PAD, K, 16, N2, 3, 7500,DCT);


	//FILE* fpWrite = fopen("data.txt", "w");
	//if (fpWrite == NULL)
	//{
	//	return 0;
	//}

	//for (i = 0; i < SIZE_H_PAD; i++) {
	//	for (j = 0; j < SIZE_W_PAD; j++) {
	//		printf("%f ", pad[i][j]);
	//		fprintf(fpWrite, "%f ", pad[i][j]);
	//	}
	//	fprintf(fpWrite, " \r\n");
	//}
	//printf("\n");
	//fclose(fpWrite);

	//test
	/*int b[SIZE_H_PAD][SIZE_W_PAD] = { 0 };
	int i, j;
	pad_matrix_symetry(a, b);
	for (i = 0; i < SIZE_H_PAD; i++)
		for (j = 0; j < SIZE_W_PAD; j++) {
			printf("%d ", b[i][j]);
		}
		printf("\n");

	}
	printf("\n");


	/*int i, j;
	float img[H][W] = { {1,2,3,4,5,6,7},{1,2,3,4,5,6,7},{1,2,3,4,5,6,7},{1,2,3,4,5,6,7},{1,2,3,4,5,6,7},{1,2,3,4,5,6,7},{1,2,3,4,5,6,7} };
	float padded[SIZE_H_PAD][SIZE_W_PAD];

	pad_matrix_symetry(img, padded);

	for (i = 0; i < SIZE_H_PAD; i++)
		for (j = 0; j < SIZE_W_PAD; j++) {
			printf("%f ", padded[i][j]);
		}
	printf("\n");
*/

	return 0;

}
