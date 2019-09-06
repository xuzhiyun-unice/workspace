#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<time.h>
#include "io_png.h"
#define DTYPE int
#define H 469
#define chnls 1
#define W 704
#define K 8
#define N 33
#define K2 4
#define N2 16
#define pHard 3 //step
#define kHard 8   //kHard = (tau_2D_hard == BIOR || sigma < 40.f ? 8 : 12); //! Must be a power of 2 if tau_2D_hard == BIOR
#define kHard_2 (kHard*kHard)
#define kWien 8  // kWien = (tau_2D_wien == BIOR || sigma < 40.f ? 8 : 12); //! Must be a power of 2 if tau_2D_wien == BIOR
#define kWien_2 (kWien*kWien)
#define NHard 16 //MAX NUMBLE BLOCK step1 , Must be a power of 2
#define NWien 32 //MAX NUMBLE BLOCK step2 , Must be a power of 2
#define KK (K*K)
#define SIZE_H_PAD  (H+2*N2)
#define SIZE_W_PAD  (W+2*N2)
#define row_index_size (2+(SIZE_H_PAD-N-K)/3)
#define column_index_size (2+(SIZE_W_PAD-N-K)/3)
//size_h_ref = H + N2 - K2
#define SIZE_H_REF (H + N2 - K2)  
//size_w_ref = W + N2 - K2
#define SIZE_W_REF (W + N2 - K2) 
#define SQRT2     1.414213562373095
#define SQRT2_INV 0.7071067811865475




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
void dct_2d(float A[K][K], int flag)
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

	if (flag == 0)
	{
		matrix_multiplication(T, A, res1);
		matrix_multiplication(res1, Tinv, res2); //T*A*T' == dct2(A)
	}
	if (flag == 1)
	{
		matrix_multiplication(Tinv, A, res1);
		matrix_multiplication(res1, T, res2); //T'*A*T == idct2(A)
	}
	int i, j;
	for (i = 0; i < K; ++i)
		for (j = 0; j < K; ++j) {
			A[i][j] = res2[i][j];
		}
}

/**
 * @brief Add boundaries by symetry
 *
 * @param orig : image to symetrize
 * @param padded : will contain img with symetrized boundaries
 *
 * @return none.
 **/
void pad_matrix_symetry(float orig[H][W], float padded[SIZE_H_PAD][SIZE_W_PAD]) {

	//! Center of the image
	for (int i = 0; i < H; i++)
		for (int j = 0; j < W; j++)
			padded[N2 + i][N2 + j] = orig[i][j];

	//! Top and bottom
	for (int j = 0; j < SIZE_W_PAD; j++)
		for (int i = 0; i < N2; i++)
		{
			padded[i][j] = padded[2 * N2 - i - 1][j];
			padded[SIZE_H_PAD - i - 1][j] = padded[SIZE_H_PAD - 2 * N2 + i][j];
		}

	//! Right and left

	for (int i = 0; i < SIZE_H_PAD; i++)
	{
		for (int j = 0; j < N2; j++)
		{
			padded[i][j] = padded[i][2 * N2 - j - 1];
			padded[i][SIZE_W_PAD - j - 1] = padded[i][SIZE_W_PAD - 2 * N2 + j];
		}
	}
}



void precompute_BM(
	float patch_table[SIZE_H_PAD * SIZE_W_PAD],
	float img[SIZE_H_PAD][SIZE_W_PAD] //H_pad*W_pad个元素，每个元素（vector）中包含k个（小于NHW最大数量）相似块的索引值（中心坐标）, 
	, int row_ind[row_index_size]
	, int column_ind[column_index_size]
	, int (*index_h)[SIZE_H_PAD * SIZE_W_PAD]
	, int (*index_w)[SIZE_H_PAD * SIZE_W_PAD]
	, const int width  //pad 后尺寸
	, const int height
	, const int kHW  //K
	, const int NHW   //最大相似块
	, const int nHW   //N
	, const int pHW   //步长
	, const float    tauMatch
) {
	int i, j, n, k, di, dj, p, q;
#pragma EXPLORE_FIX W={28} I={10} //coeff
	const float threshold = tauMatch * kHW * kHW;
	//const float threshold = 40000.0f;
#pragma EXPLORE_FIX W={28} I={10} //mat+
	static float diff_table[SIZE_H_PAD][SIZE_W_PAD] = { 0.0f };
#pragma EXPLORE_FIX W={28} I={10}  //mat+
	static float  sum_table[SIZE_H_PAD * SIZE_W_PAD][N2 + 1][N];
#pragma EXPLORE_FIX W={28} I={10}  //mat+
	float tmp;
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
		for (int dj = 0; dj < N; dj++)
		{
			const int dk = (int)(di * width + dj) - (int)nHW;
			const int ddk = di * N + dj;

			//! Process the image containing the square distance between pixels
			for (i = nHW; i < height - nHW; i++)
			{
				for (k = nHW; k < width - nHW; k++)
					diff_table[i][k] = (img[i + di][k + dj - nHW] - img[i][k]) * (img[i + di][k + dj - nHW] - img[i][k]);

			}

			//! Compute the sum for each patches, using the method of the integral images
			const int dn = nHW * width + nHW;
			//! 1st patch, top left corner
			tmp = 0.0f;
			for (int p = 0; p < kHW; p++)
			{
				int pq = p * width + dn;
				for (int q = 0; q < kHW; q++)
					tmp += diff_table[p + nHW][q + nHW];
			}
			sum_table[dn][di][dj] = tmp;

			//! 1st row, top
			for (int j = nHW + 1; j < width - nHW; j++)
			{
				const int ind = nHW * width + j - 1;
				tmp = sum_table[ind][di][dj];
				for (int p = 0; p < kHW; p++)
					tmp += diff_table[nHW + p][j - 1 + kHW] - diff_table[nHW + p][j - 1];
				sum_table[ind + 1][di][dj] = tmp;
				//printf("(%d,%d,%d): %f; ", (ind + 1), (di), (dj), sum_table[ind + 1][di][dj]);
			}

			//! General case
			for (int i = nHW + 1; i < height - nHW; i++)
			{
				const int ind = (i - 1) * width + nHW;
				float tmp = sum_table[ind][di][dj];
				//! 1st column, left
				for (int q = 0; q < kHW; q++)
					tmp += diff_table[i - 1 + kHW][nHW + q] - diff_table[i - 1][nHW + q];
				sum_table[ind + width][di][dj] = tmp;

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
			}
		}

	//! Precompute Bloc Matching
	for (int ind_i = 0; ind_i < row_index_size; ind_i++)
		//for (int ind_i = 0; ind_i < 1; ind_i++)
	{
		for (int ind_j = 0; ind_j < column_index_size; ind_j++)
			//for (int ind_j = 0; ind_j < 1; ind_j++) 		
		{
			//! Initialization
			const int  k_r = row_ind[ind_i] * width + column_ind[ind_j];
			int BlockCount = 0;


#pragma EXPLORE_FIX W={28} I={10}  //mat+
			float table_distance[NWien] = { 0.0f };//搜索窗每个像素的 相似块pair(距离差，相似块索引）

			//! Threshold distances in order to keep similar patches
			for (int dj = -(int)nHW; dj <= (int)nHW; dj++)
			{
				for (int di = 0; di <= (int)nHW; di++) {
					if (sum_table[k_r][di][dj + nHW] < threshold) {
						if (BlockCount < NHW)
						{
							table_distance[BlockCount] = sum_table[k_r][di][dj + nHW];
							index_h[BlockCount][k_r] = row_ind[ind_i] + di;
							index_w[BlockCount][k_r] = column_ind[ind_j] + dj;
							BlockCount++;
						}
						else if (BlockCount == NHW)//sort block by value of distance
						{
							
							int ind_tmp1;
							int ind_tmp2;
							for (int p = 0; p < BlockCount - 1; p++) {
								for (int q = 0; q < BlockCount - 1; q++)
								{
									if (table_distance[q] > table_distance[q + 1])
									{
										tmp = table_distance[q];
										table_distance[q] = table_distance[q + 1];
										table_distance[q + 1] = tmp;
										ind_tmp1 = index_h[q][k_r];
										index_h[q][k_r] = index_h[q + 1][k_r];
										index_h[q + 1][k_r] = ind_tmp1;
										ind_tmp2 = index_w[q][k_r];
										index_w[q][k_r] = index_w[q + 1][k_r];
										index_w[q + 1][k_r] = ind_tmp2;
									}
								}
							}
							{		if ((sum_table[k_r][di][dj + nHW]) < table_distance[BlockCount - 1])
							{
								table_distance[BlockCount - 1] = sum_table[k_r][di][dj + nHW];
								index_h[BlockCount - 1][k_r] = row_ind[ind_i] + di;
								index_w[BlockCount - 1][k_r] = column_ind[ind_j] + dj;
							}
							}
						}
					}
				}
				for (int di = -(int)nHW; di < 0; di++)
					if (sum_table[k_r][-di][-dj + nHW] < threshold) {
						if (BlockCount < NHW)
						{
							table_distance[BlockCount] = sum_table[k_r + di * width + dj][-di][-dj + nHW];
							index_h[BlockCount][k_r] = row_ind[ind_i] + di;
							index_w[BlockCount][k_r] = column_ind[ind_j] + dj;
							BlockCount++;
						}
						else if (BlockCount == NHW)
						{
							int ind_tmp1;
							int ind_tmp2;
							for (int p = 0; p < BlockCount - 1; p++) {
								for (int q = 0; q < BlockCount - 1; q++)
								{
									if (table_distance[q] > table_distance[q + 1])
									{
										tmp = table_distance[q];
										table_distance[q] = table_distance[q + 1];
										table_distance[q + 1] = tmp;
										ind_tmp1 = index_h[q][k_r];
										index_h[q][k_r] = index_h[q + 1][k_r];
										index_h[q + 1][k_r] = ind_tmp1;
										ind_tmp2 = index_w[q][k_r];
										index_w[q][k_r] = index_w[q + 1][k_r];
										index_w[q + 1][k_r] = ind_tmp2;
									}
								}
							}
							if ((sum_table[k_r + di * width + dj][-di][-dj + nHW]) < table_distance[BlockCount - 1])
							{
								table_distance[BlockCount - 1] = sum_table[k_r + di * width + dj][-di][-dj + nHW];
								index_h[BlockCount - 1][k_r] = row_ind[ind_i] + di;
								index_w[BlockCount - 1][k_r] = column_ind[ind_j] + dj;
							}
						}
					}
			}
			//Number of Blocks must be power of 2;
			if (BlockCount < NHW) {
				//if (BlockCount == 1)
				//{
				//	printf("problem size ,%d \n", k_r);
				//}
				BlockCount = closest_power_of_2(BlockCount);
			}
			patch_table[k_r] = BlockCount; //get number of block
			//printf("block:%d: ", BlockCount);
			//要考虑下 blockcount 不满16个的情况，动态分配
			//for (int p = 0; p < BlockCount; p++)
			//{
			//	if (BlockCount < NHW)
			//	patch_table[k_r] = table_distance[p];
			//	//printf("k_r:%d,index:(%d,%d),value%f: ", k_r, index_h[k_r][p], index_w[k_r][p], patch_table[k_r][p]);
			//}
		}
	}
}


/**
 * @brief Precompute a 2D DCT transform on all patches contained in
 *        a part of the image.
 *
 * @param table_2D : will contain the 2d DCT transform for all
 *        chosen patches;
 * @param img : image on which the 2d DCT will be processed;
 * @param i_r: current index of the reference patches;
 * @param pHard: step,space in pixels between two references patches;
 * @param i_min (resp. i_max) : minimum (resp. maximum) value
 *        for i_r. In this case the whole 2d transform is applied
 *        on every patches. Otherwise the precomputed 2d DCT is re-used
 *        without processing it.
 **/
void dct_2d_process(float table_2D[N * SIZE_W_PAD][K][K], float img[SIZE_H_PAD][SIZE_W_PAD], int i_r
	, int i_min
	, int i_max) {
	int i, j, k, p, q;
	//! If i_r == ns, then we have to process all DCT
	if (i_r == i_min || i_r == i_max)
	{
		//! Allocating Memory
		//float vec[N * SIZE_W_PAD][K][K] = { 0.0f };
		for (int i = 0; i < N; i++)
			for (int j = 0; j < SIZE_W_PAD - K; j++) {
				for (int p = 0; p < K; p++) {
					for (int q = 0; q < K; q++) {
						table_2D[(i * SIZE_W_PAD + j)][p][q] =
							img[(i_r + i - N2 + p)][j + q];
					}
				}
				dct_2d(table_2D[(i * SIZE_W_PAD + j)], 0); //! Process of all DCTs,flag=0;			
			}
	}
	else
	{
		int ds = pHard * SIZE_W_PAD;
		//! Re-use of DCT already processe  
		for (int i = 0; i < N - pHard; i++)
			for (int j = 0; j < SIZE_W_PAD - K; j++) {
				for (int p = 0; p < K; p++) {
					for (int q = 0; q < K; q++) {
						table_2D[(i * SIZE_W_PAD + j)][p][q] =
							table_2D[(i * SIZE_W_PAD + j) + ds][p][q];
					}
				}
			}
		//! Compute the new DCT
		for (int i = 0; i < pHard; i++)
			for (int j = 0; j < SIZE_W_PAD - K; j++) {
				for (int p = 0; p < K; p++) {
					for (int q = 0; q < K; q++) {
						table_2D[(i + N - pHard) * SIZE_W_PAD + j][p][q] =
							img[(p + i + N - pHard + i_r - N2)][j + q];
					}
				}
				dct_2d(table_2D[(i + N - pHard) * SIZE_W_PAD + j], 0); //! Process of new DCTs,flag=0;
			}
	}
}


void dct_2d_inverse(
	float  (*group_3D_table)[K][K],
	float NHW
) {
	//! 2D dct inverse

	for (int i = 0; i < column_index_size; i++)
	{
		for (int n = 0; n < NHW; n++)
			if (NHW == NHard)
			{
				dct_2d(group_3D_table[n + NHard * i], 1); //flag==1,inverse dct_2d;
			}
			else
			{
				dct_2d(group_3D_table[n + NWien * i], 1);
			}

	}
}


///**
// * @brief Precompute a 2D bior1.5 transform on all patches contained in
// *        a part of the image.
// *
// * @param bior_table_2D : will contain the 2d bior1.5 transform for all
// *        chosen patches;
// * @param img : image on which the 2d transform will be processed;
// * @param width, height, chnls: size of img;
// * @param kHW : size of patches (kHW x kHW). MUST BE A POWER OF 2 !!!
// * @param i_r: current index of the reference patches;
// * @param step: space in pixels between two references patches;
// * @param i_min (resp. i_max) : minimum (resp. maximum) value
// *        for i_r. In this case the whole 2d transform is applied
// *        on every patches. Otherwise the precomputed 2d DCT is re-used
// *        without processing it;
// * @param lpd : low pass filter of the forward bior1.5 2d transform;
// * @param hpd : high pass filter of the forward bior1.5 2d transform.
// **/
//
//void bior_2d_process(float table_2D[N * SIZE_W_PAD][K][K], float img[SIZE_H_PAD][SIZE_W_PAD], int i_r
//	, int i_min
//	, int i_max) {
//
//}





/**
 * @brief Apply Welsh-Hadamard transform on vec (non normalized !!)
 *
 * @param vec: vector on which a Hadamard transform will be applied.
 *        It will contain the transform at the end;
 * @param tmp: must have the same size as vec. Used for convenience;
 * @param number, d: the Hadamard transform will be applied on vec[d] -> vec[d + number].
 *        number must be a power of 2!!!!
 *
 * @return numberone.
 **/
void hadamard_transform(
	float* vec
	, float* tmp
	, const unsigned number
	, const unsigned D
) {
	if (number == 1)
		return;
	else if (number == 2)
	{
		const float a = vec[D + 0];
		const float b = vec[D + 1];
		vec[D + 0] = a + b;
		vec[D + 1] = a - b;
	}
	else
	{
		const unsigned n = number / 2;
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
	float *group_3D
	, float *tmp
	, const unsigned nSx_r
	, float sigma
	, const float lambdaHard3D
	, float* weight_table
) {
	//! Declarations

	const float coef_norm = sqrtf((float)nSx_r);
	const float coef = 1.0f / (float)nSx_r;
	int count = 0.0f;
	//! Process the Welsh-Hadamard transform on the 3rd dimension
	for (unsigned n = 0; n < kHard_2; n++) {
		hadamard_transform(group_3D, tmp, nSx_r, n * nSx_r);
	}
	//! Hard Thresholding

	const float T = lambdaHard3D * sigma * coef_norm;
	for (unsigned k = 0; k < kHard_2 * nSx_r; k++)
	{
		if (fabs(group_3D[k]) > T) {
			count++;
		}
		else
			group_3D[k] = 0.0f;
	}
	*weight_table = (float)count;

	//! Process of the Welsh-Hadamard inverse transform
	for (unsigned n = 0; n < kHard_2; n++)
		hadamard_transform(group_3D, tmp, nSx_r, n * nSx_r);

	for (unsigned k = 0; k < nSx_r * KK; k++) {
		group_3D[k] *= coef;
		//	printf("T2: %d ", (group_3D[k]));
	}
	//! Weight for aggregation
	//if (doWeight)
	//	for (unsigned c = 0; c < chnls; c++)
	//		weight_table[c] = (weight_table[c] > 0.0f ? 1.0f / (float)
	//		(sigma_table[c] * sigma_table[c] * weight_table[c]) : 1.0f);
}

void wiener_filtering_hadamard(
	float group_3D[NWien * KK]
	, float group_3D_est[NWien * KK]
	, float tmp[NWien]
	, const unsigned nSx_r
	, float sigma
	, float* weight_table
) {
	const float coef = 1.0f / (float)nSx_r;
	float count = 0.0f;
	//! Process of the Welsh-Hadamard inverse transform
	for (unsigned n = 0; n < kWien_2; n++) {
		hadamard_transform(group_3D, tmp, nSx_r, n * nSx_r);
		hadamard_transform(group_3D_est, tmp, nSx_r, n * nSx_r);
	}

	//! Wiener Filtering
	for (unsigned k = 0; k < kWien_2 * nSx_r; k++)
	{
		float value = group_3D_est[k] * group_3D_est[k] * coef;
		value /= (value + sigma * sigma);
		group_3D_est[k] = group_3D[k] * value * coef;
		count += value;
	}
	*weight_table = (float)count;
	//printf("T2: %f ", (weight_table));
//! Process of the Welsh-Hadamard inverse transform
	for (unsigned n = 0; n < kWien_2; n++)
		hadamard_transform(group_3D_est, tmp, nSx_r, n * nSx_r);

	////! Weight for aggregation
	//if (doWeight)
	//	for (unsigned c = 0; c < chnls; c++)
	//		weight_table[c] = (weight_table[c] > 0.0f ? 1.0f / (float)
	//		(sigma_table[c] * sigma_table[c] * weight_table[c]) : 1.0f);
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
void compute_psnr(
	float img_1[H][W]
	, float img_2[H][W]
	, float* psnr
	, float* rmse
)
{
	float tmp = 0.0f;
	for (unsigned i = 0; i < H; i++)
		for (unsigned j = 0; j < W; j++) {
			tmp += (img_1[i][j] - img_2[i][j]) * (img_1[i][j] - img_2[i][j]);
		}
	(*rmse) = sqrtf(tmp / (float)(H * W));
	(*psnr) = 20.0f * log10f(255.0f / (*rmse));
}


int main()
{
	static float Image[H][W], image_step1[H][W], img_out[H * W];
	static float pad[SIZE_H_PAD][SIZE_W_PAD], img_basic[SIZE_H_PAD][SIZE_W_PAD], img_denoised[SIZE_H_PAD][SIZE_W_PAD];
	static float patch_table[SIZE_H_PAD * SIZE_W_PAD] = { 0.0f }; //number of block
	static int index_w[NWien][SIZE_H_PAD * SIZE_W_PAD] = { 0 };//index of patch_table block
	static int index_h[NWien][SIZE_H_PAD * SIZE_W_PAD] = { 0 };//index of patch_table block
	float sigma = 10.0f;
	static float tauMatch[2] = { 0 }; //! threshold used to determinate similarity between patches
	tauMatch[0] = 7500.0f;
	tauMatch[1] = (sigma < 35.0f ? 400 : 3500); //! threshold used to determinate similarity between patches
	float psnr[2], rmse[2];
	float    lambdaHard3D = 2.1f;
	//! Kaiser Window coefficients
	float kaiser_window[K][K] =
	{ { 0.1924f, 0.2989f, 0.3846f, 0.4325f, 0.4325f, 0.3846f, 0.2989f, 0.1924f},
	  { 0.2989f, 0.4642f, 0.5974f, 0.6717f, 0.6717f, 0.5974f, 0.4642f, 0.2989f },
	  { 0.3846f, 0.5974f, 0.7688f, 0.8644f,0.8644f, 0.7688f, 0.5974f, 0.3846f },
	  { 0.4325f, 0.6717f, 0.8644f, 0.9718f, 0.9718f, 0.8644f,0.6717f, 0.4325f },
	  { 0.4325f, 0.6717f, 0.8644f, 0.9718f, 0.9718f, 0.8644f, 0.6717f, 0.4325f },
	  { 0.3846f, 0.5974f, 0.7688f, 0.8644f, 0.8644f, 0.7688f, 0.5974f, 0.3846f },
	  { 0.2989f, 0.4642f,0.5974f, 0.6717f, 0.6717f, 0.5974f, 0.4642f, 0.2989f },
   {0.1924f, 0.2989f, 0.3846f, 0.4325f, 0.4325f, 0.3846f, 0.2989f, 0.1924f } };
	int i, j, p, q, m, n;
	char input_img[32];
	char output_img[32];
	char output_img_step2[32];
	sprintf(input_img, "ImNoisy.png");
	sprintf(output_img, "denoised2.png");
	sprintf(output_img_step2, "denoised2_step2.png");

	//sprintf(input_img, "test_sig10.png");
	//sprintf(output_img, "denoised_sig10.png");

	size_t nx, ny, nc;
	static float* mat_in = NULL;
	static float* mat_out = NULL;
	mat_in = io_png_read_f32(input_img, &nx, &ny, &nc);
	if (!mat_in) {
		printf("error :: %s not found  or not a correct png image \n", input_img);
		exit(-1);
	}
	printf("image size :\n");
	printf(" - width          = %d  \n", nx);
	printf(" - height         = %d  \n", ny);
	printf(" - nb of channels = %d  \n", nc);

	for (i = 0; i < H; i++)
	{
		for (j = 0; j < W; j++)
		{
			Image[i][j] = mat_in[j + i * W];
		}
	}


	pad_matrix_symetry(Image, pad);
	//do_loop(img, pad, mat_sqdiff);

	static int row_ind[row_index_size] = { 0 };
	static int column_ind[column_index_size] = { 0 };
	ind_initialize(row_ind, SIZE_H_PAD - K + 1, N2, pHard);
	ind_initialize(column_ind, SIZE_W_PAD - K + 1, N2, pHard);
	//printf("%d,%d,%d,", row_ind[0], column_ind[168], row_ind[126]);
	precompute_BM(patch_table, pad, row_ind, column_ind, index_h, index_w, SIZE_W_PAD, SIZE_H_PAD, K, NHard, N2, pHard, tauMatch[0]);



	//! For aggregation part
	//vector float denominator(width * height * chnls, 0.0f);
	//vector<float> numerator(width * height * chnls, 0.0f);
	static float denominator[SIZE_H_PAD][SIZE_W_PAD] = { 0.0f };
	static float numerator[SIZE_H_PAD][SIZE_W_PAD] = { 0.0f };

	//! table_2D[p * N + q + (i * width + j) * kHard_2 + c * (2 * nHard + 1) * width * kHard_2]
	static float table_2D[N * SIZE_W_PAD][K][K] = { 0.0f };
	static float table_2D_est[N * SIZE_W_PAD][K][K] = { 0.0f };
	//vector<float> group_3D_table(chnls * kHard_2 * NHard * column_ind.size());
	//vector<float> wx_r_table;
	//wx_r_table.reserve(chnls * column_ind.size());
	//vector<float> hadamard_tmp(NHard);
	static float group_3D_table[NHard * column_index_size][K][K] = { 0.0f };
	static float weight_table[column_index_size] = { 0.0f };
	static float hadamard_tmp[NHard] = { 0.0f };



	for (int ind_i = 0; ind_i < row_index_size; ind_i++)
		//for (int ind_i = 0; ind_i < 1; ind_i++)
	{
		const int i_r = row_ind[ind_i];
		//printf("%d  ", i_r);
		//! Update of table_2D
	/*		dct_2d_process(table_2D, pad, i_r,
				row_ind[0], row_ind[row_index_size - 1]);*/
		dct_2d_process(table_2D, pad, i_r, row_ind[0], row_ind[0]);

		// //tau_2D == BIOR
		//bior_2d_process(table_2D, pad, i_r, row_ind[0], row_ind[0], lpd, hpd);
	//wx_r_table.clear();
	//group_3D_table.clear();

	//! Loop on j_r
		for (unsigned ind_j = 0; ind_j < column_index_size; ind_j++)
			//for (unsigned ind_j = 0; ind_j < 1; ind_j++)
		{
			//! Initialization
			const unsigned j_r = column_ind[ind_j];
			const unsigned k_r = i_r * SIZE_W_PAD + j_r;
			//! Number of similar patches
			const unsigned nSx_r = patch_table[k_r];
			//! Build of the 3D group
			static float group_3D[NHard * KK] = { 0.0f };
			//Float_group* group_3D = (Float_group*)calloc(nSx_r * KK, sizeof(Float_group));//vector<float> group_3D(chnls * nSx_r * kHard_2, 0.0f);
			for (unsigned n = 0; n < nSx_r; n++)
				//for (unsigned n = 0; n < 1; n++)
			{
				int index = index_h[n][k_r] * SIZE_W_PAD + index_w[n][k_r] + (N2 - i_r) * SIZE_W_PAD;
				for (int p = 0; p < K; p++) {
					for (int q = 0; q < K; q++) {
						group_3D[n + (p * K + q) * nSx_r] = table_2D[index][p][q];
						//printf("k_r:%d,index:(%d,%d),value%f: ", k_r, index_h[k_r][n], index_w[k_r][n],group_3D[n + (p * K + q) * nSx_r]);
					}
				}
			}
			//! HT filtering of the 3D group
			ht_filtering_hadamard(group_3D, hadamard_tmp, nSx_r, sigma, lambdaHard3D, &weight_table[ind_j]);

			//! 3D weighting using Standard Deviation

			//if (useSD)
			//	sd_weighting(group_3D, nSx_r, kHard, chnls, weight_table);

			//! Save the 3D group. The DCT 2D inverse will be done after.
			for (unsigned n = 0; n < nSx_r; n++) {
				for (int p = 0; p < K; p++) {
					for (int q = 0; q < K; q++) {
						group_3D_table[n + ind_j * NHard][p][q] = group_3D[n + (p * K + q) * nSx_r];
					}
				}
			}
			//free(group_3D);
			//printf("index:(%d,%d) ", ind_i, ind_j);
		} //! End of loop on j_r

		   //!  Apply 2D inverse transform
		dct_2d_inverse(group_3D_table,NHard);

		//for (unsigned n = 0; n < NHard; n++) {
		//	for (int p = 0; p < K; p++) {
		//		for (int q = 0; q < K; q++) {
		//			printf("%f  ", group_3D_table[n][p][q]);
		//		}
		//	}
		//	printf(" \n");
		//}
		//! Registration of the weighted estimation

		for (unsigned ind_j = 0; ind_j < column_index_size; ind_j++)
			//for (unsigned ind_j = 0; ind_j < 1; ind_j++)
		{
			const unsigned j_r = column_ind[ind_j];
			const unsigned k_r = i_r * SIZE_W_PAD + j_r;
			const unsigned nSx_r = patch_table[k_r];
			for (unsigned n = 0; n < nSx_r; n++)
			{
				for (unsigned p = 0; p < kHard; p++)
					for (unsigned q = 0; q < kHard; q++)
					{
						numerator[(index_h[n][k_r] + p)][(index_w[n][k_r] + q)] += kaiser_window[p][q]
							* weight_table[ind_j]
							* group_3D_table[n + ind_j * NHard][p][q];
						//printf(" ka %f ,W : %f, g: %f ", kaiser_window[p][q], wx_r_table[ind_j], group_3D_table[n + ind_j * NHard][p][q]);

						denominator[(index_h[n][k_r] + p)][(index_w[n][k_r] + q)] += kaiser_window[p][q]
							* weight_table[ind_j];
					}
			}
		}
		//printf(" numerator: %f  ", numerator[ind_i][0]);
	}//! End of loop on i_r

	//! Final reconstruction
	for (int i = 0; i < SIZE_H_PAD; i++)
	{
		for (int j = 0; j < SIZE_W_PAD; j++)
		{
			img_basic[i][j] = numerator[i][j] / denominator[i][j];
			//	printf(" img_basic: %f  ", img_basic[i][j]);
		}
	}

	for (i = 0; i < H; i++)
	{
		for (j = 0; j < W; j++)
		{
			img_out[i * W + j] = img_basic[i + N2][j + N2];
		}
	}
	if (io_png_write_f32(output_img, img_out, (size_t)nx, (size_t)ny, (size_t)nc) != 0) {
		printf("... failed to save png image %s", output_img);
	}

	for (i = 0; i < H; i++)
	{
		for (j = 0; j < W; j++)
		{
			image_step1[i][j] = img_basic[i + N2][j + N2];
		}
	}
	compute_psnr(Image, image_step1, &psnr[0], &rmse[0]);
	printf("For image after step 1 : \n");
	printf("PSNR : %f \n", psnr[0]);
	printf("RMSE : %f \n", rmse[0]);



	//step2
	static float wien_tmp[NWien] = { 0.0f };
	static float group_3D_table_wien[NWien * column_index_size][K][K] = { 0.0f };
	pad_matrix_symetry(image_step1, img_basic);
	precompute_BM(patch_table, img_basic, row_ind, column_ind, index_h, index_w, SIZE_W_PAD, SIZE_H_PAD, K, NWien, N2, pHard, tauMatch[1]);

	for (int ind_i = 0; ind_i < row_index_size; ind_i++)
		//for (int ind_i = 0; ind_i < 1; ind_i++)
	{
		const int i_r = row_ind[ind_i];

		dct_2d_process(table_2D, pad, i_r, row_ind[0], row_ind[0]);
		dct_2d_process(table_2D_est, img_basic, i_r, row_ind[0], row_ind[0]);


		//! Loop on j_r
		for (unsigned ind_j = 0; ind_j < column_index_size; ind_j++)
			//for (unsigned ind_j = 0; ind_j < 1; ind_j++)
		{
			//! Initialization
			const unsigned j_r = column_ind[ind_j];
			const unsigned k_r = i_r * SIZE_W_PAD + j_r;
			//! Number of similar patches
			const unsigned nSx_r = patch_table[k_r];
			//! Build of the 3D group
			static float group_3D[NWien * KK] = { 0.0f };
			static float group_3D_est[NWien * KK] = { 0.0f };
			//Float_group* group_3D = (Float_group*)calloc(nSx_r * KK, sizeof(Float_group));//vector<float> group_3D(chnls * nSx_r * kHard_2, 0.0f);
			for (unsigned n = 0; n < nSx_r; n++)
				//for (unsigned n = 0; n < 1; n++)
			{
				int index = index_h[n][k_r] * SIZE_W_PAD + index_w[n][k_r] + (N2 - i_r) * SIZE_W_PAD;
				for (int p = 0; p < K; p++) {
					for (int q = 0; q < K; q++) {
						group_3D[n + (p * K + q) * nSx_r] = table_2D[index][p][q];
						group_3D_est[n + (p * K + q) * nSx_r] = table_2D_est[index][p][q];
						//printf("k_r:%d,index:(%d,%d),v1: %f, v2: %f ", k_r, index_h[k_r][n], index_w[k_r][n],group_3D[n + (p * K + q) * nSx_r], group_3D_est[n + (p * K + q) * nSx_r]);
					}
				}
			}
			//! HT filtering of the 3D group
			wiener_filtering_hadamard(group_3D, group_3D_est, wien_tmp, nSx_r, sigma, &weight_table[ind_j]);
			//for (unsigned n = 0; n < 2; n++)
			//		for (int p = 0; p < K; p++) {
			//			for (int q = 0; q < K; q++) {
			//			printf("v1: %f, v2: %f ", group_3D[n + (p * K + q) * nSx_r], group_3D_est[n + (p * K + q) * nSx_r]);
			//		}
			//	}

			//! 3D weighting using Standard Deviation

			//if (useSD)
			//	sd_weighting(group_3D, nSx_r, kHard, chnls, weight_table);

			//! Save the 3D group. The DCT 2D inverse will be done after.
			for (unsigned n = 0; n < nSx_r; n++) {
				for (int p = 0; p < K; p++) {
					for (int q = 0; q < K; q++) {
						group_3D_table_wien[n + ind_j * NWien][p][q] = group_3D_est[n + (p * K + q) * nSx_r];
					}
				}
			}
			//free(group_3D);
			//printf("index:(%d,%d) ", ind_i, ind_j);
		} //! End of loop on j_r

		   //!  Apply 2D inverse transform
		dct_2d_inverse(group_3D_table_wien,NWien);
		//! Registration of the weighted estimation

		for (unsigned ind_j = 0; ind_j < column_index_size; ind_j++)
			//for (unsigned ind_j = 0; ind_j < 1; ind_j++)
		{
			const unsigned j_r = column_ind[ind_j];
			const unsigned k_r = i_r * SIZE_W_PAD + j_r;
			const unsigned nSx_r = patch_table[k_r];
			for (unsigned n = 0; n < nSx_r; n++)
			{
				for (unsigned p = 0; p < kHard; p++)
					for (unsigned q = 0; q < kHard; q++)
					{
						numerator[(index_h[n][k_r] + p)][(index_w[n][k_r] + q)] += kaiser_window[p][q]
							* weight_table[ind_j]
							* group_3D_table_wien[n + ind_j * NWien][p][q];
						//printf(" ka %f ,W : %f, g: %f ", kaiser_window[p][q], weight_table[ind_j],group_3D_table_wien[n + ind_j * NWien][p][q]);

						denominator[(index_h[n][k_r] + p)][(index_w[n][k_r] + q)] += kaiser_window[p][q]
							* weight_table[ind_j];
					}
			}
		}
		//printf(" numerator: %f  ", numerator[ind_i][0]);
	}//! End of loop on i_r


	//! Final reconstruction
	for (int i = 0; i < SIZE_H_PAD; i++)
	{
		for (int j = 0; j < SIZE_W_PAD; j++)
		{
			img_basic[i][j] = numerator[i][j] / denominator[i][j];
			//printf(" img_basic: %f  ", img_basic[i][j]);
		}
	}

	for (i = 0; i < H; i++)
	{
		for (j = 0; j < W; j++)
		{
			img_out[i * W + j] = img_basic[i + N2][j + N2];
		}
	}
	if (io_png_write_f32(output_img_step2, img_out, (size_t)nx, (size_t)ny, (size_t)nc) != 0) {
		printf("... failed to save png image %s", output_img);
	}

	for (i = 0; i < H; i++)
	{
		for (j = 0; j < W; j++)
		{
			image_step1[i][j] = img_basic[i + N2][j + N2];
		}
	}
	compute_psnr(Image, image_step1, &psnr[1], &rmse[1]);
	printf("For image after step 2 : \n");
	printf("PSNR : %f \n", psnr[1]);
	printf("RMSE : %f \n", rmse[1]);
	return 0;

}
