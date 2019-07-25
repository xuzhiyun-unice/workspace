#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<time.h>
#include "io_png.h"
#define DTYPE int
#define H 384
#define chnls 1
#define W 512
#define K 8
#define N 33
#define K2 4
#define N2 16
#define pHard 3 //step
#define row_index_size 127
#define column_index_size 169
#define kHard 8   //kHard = (tau_2D_hard == BIOR || sigma < 40.f ? 8 : 12); //! Must be a power of 2 if tau_2D_hard == BIOR
#define kHard_2 (kHard*kHard)
#define kWien 8  // kWien = (tau_2D_wien == BIOR || sigma < 40.f ? 8 : 12); //! Must be a power of 2 if tau_2D_wien == BIOR
#define kWien_2 (kWien*kWien)
#define NHard 16 //MAX NUMBLE BLOCK step1 , Must be a power of 2
#define NWien 32 //MAX NUMBLE BLOCK step2 , Must be a power of 2
#define KK (K*K)
#define SIZE_H_PAD  (H+2*N2)
#define SIZE_W_PAD  (W+2*N2)
//size_h_ref = H + N2 - K2
#define SIZE_H_REF (H + N2 - K2)  
//size_w_ref = W + N2 - K2
#define SIZE_W_REF (W + N2 - K2) 
#define SQRT2     1.414213562373095
#define SQRT2_INV 0.7071067811865475

typedef float Float1;
typedef float Float2;
typedef float Float_group;


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


void pad_matrix_symetry(float orig[H][W], float padded[SIZE_H_PAD][SIZE_W_PAD]) {
	// just for chls=1;
	int w = SIZE_W_PAD;
	int h = SIZE_H_PAD;

	//! Center of the image
	for (int i = 0; i < H; i++)
		for (int j = 0; j < W; j++)
			padded[N2 + i][N2 + j] = orig[i][j];

	//! Top and bottom
	for (int j = 0; j < w; j++)
		for (int i = 0; i < N2; i++)
		{
			padded[i][j] = padded[2 * N2 - i - 1][j];
			padded[h - i - 1][j] = padded[h - 2 * N2 + i][j];
		}

	//! Right and left

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < N2; j++)
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

	for (int k = 0; k < 10; k++)
	{
		lp1[k] *= coef_norm;
		hp2[k] *= coef_norm;
	}
}


void precompute_BM(
	float patch_table[SIZE_H_PAD * SIZE_W_PAD],
	float img[SIZE_H_PAD][SIZE_W_PAD] //H_pad*W_pad个元素，每个元素（vector）中包含k个（小于NHW最大数量）相似块的索引值（中心坐标）, 
	, int row_ind[row_index_size]
	, int column_ind[column_index_size]
	, int index_h[SIZE_H_PAD * SIZE_W_PAD][NHard]
	, int index_w[SIZE_H_PAD * SIZE_W_PAD][NHard]
	, const int width  //pad 后尺寸
	, const int height
	, const int kHW  //K
	, const int NHW   //最大相似块
	, const int nHW   //N
	, const int pHW   //步长
	, const float    tauMatch
) {
	int i, j, n, k, di, dj, p, q;
	const int Ns = N;
	//const float threshold = tauMatch * kHW * kHW;
	const float threshold = 40000.0f;
	float diff_table[SIZE_H_PAD][SIZE_W_PAD] = { 0 };
	static float  sum_table[SIZE_H_PAD * SIZE_W_PAD][N2 + 1][N];

	//printf("%d,%d,%d,", row_ind[0], column_ind[168], row_ind[126]);
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
			}
		}

	//! Precompute Bloc Matching
	for (int  ind_i = 0; ind_i <row_index_size ; ind_i++)
	//for (int ind_i = 0; ind_i < 1; ind_i++)
	{
		for (int ind_j = 0; ind_j < column_index_size; ind_j++)
		//for (int ind_j = 0; ind_j < 1; ind_j++) 		
		{
			//! Initialization
			const int  k_r = row_ind[ind_i] * width + column_ind[ind_j];
			int BlockCount = 0;
			float table_distance[NHard] = { 0 };//搜索窗每个像素的 相似块pair(距离差，相似块索引）

			//! Threshold distances in order to keep similar patches
			for (int dj = -(int)nHW; dj <= (int)nHW; dj++)
			{
				for (int di = 0; di <= (int)nHW; di++) {
					if (sum_table[k_r][di][dj + nHW] < threshold) {
						if (BlockCount < NHard)
						{
							table_distance[BlockCount] = sum_table[k_r][di][dj + nHW];
							index_h[k_r][BlockCount] = row_ind[ind_i] + di;
							index_w[k_r][BlockCount] = column_ind[ind_j] + dj;
							BlockCount++;
						}
						else if (BlockCount == NHard)//sort block by value of distance
						{
							float tmp;
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
										ind_tmp1 = index_h[k_r][q];
										index_h[k_r][q] = index_h[k_r][q + 1];
										index_h[k_r][q + 1] = ind_tmp1;
										ind_tmp2 = index_w[k_r][q];
										index_w[k_r][q] = index_w[k_r][q + 1];
										index_w[k_r][q + 1] = ind_tmp2;
									}
								}
							}
							{		if ((sum_table[k_r][di][dj + nHW]) < table_distance[BlockCount - 1])
							{
								table_distance[BlockCount - 1] = sum_table[k_r][di][dj + nHW];
								index_h[k_r][BlockCount - 1] = row_ind[ind_i] + di;
								index_w[k_r][BlockCount - 1] = column_ind[ind_j] + dj;
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
							index_h[k_r][BlockCount] = row_ind[ind_i] + di;
							index_w[k_r][BlockCount] = column_ind[ind_j] + dj;
							BlockCount++;
						}
						else if (BlockCount == NHard)
						{
							float tmp;
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
										ind_tmp1 = index_h[k_r][q];
										index_h[k_r][q] = index_h[k_r][q + 1];
										index_h[k_r][q + 1] = ind_tmp1;
										ind_tmp2 = index_w[k_r][q];
										index_w[k_r][q] = index_w[k_r][q + 1];
										index_w[k_r][q + 1] = ind_tmp2;
									}
								}
							}
							if ((sum_table[k_r + di * width + dj][-di][-dj + nHW]) < table_distance[BlockCount - 1])
							{
								table_distance[BlockCount - 1] = sum_table[k_r + di * width + dj][-di][-dj + nHW];
								index_h[k_r][BlockCount - 1] = row_ind[ind_i] + di;
								index_w[k_r][BlockCount - 1] = column_ind[ind_j] + dj;
							}
						}
					}
			}
			//Number of Blocks must be power of 2;
			if (BlockCount < NHard) {
				if (BlockCount == 1)
				{
					printf("problem size \n");
				}
				BlockCount = closest_power_of_2(BlockCount);
			}
			patch_table[k_r] = BlockCount; //get number of block
			//printf("block:%d: ", BlockCount);
			//要考虑下 blockcount 不满16个的情况，动态分配
			//for (int p = 0; p < BlockCount; p++)
			//{
			//	if (BlockCount < NHard)
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
 //只对单独行上操作，然后到搜索窗下一个step的行进行dct；调用上次计算好的区域节省时间；

void dct_2d_process(float table_2D[N * SIZE_W_PAD][K][K], float img[SIZE_H_PAD][SIZE_W_PAD], int i_r
	, int i_min
	, int i_max) {
	const int size = chnls * SIZE_W_PAD * N;
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

		//! Re-use of DCT already processe  搜索窗在下一个pHard时，重复区域用上个pHard时计算好dct，节省时间
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
		//多出来step个地方像素要做dct而已
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
	//for (int p = 0; p < K; p++)
	//	for (int q = 0; q < K; q++) {
	//		printf("%f ", table_2D[0][p][q]);
	//	}
}


void dct_2d_inverse(
	float  group_3D_table[NHard * column_index_size][K][K]
) {
	//! 2D dct inverse

	for (int i = 0; i < column_index_size; i++)
	{
		for (int n = 0; n < NHard; n++)
		{
			dct_2d(group_3D_table[n + NHard * i], 1); //flag==1,inverer;
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
	float* group_3D
	, float tmp[NHard]
	, const unsigned nSx_r
	, float sigma
	, const float lambdaHard3D
	, float weight_table
) {
	//! Declarations

	const float coef_norm = sqrtf((float)nSx_r);
	const float coef = 1.0f / (float)nSx_r;

	//! Process the Welsh-Hadamard transform on the 3rd dimension
	for (unsigned n = 0; n < kHard_2 ; n++){
		hadamard_transform(group_3D, tmp, nSx_r, n * nSx_r);
	}
	//! Hard Thresholding

	const float T = lambdaHard3D * sigma * coef_norm;
	for (unsigned k = 0; k < kHard_2 * nSx_r; k++)
	{
		if (fabs(group_3D[k]) > T)
			weight_table++;
		else
			group_3D[k] = 0.0f;
	}


	//! Process of the Welsh-Hadamard inverse transform
	for (unsigned n = 0; n < kHard_2 * chnls; n++)
		hadamard_transform(group_3D, tmp, nSx_r, n * nSx_r);

	for (unsigned k = 0; k < nSx_r * KK; k++)
		group_3D[k] *= coef;

	//! Weight for aggregation
	//if (doWeight)
	//	for (unsigned c = 0; c < chnls; c++)
	//		weight_table[c] = (weight_table[c] > 0.0f ? 1.0f / (float)
	//		(sigma_table[c] * sigma_table[c] * weight_table[c]) : 1.0f);


}


int main()
{
	static float Image[H][W],img_out[H][W];
	static float pad[SIZE_H_PAD][SIZE_W_PAD], img_basic[SIZE_H_PAD][SIZE_W_PAD];
	static float patch_table[SIZE_H_PAD * SIZE_W_PAD] = { 0 }; //number of block
	static int index_w[SIZE_H_PAD * SIZE_W_PAD][NHard] = { 0 };//index of patch_table block
	static int index_h[SIZE_H_PAD * SIZE_W_PAD][NHard] = { 0 };//index of patch_table block
	float weight_table = 0.0f;	//vector<float> weight_table(chnls);
	float sigma = 10.0f;
	const float    lambdaHard3D = 2.7f;

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

	char input_img[32];
	char output_img[32];
	sprintf(input_img, "test.png");
	sprintf(output_img, "denoised.png");

	size_t nx, ny, nc;
	float* mat_in = NULL;
	float* mat_out = NULL;
	mat_in = read_png_f32(input_img, &nx, &ny, &nc);
	if (!mat_in) {
		printf("error :: %s not found  or not a correct png image \n", input_img);
		exit(-1);
	}

	int i, j, p, q, m, n;
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
	precompute_BM(patch_table, pad, row_ind, column_ind, index_h, index_w, SIZE_W_PAD, SIZE_H_PAD, K, NHard, N2, 3, 7500);

	//! Preprocessing of Bior table
	float lpd[10], hpd[10], lpr[10], hpr[10];
	bior15_coef(lpd, hpd, lpr, hpr);

	//! For aggregation part
	//vector float denominator(width * height * chnls, 0.0f);
	//vector<float> numerator(width * height * chnls, 0.0f);
	static float denominator[SIZE_H_PAD][SIZE_W_PAD] = { 0.0f };
	static float numerator[SIZE_H_PAD][SIZE_W_PAD] = { 0.0f };

	//! table_2D[p * N + q + (i * width + j) * kHard_2 + c * (2 * nHard + 1) * width * kHard_2]
	static float table_2D[N * SIZE_W_PAD][K][K] = { 0.0f };

	//vector<float> group_3D_table(chnls * kHard_2 * NHard * column_ind.size());
	//vector<float> wx_r_table;
	//wx_r_table.reserve(chnls * column_ind.size());
	//vector<float> hadamard_tmp(NHard);
	static float group_3D_table[NHard * column_index_size][K][K] = { 0.0f };
	static float wx_r_table[column_index_size] = { 0.0f };
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
			Float_group* group_3D = (Float_group*)calloc(nSx_r * KK, sizeof(Float_group));//vector<float> group_3D(chnls * nSx_r * kHard_2, 0.0f);
			for (unsigned n = 0; n < nSx_r; n++)
				//for (unsigned n = 0; n < 1; n++)
			{
				int index = index_h[k_r][n] * SIZE_W_PAD + index_w[k_r][n] + (N2 - i_r) * SIZE_W_PAD;
				for (int p = 0; p < K; p++) {
					for (int q = 0; q < K; q++) {
						group_3D[n + (p * K + q) * nSx_r] = table_2D[index][p][q];
						// printf("k_r:%d,index:(%d,%d),value%f: ", k_r, index_h[k_r][n], index_w[k_r][n], group_3D[n + (p * K + q) * nSx_r]);
					}
				}
			}

			//! HT filtering of the 3D group
			ht_filtering_hadamard(group_3D, hadamard_tmp, nSx_r, sigma, lambdaHard3D, weight_table);


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

			//! Save weighting
			wx_r_table[ind_j] = weight_table;
			free(group_3D);
			//printf("index:(%d,%d) ", ind_i, ind_j);
		} //! End of loop on j_r

		   //!  Apply 2D inverse transform
		dct_2d_inverse(group_3D_table);

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
							numerator[(index_h[k_r][n]+p)][(index_w[k_r][n]+q)] += kaiser_window[p][q]
								* wx_r_table[ ind_j ]
								* group_3D_table[  n+ind_j*NHard][p][q];
							denominator[(index_h[k_r][n] + p)][(index_w[k_r][n] + q)] += kaiser_window[p][q]
								* wx_r_table[ind_j ];
						}
				}
		}
		printf(" numerator: %f  ", numerator[ind_i][0]);
	}//! End of loop on i_r

	//! Final reconstruction
	for (int i = 0; i < SIZE_H_PAD; i++)
	{
		for (int j = 0; j < SIZE_W_PAD; j++)
		{
		img_basic[i][j] = numerator[i][j] / denominator[i][j];
		}
	}


	for (i = 0; i < H; i++)
	{
		for (j = 0; j < W; j++)
		{
			img_out[H][W] = img_basic[i + N2][j + N2];
		}
	}
	if (write_png_f32(output_img, img_out, (size_t)nx, (size_t)ny, (size_t)nc) != 0) {
		printf("... failed to save png image %s", output_img);
	}


	return 0;

}
