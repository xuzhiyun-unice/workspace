#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#define BlockSize1 8
#define BlockSize2 8
#define BlockStep1 3
#define BlockStep2 3
#define BlockSearch1 19
#define BlockSearch2 19
#define BlockMatch1 16
#define BlockMatch2 32
#define Threshold1 2500
#define Threshold2 400
#define Lemda1_3D 2.7//注意使用时要乘上噪声标准差
#define BetaHt 2.0
#define BetaWie 2.0
#define Width 256
#define Height 256
#define NoiseStadard 25.0
#define fillen_haar 2
#define fillen_bior 10
#define M_PI 3.1415926
///////////////////////////////////
static double KDCT[8][8];
static double KIDCT[8][8];
void DCTInit()
{
  int i,j;
  double c0,c1;
  c0=1/sqrt(8.0);
  c1=c0*sqrt(2.0);
  for(i=0;i<8;i++)
    for(j=0;j<8;j++)
      KDCT[i][j]=cos((2*j+1)*i*M_PI/16.0);
  //Normalization
  for(i=0;i<8;i++)
    if(i==0)
      for(j=0;j<8;j++)
        KDCT[i][j]*=c0;
    else
      for(j=0;j<8;j++)
        KDCT[i][j]*=c1;
  for(i=0;i<8;i++)
    for(j=0;j<8;j++)
      KIDCT[i][j]=KDCT[j][i];
}
void DCT1D(double * x)//routine to perform 8 point 1-D DCT
{
  int i,j;
  double y[8];//temp array
  double t;
  for(i=0;i<8;i++)
  {
    t=0.0;
    for(j=0;j<8;j++)
      t+=x[j]*KDCT[i][j];
    y[i]=t;
  }
  for(i=0;i<8;i++)
    x[i]=y[i];
}
void DCT2D(double * x)//routine to perform 8*8 point 2-D DCT
{
  int i,j;
  double y[8];
  //first row direction
  for(i=0;i<8;i++)
    DCT1D(x+8*i);
  //then column direction
  for(i=0;i<8;i++)
  {
    for(j=0;j<8;j++)
      y[j]=*(x+8*j+i);
    DCT1D(y);
    for(j=0;j<8;j++)
      *(x+8*j+i)=y[j];
  }
}
void IDCT1D(double * x)//routine to perform 8 point 1-D Inverse DCT
{
  int i,j;
  double y[8];//temp array
  double t;
  for(i=0;i<8;i++)
  {
    t=0.0;
    for(j=0;j<8;j++)
      t+=x[j]*KIDCT[i][j];
    y[i]=t;
  }
  for(i=0;i<8;i++)
    x[i]=y[i];
}
void IDCT2D(double * x)//routine to perform 8*8 point 2-D Inverse DCT
{
  int i,j;
  double y[8];
  //first row direction
  for(i=0;i<8;i++)
    IDCT1D(x+8*i);
  //then column direction
  for(i=0;i<8;i++)
  {
    for(j=0;j<8;j++)
      y[j]=*(x+8*j+i);
    IDCT1D(y);
    for(j=0;j<8;j++)
      *(x+8*j+i)=y[j];
  }
}
///////////////////////////////////////////////
struct Trans_2D
{
	double Trans[BlockSize1][BlockSize1];
};
struct ValueWeight
{
	double SumValue;
	double SumWeight;
};
struct Index
{
	int IndexH;
	int IndexW;
};
///////////////////////////////////////
double WSExt(double * x,int n,int N)
{
  int P=2*(N-1);
  if(n<0)
    while(n<0)
      n+=P;
  else
    if(n>=P)
      while(n>=P)
        n-=P;
  if(n>=N)
    return x[P-n];
  else
    return x[n];
}
double WAExt(double * x,int n,int N)
{
  int P=2*N;
  if(n<0)
    while(n<0)
      n+=P;
  else
    if(n>=P)
      while(n>=P)
        n-=P;
  if(n>=N)
    if(n==P-1)
      return 0;
    else
      return -x[P-2-n];
  else
    return x[n];
}
double HSExt(double * x,int n,int N)
{
  int P=2*N;
  if(n<0)
    while(n<0)
      n+=P;
  else
    if(n>=P)
      while(n>=P)
        n-=P;
  if(n>=N)
    return x[P-1-n];
  else
    return x[n];
}

void Bior15_D(double * x,int N)
{
  int i,j;
  double sum;
  double LP[10]={0.0166,-0.0166,-0.1215,0.1215,0.7071,0.7071,0.1215,-0.1215,-0.0166,0.0166};
  double HP[2]={-0.7071,0.7071};
  double * t1 = NULL;
  double * t2 = NULL;
  t1=(double *)calloc(N/2, sizeof(double));
  t2=(double *)calloc(N/2, sizeof(double));
  //First,Lowpass Filtering
  for(i=0;i<N/2;i++)
  {
    sum=0.0;
    for(j=0;j<10;j++)
      sum+=HSExt(x,2*i-4+j,N)*LP[j];
    t1[i]=sum;
    sum=0.0;
    for(j=0;j<2;j++)
      sum+=HSExt(x,2*i+j,N)*HP[j];
    t2[i]=sum;
  }
  for(i=0;i<N/2;i++)
  {
    x[i]=t1[i];
    x[i+N/2]=t2[i];
  }
  free(t1);free(t2);
}
void Bior15_R(double * x,int N)
{
  int i,j;
  double sum;
  double LP[2]={0.7071,0.7071};
  double HP[10]={0.0166,0.0166,-0.1215,-0.1215,0.7071,-0.7071,0.1215,0.1215,-0.0166,-0.0166};
  double *t1 = NULL;
  double *t2 = NULL;
  t1=(double *)calloc(N, sizeof(double));
  t2=(double *)calloc(N, sizeof(double));
  //First,Upsampling
  for(i=0;i<N/2;i++)
  {
    j=2*i;
    t1[j]=x[i];
    t2[j]=x[N/2+i];
  }
  //Then Filtering With Shift
  for(i=0;i<N;i++)
  {
    sum=0.0;
    for(j=0;j<2;j++)
      sum+=WSExt(t1,i-1+j,N)*LP[j];
    for(j=0;j<10;j++)
      sum+=WAExt(t2,i-5+j,N)*HP[j];
    x[i]=sum;
  }
  free(t1);free(t2);
}
void Bior15_1D(double * x,int N,int level)
{
  int i;
  for(i=0;i<level;i++)
  {
    Bior15_D(x,N);
    N/=2;
  }
}
void IBior15_1D(double * x,int N,int level)
{
  int i;
  N>>=(level-1);
  for(i=0;i<level;i++)
  {
    Bior15_R(x,N);
    N*=2;
  }
}
void Bior15_2D(double * x,int N,int level)//routine to perform 2D bior15
{
  int i,j;
  double y[8];
  //first row direction
  for(i=0;i<8;i++)
    Bior15_1D(x+8*i,N,level);
  //then column direction
  for(i=0;i<8;i++)
  {
    for(j=0;j<8;j++)
      y[j]=*(x+8*j+i);
    Bior15_1D((double *)y,N,level);
    for(j=0;j<8;j++)
      *(x+8*j+i)=y[j];
  }
}
void IBior15_2D(double * x,int N,int level)//routine to perform 2D bior15
{
  int i,j;
  double y[8];
  //first row direction
  for(i=0;i<8;i++)
    IBior15_1D(x+8*i,N,level);
  //then column direction
  for(i=0;i<8;i++)
  {
    for(j=0;j<8;j++)
      y[j]=*(x+8*j+i);
    IBior15_1D((double *)y,N,level);
    for(j=0;j<8;j++)
      *(x+8*j+i)=y[j];
  }
}
void incr(int *i)
{
	if(*i<(Height-BlockSize1-((Height-BlockSize1)%BlockStep1)))
		(*i)+=BlockStep1;
	else
		(*i)++;
}
///////////////////////////////////// 
void wavedcp_bior(double *position_biordcp,int datalenth_biordcp);
void waverecs_bior(double *position_biorrecs,int datalenth_biorrecs);
int BlockDetecet1(int input_h,int input_w);//判断某个坐标是否有对应的完整的block
double GauseNoise(double var_noise);
double psnr(double *start1_psnr,double *start2_psnr,int height_psnr,int width_psnr);
void dct_block(double *start_position_dct1,double *dct1_trans,int datalenth_dct1);
void idct_block(double *start_position_idct1,double *idct1_trans,int datalenth_idct1);
void Gidct_1D(double *start_position_idct,int datalenth_idct);
void Gdct_1D(double *start_position_dct,int datalenth_dct);
unsigned char Trans(double input_Trans);
double bessio(double x);
double KaiserW(int x,double beta,int N);
void Bior_block(double *Data,int dcp_level,int size);
void iBior_block(double *Data,int level,int size);
void haar_recs(double *dataihaar,int len_ihaar);
void haar_dcp(double *datahaar,int len_haar);
void creatDCTMatrix(double *position_DCTMatrix,double *position_InvertDCTMatrix,int size_DCTMatrix);
void main()
{
	static unsigned char Image[Height][Width],ImageBasicOutput[Height][Width],ImageFinalOutput[Height][Width],ImageNoisyOutput[Height][Width];
	static double ImageOrignal[Height][Width],ImageNoisy[Height][Width],ImageBasic[Height][Width],BlockTemp[BlockSize1][BlockSize1],
		          D[BlockMatch2],Kaiser[BlockSize1][BlockSize1],DataTemp[BlockSize1],Group[BlockMatch1][BlockSize1][BlockSize1],
				  Trans_1D[BlockMatch2],Column[BlockSize1],Row[BlockSize1],ImagePSNR[Height][Width],dctTrans[BlockSize1][BlockSize1],
				  idctTrans[BlockSize1][BlockSize1],GroupBasic[BlockMatch2][BlockSize2][BlockSize2],GroupNoisy[BlockMatch2][BlockSize2][BlockSize2],
				  Wiener[BlockMatch2][BlockSize2][BlockSize2],ImageFinal[Height][Width];
	static struct Trans_2D BlockTrans1[Height-BlockSize1+1][Width-BlockSize1+1],BlockTransBasic[Height-BlockSize2+1][Width-BlockSize2+1];
	static struct ValueWeight ImgaeVM[Height][Width];
	static struct Index Pos[BlockMatch2];
	int i,j,u,v,p,q,x,y,BlockCount,Nonzero;
	double Distance,WeightHt,WeightWie,PSNR,pi=3.1415926;
	FILE *fp1,*fp2,*fp3,*fp4;
	fopen_s(&fp1,"f:\\image\\BM3D\\cameraman256.raw","rb");
	//fopen_s(&fp1,"f:\\image\\BM3D\\montage256.raw","rb");
	fread(Image,1,Width*Height,fp1);
	srand((unsigned)time(NULL));//very important
	for(i=0;i<Height;i++)
		for(j=0;j<Width;j++)
		{
			ImageOrignal[i][j]=(double)Image[i][j];
			ImageNoisy[i][j]=ImageOrignal[i][j]+GauseNoise(NoiseStadard);
			ImageNoisyOutput[i][j]=Trans(ImageNoisy[i][j]);
			ImgaeVM[i][j].SumValue=ImgaeVM[i][j].SumWeight=ImageBasic[i][j]=0.0;
		}
	PSNR=psnr((double *)ImageOrignal,(double *)ImageNoisy,Height,Width);
	printf("PSNR befors Denoising is %f\n",PSNR);
	fopen_s(&fp2,"f:\\image\\BM3D\\house256Gause.dat","wb");
	fwrite(ImageNoisy,sizeof(double),Width*Height,fp2);
////////////////////////////////////////////////////////////////
	creatDCTMatrix((double *)dctTrans,(double *)idctTrans,BlockSize1);
//////////////////////////////////////////////////////////////////Kaiser窗
	for(u=0;u<BlockSize1;u++)
		for(v=0;v<BlockSize1;v++)
			Kaiser[u][v]=KaiserW(u,BetaHt,BlockSize1)*KaiserW(v,BetaHt,BlockSize1);
	////////////////////////////////////////////////////////////////////////////
	for(i=0;i<=(Height-BlockSize1);i++)
		for(j=0;j<=(Width-BlockSize1);j++)
		{
			for(u=0;u<BlockSize1;u++)
				for(v=0;v<BlockSize1;v++)
				{
					BlockTrans1[i][j].Trans[u][v]=0.0;
					BlockTemp[u][v]=ImageNoisy[i+u][j+v];
				}
			//dct_block((double *)BlockTemp,(double *)BiorMatrix,BlockSize1);
			Bior_block((double *)BlockTemp,3,BlockSize1);
			//Bior15_2D((double *)BlockTemp,BlockSize1,3);
			for(u=0;u<BlockSize1;u++)
				for(v=0;v<BlockSize1;v++)
					BlockTrans1[i][j].Trans[u][v]=BlockTemp[u][v];
			//printf("%d %d\n",i,j);
		}
		printf("translation finish\n");
		/////////////////////////////////////////////////////////////////////////////


	for(i=0;i<=(Height-BlockSize1);incr(&i))
	{
		for(j=0;j<=(Width-BlockSize1);incr(&j))
		{
			for(u=0;u<BlockSize1;u++)
				for(v=0;v<BlockSize1;v++)
					Group[0][u][v]=BlockTrans1[i][j].Trans[u][v];
			Pos[0].IndexH=i;
			Pos[0].IndexW=j;
			D[0]=0;
			for(u=1;u<BlockMatch1;u++)
			{
				Pos[u].IndexH=Pos[u].IndexW=0;
				D[u]=Threshold1;
			}
			BlockCount=1;
			for(u=-BlockSearch1;u<=BlockSearch1;u++)
				for(v=-BlockSearch1;v<=BlockSearch1;v++)
				{
					////////////////////构造group
					if(BlockDetecet1(i+u,j+v)==1&&(u!=0||v!=0))
					{
						Distance=0;
						for(x=0;x<BlockSize1;x++)
							for(y=0;y<BlockSize1;y++)
								Distance+=(BlockTrans1[i][j].Trans[x][y]-BlockTrans1[i+u][j+v].Trans[x][y])
								         *(BlockTrans1[i][j].Trans[x][y]-BlockTrans1[i+u][j+v].Trans[x][y]);
						Distance/=(double)(BlockSize1*BlockSize1);
						if(Distance<Threshold1)
						{
							if(BlockCount<BlockMatch1)
							{
								for(x=1;x<=BlockCount;x++)
								{
									if(Distance<D[x])
									{
										for(y=BlockCount;y>=x+1;y--)//从后往前
										{
											D[y]=D[y-1];
											Pos[y]=Pos[y-1];
											for(p=0;p<BlockSize1;p++)
												for(q=0;q<BlockSize1;q++)
													Group[y][p][q]=Group[y-1][p][q];
										}
										D[x]=Distance;
										Pos[x].IndexH=i+u;
										Pos[x].IndexW=j+v;
										for(p=0;p<BlockSize1;p++)
											for(q=0;q<BlockSize1;q++)
												Group[x][p][q]=BlockTrans1[i+u][j+v].Trans[p][q];
										BlockCount++;
										break;
									}
								}
							}
							else if(BlockCount==BlockMatch1)
							{
								if(Distance<D[BlockMatch1-1])
								{
									for(x=1;x<=(BlockMatch1-1);x++)
									{
										if(Distance<D[x])
										{
											for(y=(BlockMatch1-1);y>=x+1;y--)
											{
												D[y]=D[y-1];
											    Pos[y]=Pos[y-1];
											    for(p=0;p<BlockSize1;p++)
												    for(q=0;q<BlockSize1;q++)
													    Group[y][p][q]=Group[y-1][p][q];
											}
											D[x]=Distance;
										    Pos[x].IndexH=i+u;
										    Pos[x].IndexW=j+v;
											for(p=0;p<BlockSize1;p++)
											    for(q=0;q<BlockSize1;q++)
												    Group[x][p][q]=BlockTrans1[i+u][j+v].Trans[p][q];
											break;
										}
									}
								}
							}
						}
					}
				}
		/////////再进行一维变换，稀疏处理，一维重建
			Nonzero=0;
			BlockCount=(int)(log((double)BlockCount)/log(2.0));
			BlockCount=(int)pow(2.0,(double)BlockCount);
		///////////////////////////////////////////////////////
			for(u=0;u<BlockSize1;u++)
				for(v=0;v<BlockSize1;v++)
			{
				for(x=0;x<BlockCount;x++)
					Trans_1D[x]=Group[x][u][v];
				haar_dcp(Trans_1D,BlockCount);
				for(x=0;x<BlockCount;x++)
				{
					if(fabs(Trans_1D[x])<(Lemda1_3D*NoiseStadard))
						Trans_1D[x]=0.0;
					else
						Nonzero++;
				}
				haar_recs(Trans_1D,BlockCount);
				for(x=0;x<BlockCount;x++)
					Group[x][u][v]=Trans_1D[x];
			}
			if(Nonzero>0)
			WeightHt=1/NoiseStadard/NoiseStadard/Nonzero;
			else
				WeightHt=1;
		///////////////////////////进行二维重建并将数据写回到对应位置的像素
			for(x=0;x<BlockCount;x++)
			{
				for(u=0;u<BlockSize1;u++)
					for(v=0;v<BlockSize1;v++)
						BlockTemp[u][v]=Group[x][u][v];
			    //idct_block((double *)BlockTemp,(double *)idctTrans,BlockSize1);
				iBior_block((double *)BlockTemp,3,BlockSize1);
				//IBior15_2D((double *)BlockTemp,BlockSize1,3);
				for(u=0;u<BlockSize1;u++)
					for(v=0;v<BlockSize1;v++)
						Group[x][u][v]=BlockTemp[u][v];
			}
		///////////////////////////////////////////////////放回原位置，记录像素值和权值
			for(x=0;x<BlockCount;x++)
			{
				p=Pos[x].IndexH;
				q=Pos[x].IndexW;
				for(u=0;u<BlockSize1;u++)
					for(v=0;v<BlockSize1;v++)
					{
						ImgaeVM[p+u][q+v].SumValue+=WeightHt*Kaiser[u][v]*Group[x][u][v];
						ImgaeVM[p+u][q+v].SumWeight+=WeightHt*Kaiser[u][v];
					}
			}
		}//j
	}//i
	for(i=0;i<Height;i++)
		for(j=0;j<Width;j++)
		{
			if(ImgaeVM[i][j].SumWeight!=0){//u++;
			ImageBasic[i][j]=(ImgaeVM[i][j].SumValue)/(ImgaeVM[i][j].SumWeight);
			ImageBasicOutput[i][j]=Trans(ImageBasic[i][j]);
			ImagePSNR[i][j]=(double)ImageBasicOutput[i][j];}
		}
	////////////////////////////////
    PSNR=psnr((double *)ImageOrignal,(double *)ImagePSNR,Height,Width);
	/*PSNR=0;
	for(i=0;i<Height;i++)
		for(j=0;j<Width;j++)
			if(ImgaeVM[i][j].SumWeight!=0)
			PSNR+=(ImagePSNR[i][j]-ImageOrignal[i][j])*(ImagePSNR[i][j]-ImageOrignal[i][j]);
	PSNR=10*log10(255*255/(PSNR/u));*/
	printf("PSNR after Basic Estimate is %f\n",PSNR);
	fopen_s(&fp3,"f:\\image\\BM3D\\imageStudent1.dat","wb");
	fwrite(ImageBasic,sizeof(double),Width*Height,fp3);
    printf("Basic Estimate Finished\n");







	//////////////////以下为Final Estimate
	for(i=0;i<Height;i++)
		for(j=0;j<Width;j++)
	        ImgaeVM[i][j].SumValue=ImgaeVM[i][j].SumWeight=0.0;
	for(i=0;i<=(Height-BlockSize2);i++)
		for(j=0;j<=(Width-BlockSize2);j++)
		{
			for(u=0;u<BlockSize2;u++)
				for(v=0;v<BlockSize2;v++)
				{
					BlockTransBasic[i][j].Trans[u][v]=0.0;
					BlockTemp[u][v]=ImageBasic[i+u][j+v];
				}
			dct_block((double *)BlockTemp,(double *)dctTrans,BlockSize2);
			for(u=0;u<BlockSize2;u++)
				for(v=0;v<BlockSize2;v++)
					BlockTransBasic[i][j].Trans[u][v]=BlockTemp[u][v];
		}
	for(i=0;i<=(Height-BlockSize2);i++)
		for(j=0;j<=(Width-BlockSize2);j++)
		{
			for(u=0;u<BlockSize2;u++)
				for(v=0;v<BlockSize2;v++)
				{
					BlockTrans1[i][j].Trans[u][v]=0.0;
					BlockTemp[u][v]=ImageNoisy[i+u][j+v];
				}
			dct_block((double *)BlockTemp,(double *)dctTrans,BlockSize2);
			for(u=0;u<BlockSize1;u++)
				for(v=0;v<BlockSize1;v++)
					BlockTrans1[i][j].Trans[u][v]=BlockTemp[u][v];
		}
		printf("translation basic finished\n");
	for(i=0;i<=(Height-BlockSize2);incr(&i))
	{
		for(j=0;j<=(Width-BlockSize2);incr(&j))
		{
			for(u=0;u<BlockSize2;u++)
				for(v=0;v<BlockSize2;v++)
					GroupBasic[0][u][v]=BlockTransBasic[i][j].Trans[u][v];
			Pos[0].IndexH=i;
			Pos[0].IndexW=j;
			D[0]=0;
			for(u=1;u<BlockMatch2;u++)
			{
				Pos[u].IndexH=Pos[u].IndexW=0;
				D[u]=Threshold2;
			}
			BlockCount=1;
			////////////////////////////////////////////
			for(u=-BlockSearch2;u<=BlockSearch2;u++)
				for(v=-BlockSearch2;v<=BlockSearch2;v++)
				{
					if(BlockDetecet1(i+u,j+v)==1&&(u!=0||v!=0))
					{
						Distance=0.0;
						for(x=0;x<BlockSize2;x++)
							for(y=0;y<BlockSize2;y++)
								Distance+=(ImageBasic[i+x][j+y]-ImageBasic[i+u+x][j+v+y])*(ImageBasic[i+x][j+y]-ImageBasic[i+u+x][j+v+y]);
						Distance/=(double)(BlockSize2*BlockSize2);
						if(Distance<Threshold2)
						{
							if(BlockCount<BlockMatch2)
							{
								for(x=1;x<=BlockCount;x++)
								{
									if(Distance<D[x])
									{
										for(y=BlockCount;y>=x+1;y--)//从后往前
										{
											D[y]=D[y-1];
											Pos[y]=Pos[y-1];
											for(p=0;p<BlockSize2;p++)
												for(q=0;q<BlockSize2;q++)
													GroupBasic[y][p][q]=GroupBasic[y-1][p][q];
										}
										D[x]=Distance;
										Pos[x].IndexH=i+u;
										Pos[x].IndexW=j+v;
										for(p=0;p<BlockSize2;p++)
											for(q=0;q<BlockSize2;q++)
												GroupBasic[x][p][q]=BlockTransBasic[i+u][j+v].Trans[p][q];
										BlockCount++;
										break;
									}
								}
							}
							else if(BlockCount==BlockMatch2)
							{
								if(Distance<D[BlockMatch2-1])
								{
									for(x=1;x<=(BlockMatch2-1);x++)
									{
										if(Distance<D[x])
										{
											for(y=(BlockMatch2-1);y>=x+1;y--)
											{
												D[y]=D[y-1];
											    Pos[y]=Pos[y-1];
											    for(p=0;p<BlockSize2;p++)
												    for(q=0;q<BlockSize2;q++)
													    GroupBasic[y][p][q]=GroupBasic[y-1][p][q];
											}
											D[x]=Distance;
										    Pos[x].IndexH=i+u;
										    Pos[x].IndexW=j+v;
											for(p=0;p<BlockSize2;p++)
											    for(q=0;q<BlockSize2;q++)
												    GroupBasic[x][p][q]=BlockTransBasic[i+u][j+v].Trans[p][q];
											break;
										}
									}
								}
							}
						}
					}
				}
		BlockCount=(int)(log((double)BlockCount)/log(2.0));
		BlockCount=(int)pow(2.0,(double)BlockCount);
		////////////////////////////////////
			for(x=0;x<BlockCount;x++)
			{
				p=Pos[x].IndexH;
				q=Pos[x].IndexW;
				for(u=0;u<BlockSize2;u++)
				    for(v=0;v<BlockSize2;v++)
						GroupNoisy[x][u][v]=BlockTrans1[p][q].Trans[u][v];
			}
		//////////////////////////////////////////一维变换
			for(u=0;u<BlockSize2;u++)
				for(v=0;v<BlockSize2;v++)
				{
					for(x=0;x<BlockCount;x++)
					    Trans_1D[x]=GroupBasic[x][u][v];
				    haar_dcp((double *)Trans_1D,BlockCount);
				    for(x=0;x<BlockCount;x++)
					    GroupBasic[x][u][v]=Trans_1D[x];
				}
			for(u=0;u<BlockSize2;u++)
				for(v=0;v<BlockSize2;v++)
				{
					for(x=0;x<BlockCount;x++)
					    Trans_1D[x]=GroupNoisy[x][u][v];
				   haar_dcp((double *)Trans_1D,BlockCount);
				   for(x=0;x<BlockCount;x++)
					    GroupNoisy[x][u][v]=Trans_1D[x];
				}
		/////////////////////////////////////////////////////////////////////////////////
			WeightWie=0.0;
			for(x=0;x<BlockCount;x++)
				for(u=0;u<BlockSize2;u++)
				    for(v=0;v<BlockSize2;v++)
					{
						Wiener[x][u][v]=(fabs(GroupBasic[x][u][v])*fabs(GroupBasic[x][u][v]))
						               /(fabs(GroupBasic[x][u][v])*fabs(GroupBasic[x][u][v])+NoiseStadard*NoiseStadard);
						 GroupNoisy[x][u][v]*=Wiener[x][u][v];
						 WeightWie+=(Wiener[x][u][v]*Wiener[x][u][v]);
					}
			WeightWie=1/(NoiseStadard*NoiseStadard)/WeightWie;
			///////////////////////一维反变换
			for(u=0;u<BlockSize2;u++)
				for(v=0;v<BlockSize2;v++)
				{
					for(x=0;x<BlockCount;x++)
					    Trans_1D[x]=GroupNoisy[x][u][v];
					haar_recs((double *)Trans_1D,BlockCount);
				    for(x=0;x<BlockCount;x++)
					    GroupNoisy[x][u][v]=Trans_1D[x];
				}
			///////////////////////二维反变换,放回原处
			for(x=0;x<BlockCount;x++)
			{
				for(u=0;u<BlockSize2;u++)
					for(v=0;v<BlockSize2;v++)
						BlockTemp[u][v]=GroupNoisy[x][u][v];
				idct_block((double *)BlockTemp,(double *)idctTrans,BlockSize2);
				for(u=0;u<BlockSize2;u++)
					for(v=0;v<BlockSize2;v++)
						GroupNoisy[x][u][v]=BlockTemp[u][v];
			}
			for(x=0;x<BlockCount;x++)
			{
				p=Pos[x].IndexH;
				q=Pos[x].IndexW;
				for(u=0;u<BlockSize2;u++)
					for(v=0;v<BlockSize2;v++)
					{
						ImgaeVM[p+u][q+v].SumValue+=WeightWie*Kaiser[u][v]*GroupNoisy[x][u][v];
						ImgaeVM[p+u][q+v].SumWeight+=WeightWie*Kaiser[u][v];
					}
			}
		////////////////////////////////////////////////////////////////
		}
	}
	for(i=0;i<Height;i++)
		for(j=0;j<Width;j++)
		{
			if(ImgaeVM[i][j].SumWeight!=0)
			{
			ImageFinal[i][j]=(ImgaeVM[i][j].SumValue)/(ImgaeVM[i][j].SumWeight);
			ImageFinalOutput[i][j]=Trans(ImageFinal[i][j]);
			ImagePSNR[i][j]=(double)ImageFinalOutput[i][j];
			}
		}
	PSNR=psnr((double *)ImageOrignal,(double *)ImagePSNR,Height,Width);
	/*PSNR=0;
	for(i=0;i<Height-2;i++)
		for(j=0;j<Width-2;j++)
			if(ImgaeVM[i][j].SumWeight!=0)
			PSNR+=(ImagePSNR[i][j]-ImageOrignal[i][j])*(ImagePSNR[i][j]-ImageOrignal[i][j]);
	PSNR=10*log10(255*255/(PSNR/u));*/
	printf("PSNR after Final Estimate is %f\n",PSNR);
	fopen_s(&fp4,"f:\\image\\BM3D\\imageStudent2.dat","wb");
	fwrite(ImageFinal,sizeof(double),Width*Height,fp4);
    printf("Final Estimate Finished\n");
	getchar();
}
double GauseNoise(double var_noise)
{  
   int k;
   double sum=0.0;
   for(k=0;k<12;k++) 
	   sum+=((double)rand()/(double)RAND_MAX-0.5);
   return var_noise*sum;
}
int BlockDetecet1(int input_h,int input_w)
{
	int output_BD1;
	if(input_h>(Height-BlockSize1)||input_w>(Width-BlockSize1)||input_h<0||input_w<0)
		output_BD1=0;
	else
		output_BD1=1;
	return 
		output_BD1;
}
double psnr(double *start1_psnr,double *start2_psnr,int height_psnr,int width_psnr)
{
	int psnr_i,psnr_j;
	double sum_psnr=0.0,output_psnr;
	for(psnr_i=0;psnr_i<height_psnr;psnr_i++)
		for(psnr_j=0;psnr_j<width_psnr;psnr_j++)
			sum_psnr+=((*(start1_psnr+psnr_i*width_psnr+psnr_j))-(*(start2_psnr+psnr_i*width_psnr+psnr_j)))*((*(start1_psnr+psnr_i*width_psnr+psnr_j))-(*(start2_psnr+psnr_i*width_psnr+psnr_j)));
	output_psnr=10*(2*log10(255.0)+log10((double)height_psnr*width_psnr)-log10(sum_psnr));
	return output_psnr;
}
void Gdct_1D(double *start_position_dct,int datalenth_dct)
{
	int i_dct,j_dct;
	double pi=3.1415926,*trans_dct;
	trans_dct=(double *)malloc(sizeof(double)*datalenth_dct);
	for(i_dct=0;i_dct<datalenth_dct;i_dct++)
		*(trans_dct+i_dct)=0;
	for(i_dct=0;i_dct<datalenth_dct;i_dct++)
		for(j_dct=0;j_dct<datalenth_dct;j_dct++)
			*(trans_dct+i_dct)+=(*(start_position_dct+j_dct))*cos((j_dct+0.5)*i_dct*pi/datalenth_dct);
	*(start_position_dct)=*(trans_dct)/sqrt((double)datalenth_dct);
	for(i_dct=1;i_dct<datalenth_dct;i_dct++)
		(*(start_position_dct+i_dct))=*(trans_dct+i_dct)/sqrt((double)datalenth_dct/2.0);
	free(trans_dct);
}
void Gidct_1D(double *start_position_idct,int datalenth_idct)
{
	int i_idct,j_idct;
	double pi=3.1415926,*trans_idct;
	trans_idct=(double *)malloc(sizeof(double)*datalenth_idct);
	for(i_idct=0;i_idct<datalenth_idct;i_idct++)
		*(trans_idct+i_idct)=0;
	for(i_idct=0;i_idct<datalenth_idct;i_idct++)
		for(j_idct=0;j_idct<datalenth_idct;j_idct++)
		{
			if(j_idct==0)
				*(trans_idct+i_idct)+=(*(start_position_idct))/sqrt((double)datalenth_idct);
			else
			    *(trans_idct+i_idct)+=(*(start_position_idct+j_idct))*cos((i_idct+0.5)*j_idct*pi/datalenth_idct)/sqrt((double)datalenth_idct/2.0);
		}
	for(i_idct=0;i_idct<datalenth_idct;i_idct++)
		(*(start_position_idct+i_idct))=*(trans_idct+i_idct);
	free(trans_idct);
}
void dct_block(double *DataDCT_block,double *MatrixDCT_block,int SizeDCT_block)
{
	int i_DCTblock,j_DCTblock,k_DCTblock;
	double *TempDCTblock;
	TempDCTblock=(double *)malloc(sizeof(double)*SizeDCT_block);
	for(i_DCTblock=0;i_DCTblock<SizeDCT_block;i_DCTblock++)
	{
		for(j_DCTblock=0;j_DCTblock<SizeDCT_block;j_DCTblock++)
			*(TempDCTblock+j_DCTblock)=0.0;
		for(j_DCTblock=0;j_DCTblock<SizeDCT_block;j_DCTblock++)
			for(k_DCTblock=0;k_DCTblock<SizeDCT_block;k_DCTblock++)
				*(TempDCTblock+j_DCTblock)+=*(MatrixDCT_block+j_DCTblock*SizeDCT_block+k_DCTblock)*(*(DataDCT_block+i_DCTblock*SizeDCT_block+k_DCTblock));
		for(j_DCTblock=0;j_DCTblock<SizeDCT_block;j_DCTblock++)
			*(DataDCT_block+i_DCTblock*SizeDCT_block+j_DCTblock)=*(TempDCTblock+j_DCTblock);
	}
	for(j_DCTblock=0;j_DCTblock<SizeDCT_block;j_DCTblock++)
	{
		for(i_DCTblock=0;i_DCTblock<SizeDCT_block;i_DCTblock++)
			*(TempDCTblock+i_DCTblock)=0.0;
		for(i_DCTblock=0;i_DCTblock<SizeDCT_block;i_DCTblock++)
			for(k_DCTblock=0;k_DCTblock<SizeDCT_block;k_DCTblock++)
				*(TempDCTblock+i_DCTblock)+=*(MatrixDCT_block+i_DCTblock*SizeDCT_block+k_DCTblock)*(*(DataDCT_block+k_DCTblock*SizeDCT_block+j_DCTblock));
		for(i_DCTblock=0;i_DCTblock<SizeDCT_block;i_DCTblock++)
			*(DataDCT_block+i_DCTblock*SizeDCT_block+j_DCTblock)=*(TempDCTblock+i_DCTblock);
	}
	free(TempDCTblock);
}
void idct_block(double *DataIDCT_block,double *MatrixIDCT_block,int SizeIDCT_block)
{
	int i_IDCTblock,j_IDCTblock,k_IDCTblock;
	double *TempIDCTblock;
	TempIDCTblock=(double *)malloc(sizeof(double)*SizeIDCT_block);
	for(j_IDCTblock=0;j_IDCTblock<SizeIDCT_block;j_IDCTblock++)
	{
		for(i_IDCTblock=0;i_IDCTblock<SizeIDCT_block;i_IDCTblock++)
			*(TempIDCTblock+i_IDCTblock)=0.0;
		for(i_IDCTblock=0;i_IDCTblock<SizeIDCT_block;i_IDCTblock++)
			for(k_IDCTblock=0;k_IDCTblock<SizeIDCT_block;k_IDCTblock++)
				*(TempIDCTblock+i_IDCTblock)+=*(MatrixIDCT_block+i_IDCTblock*SizeIDCT_block+k_IDCTblock)
				                              *(*(DataIDCT_block+k_IDCTblock*SizeIDCT_block+j_IDCTblock));
		for(i_IDCTblock=0;i_IDCTblock<SizeIDCT_block;i_IDCTblock++)
			*(DataIDCT_block+i_IDCTblock*SizeIDCT_block+j_IDCTblock)=*(TempIDCTblock+i_IDCTblock);
	}
	for(i_IDCTblock=0;i_IDCTblock<SizeIDCT_block;i_IDCTblock++)
	{
		for(j_IDCTblock=0;j_IDCTblock<SizeIDCT_block;j_IDCTblock++)
			*(TempIDCTblock+j_IDCTblock)=0.0;
		for(j_IDCTblock=0;j_IDCTblock<SizeIDCT_block;j_IDCTblock++)
			for(k_IDCTblock=0;k_IDCTblock<SizeIDCT_block;k_IDCTblock++)
				*(TempIDCTblock+j_IDCTblock)+=*(MatrixIDCT_block+j_IDCTblock*SizeIDCT_block+k_IDCTblock)
				                              *(*(DataIDCT_block+i_IDCTblock*SizeIDCT_block+k_IDCTblock));
		for(j_IDCTblock=0;j_IDCTblock<SizeIDCT_block;j_IDCTblock++)
			*(DataIDCT_block+i_IDCTblock*SizeIDCT_block+j_IDCTblock)=*(TempIDCTblock+j_IDCTblock);
	}
	free(TempIDCTblock);
}
void Bior_block(double *Data,int dcp_level,int size)
{
	int i,j,k,len;
	double *Temp;
	Temp=(double *)malloc(sizeof(double)*size);
	for(k=0;k<dcp_level;k++)
	{
		len=(int)(size/pow(2.0,(double)k));
		for(i=0;i<len;i++)
		{
			for(j=0;j<len;j++)
				*(Temp+j)=*(Data+i*size+j);
			wavedcp_bior(Temp,len);
			for(j=0;j<len;j++)
				*(Data+i*size+j)=*(Temp+j);
		}
		for(j=0;j<len;j++)
		{
			for(i=0;i<len;i++)
				*(Temp+i)=*(Data+i*size+j);
			wavedcp_bior(Temp,len);
			for(i=0;i<len;i++)
				*(Data+i*size+j)=*(Temp+i);
		}
	}
	free(Temp);
}
void iBior_block(double *Data,int level,int size)
{
	int i,j,k,len;
	double *Temp;
	Temp=(double *)malloc(sizeof(double)*size);
	for(k=0;k<level;k++)
	{
		len=(int)(2*pow(2.0,(double)k));
		for(j=0;j<len;j++)
		{
			for(i=0;i<len;i++)
				*(Temp+i)=*(Data+i*size+j);
			waverecs_bior(Temp,len);
			for(i=0;i<len;i++)
				*(Data+i*size+j)=*(Temp+i);
		}
		for(i=0;i<len;i++)
		{
			for(j=0;j<len;j++)
				*(Temp+j)=*(Data+i*size+j);
			waverecs_bior(Temp,len);
			for(j=0;j<len;j++)
				*(Data+i*size+j)=*(Temp+j);
		}
	}
	free(Temp);
}
unsigned char Trans(double input_Trans)
{ 
  unsigned char r;
  if(input_Trans<0) r=0;
  else if(input_Trans>255) r=255;
  else r=(unsigned char)(input_Trans+0.5);
  return r;
}
double bessio(double x)
{
  double ax,ans,y;
  if((ax=fabs(x))<3.75)
  {
    y=x/3.75;
    y*=y;
    ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492+y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
  }
  else
  {
    y=3.75/x;
    ans=(exp(ax)/sqrt(ax))*(0.39894228+y*(0.1328592e-1+y*(0.225319e-2+y*(-0.157565e-2+y*(0.91628e-2+y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1+y*0.392377e-2))))))));
  }
  return ans;
}

double KaiserW(int x,double beta,int N)
{
  double s,t;
  s=(double)1-2*(double)x/((double)N-1);
  s*=s;
  t=beta*sqrt(1-s);
  return bessio(t)/bessio(beta);
}
 
void haar_dcp(double *datahaar,int len_haar)
{
	double *temp_haar;
	temp_haar=(double *)malloc(sizeof(double)*len_haar);
	int i_haar,len_temphaar;
	len_temphaar=len_haar;
	while(len_temphaar!=1)
	{
		if((len_temphaar%2)!=0)//奇数长数据
		{
			for(i_haar=0;i_haar<len_temphaar/2;i_haar++)
			{
				*(temp_haar+i_haar)=(*(datahaar+2*i_haar)+*(datahaar+2*i_haar+1))/sqrt(2.0);
				*(temp_haar+i_haar+len_temphaar/2+1)=(*(datahaar+2*i_haar)-*(datahaar+2*i_haar+1))/sqrt(2.0);
			}
			*(temp_haar+len_temphaar/2)=*(datahaar+len_temphaar-1);
			for(i_haar=0;i_haar<len_temphaar;i_haar++)
				*(datahaar+i_haar)=*(temp_haar+i_haar);
			len_temphaar=len_temphaar/2+1;
		}
		else//偶数长数据
		{
			for(i_haar=0;i_haar<len_temphaar/2;i_haar++)
			{
				*(temp_haar+i_haar)=(*(datahaar+2*i_haar)+*(datahaar+2*i_haar+1))/sqrt(2.0);
				*(temp_haar+i_haar+len_temphaar/2)=(*(datahaar+2*i_haar)-*(datahaar+2*i_haar+1))/sqrt(2.0);
			}
			for(i_haar=0;i_haar<len_temphaar;i_haar++)
				*(datahaar+i_haar)=*(temp_haar+i_haar);
			len_temphaar=len_temphaar/2;
		}
	}
	free(temp_haar);
}
void haar_recs(double *dataihaar,int len_ihaar)
{
	int i,j,k,level,*count_layer;
	double *temp;
	temp=(double *)malloc(sizeof(double)*len_ihaar);
	level=0;
	i=len_ihaar;
	while(i!=1)
	{
		if(i%2==0)
		  i/=2;
		else
		  i=i/2+1;
		level++;
	}
	count_layer=(int *)malloc(sizeof(int)*(level+1));
	*(count_layer+level)=len_ihaar;
	i=len_ihaar;
	for(j=level-1;j>=0;j--)
	{
		if(i%2==0)
		  i/=2;
		else
		  i=i/2+1;
		*(count_layer+j)=i;
	}
	for(i=0;i<level;i++)
	{
	    if(*(count_layer+i)!=(*(count_layer+i+1)-*(count_layer+i)))
		{
			j=*(count_layer+i);//low
			for(k=0;k<=j-2;k++)
			{
				*(temp+2*k)=(*(dataihaar+k)+*(dataihaar+k+j))/sqrt(2.0);
				*(temp+2*k+1)=(*(dataihaar+k)-*(dataihaar+k+j))/sqrt(2.0);
			}
			*(temp+2*j-2)=*(dataihaar+j-1);
			for(k=0;k<=2*j-2;k++)
				*(dataihaar+k)=*(temp+k);
		}
		else
		{
			j=*(count_layer+i);
			for(k=0;k<=j-1;k++)
			{
				*(temp+2*k)=(*(dataihaar+k)+*(dataihaar+k+j))/sqrt(2.0);
				*(temp+2*k+1)=(*(dataihaar+k)-*(dataihaar+k+j))/sqrt(2.0);
			}
			for(k=0;k<=2*j-1;k++)
				*(dataihaar+k)=*(temp+k);
		}
	}
	free(temp);
	free(count_layer);
}
void creatDCTMatrix(double *position_DCTMatrix,double *position_InvertDCTMatrix,int size_DCTMatrix)
{
	int i_DCTMatrix,j_DCTMatrix;
	double *temp1_DCTMatrix,pi=3.1415926;
	temp1_DCTMatrix=(double *)malloc(sizeof(double)*size_DCTMatrix);
	for(i_DCTMatrix=0;i_DCTMatrix<size_DCTMatrix;i_DCTMatrix++)
		for(j_DCTMatrix=0;j_DCTMatrix<size_DCTMatrix;j_DCTMatrix++)
		{
			if(i_DCTMatrix==0)
			    *(position_DCTMatrix+i_DCTMatrix*size_DCTMatrix+j_DCTMatrix)=cos((2*j_DCTMatrix+1)*i_DCTMatrix*pi/size_DCTMatrix/2)/sqrt((double)size_DCTMatrix);
			else
				 *(position_DCTMatrix+i_DCTMatrix*size_DCTMatrix+j_DCTMatrix)=cos((2*j_DCTMatrix+1)*i_DCTMatrix*pi/size_DCTMatrix/2)/sqrt((double)size_DCTMatrix)*sqrt(2.0);
		}
	for(i_DCTMatrix=0;i_DCTMatrix<size_DCTMatrix;i_DCTMatrix++)
		for(j_DCTMatrix=0;j_DCTMatrix<size_DCTMatrix;j_DCTMatrix++)
			*(position_InvertDCTMatrix+j_DCTMatrix*size_DCTMatrix+i_DCTMatrix)=*(position_DCTMatrix+i_DCTMatrix*size_DCTMatrix+j_DCTMatrix);
}
void wavedcp_bior(double *position_biordcp,int datalenth_biordcp)
{
	int biordcp_i,biordcp_u,biordcp_temp;
	double *output_biordcp,biorLo_D[fillen_bior]={0.0166,-0.0166,-0.1215,0.1215,0.7071,0.7071,0.1215,-0.1215,-0.0166,0.0166},
           biorHi_D[fillen_bior]={0,0,0,0,-0.7071,0.7071,0,0,0,0}; 
	output_biordcp=(double *)malloc(sizeof(double)*datalenth_biordcp);
	for(biordcp_i=0;biordcp_i<datalenth_biordcp;biordcp_i++)
		*(output_biordcp+biordcp_i)=0.0;
	for(biordcp_u=0;biordcp_u<fillen_bior;biordcp_u++)
		for(biordcp_i=0;biordcp_i<datalenth_biordcp;biordcp_i++)
		{
			biordcp_temp=(datalenth_biordcp-fillen_bior/2+biordcp_u+biordcp_i)%datalenth_biordcp;
			if((biordcp_temp%2)==0)
			{
				*(output_biordcp+biordcp_temp/2)+=(*(position_biordcp+biordcp_i))*biorLo_D[biordcp_u];
				*(output_biordcp+biordcp_temp/2+(datalenth_biordcp/2))+=(*(position_biordcp+biordcp_i))*biorHi_D[biordcp_u];
			}
		}
	for(biordcp_i=0;biordcp_i<datalenth_biordcp;biordcp_i++)
		*(position_biordcp+biordcp_i)=*(output_biordcp+biordcp_i);
		free(output_biordcp);
}
void waverecs_bior(double *position_biorrecs,int datalenth_biorrecs)
{
	int biorrecs_i,biorrecs_u,biorrecs_temp;
	double *output_biorrecs,biorLo_R[fillen_bior]={0,0,0,0,0.7071,0.7071,0,0,0,0},
           biorHi_R[fillen_bior]={0.0166,0.0166,-0.1215,-0.1215,0.7071,-0.7071,0.1215,0.1215,-0.0166,-0.0166};
	output_biorrecs=(double *)malloc(sizeof(double)*datalenth_biorrecs);
	for(biorrecs_i=0;biorrecs_i<datalenth_biorrecs;biorrecs_i++)
		*(output_biorrecs+biorrecs_i)=0.0;
	for(biorrecs_u=0;biorrecs_u<fillen_bior;biorrecs_u++)
		for(biorrecs_i=0;biorrecs_i<datalenth_biorrecs;biorrecs_i++)
		{
			if(biorrecs_i<datalenth_biorrecs/2)
			{
				biorrecs_temp=(datalenth_biorrecs-fillen_bior/2+2*biorrecs_i+biorrecs_u)%datalenth_biorrecs;
				*(output_biorrecs+biorrecs_temp)+=(*(position_biorrecs+biorrecs_i))*biorLo_R[biorrecs_u];
			}
			else if(biorrecs_i>=datalenth_biorrecs/2)
			{
				biorrecs_temp=(2*biorrecs_i-fillen_bior/2+biorrecs_u)%datalenth_biorrecs;
				*(output_biorrecs+biorrecs_temp)+=(*(position_biorrecs+biorrecs_i))*biorHi_R[biorrecs_u];
			}
		}
	(*(position_biorrecs))=*(output_biorrecs+datalenth_biorrecs-1);
	for(biorrecs_i=1;biorrecs_i<datalenth_biorrecs;biorrecs_i++)
		(*(position_biorrecs+biorrecs_i))=*(output_biorrecs+biorrecs_i-1);
	free(output_biorrecs);
}