/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) quadrotor_expl_vde_forw_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[13] = {9, 1, 0, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8};
static const casadi_int casadi_s1[93] = {9, 9, 0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8};
static const casadi_int casadi_s2[33] = {9, 3, 0, 9, 18, 27, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8};
static const casadi_int casadi_s3[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s4[3] = {0, 0, 0};

/* quadrotor_expl_vde_forw:(i0[9],i1[9x9],i2[9x3],i3[3],i4[])->(o0[9],o1[9x9],o2[9x3]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][3] : 0;
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[0]? arg[0][4] : 0;
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[0]? arg[0][5] : 0;
  if (res[0]!=0) res[0][2]=a0;
  a0=arg[0]? arg[0][6] : 0;
  a1=cos(a0);
  a2=arg[0]? arg[0][8] : 0;
  a3=cos(a2);
  a4=(a1*a3);
  a5=arg[0]? arg[0][7] : 0;
  a6=sin(a5);
  a7=(a4*a6);
  a8=sin(a0);
  a9=sin(a2);
  a10=(a8*a9);
  a7=(a7+a10);
  a10=arg[3]? arg[3][2] : 0;
  a11=(a7*a10);
  if (res[0]!=0) res[0][3]=a11;
  a11=cos(a0);
  a12=sin(a5);
  a13=(a11*a12);
  a14=sin(a2);
  a15=(a13*a14);
  a16=cos(a2);
  a17=sin(a0);
  a18=(a16*a17);
  a15=(a15-a18);
  a18=(a15*a10);
  if (res[0]!=0) res[0][4]=a18;
  a18=-9.8100000000000005e+00;
  a19=cos(a5);
  a20=cos(a0);
  a21=(a19*a20);
  a22=(a21*a10);
  a18=(a18+a22);
  if (res[0]!=0) res[0][5]=a18;
  a18=1.4770000000000001e+00;
  a22=arg[3]? arg[3][0] : 0;
  a22=(a18*a22);
  a22=(a22-a0);
  a23=4.7699999999999998e-01;
  a22=(a22/a23);
  if (res[0]!=0) res[0][6]=a22;
  a22=arg[3]? arg[3][1] : 0;
  a18=(a18*a22);
  a18=(a18-a5);
  a18=(a18/a23);
  if (res[0]!=0) res[0][7]=a18;
  a18=0.;
  if (res[0]!=0) res[0][8]=a18;
  a23=arg[1]? arg[1][3] : 0;
  if (res[1]!=0) res[1][0]=a23;
  a23=arg[1]? arg[1][4] : 0;
  if (res[1]!=0) res[1][1]=a23;
  a23=arg[1]? arg[1][5] : 0;
  if (res[1]!=0) res[1][2]=a23;
  a23=cos(a5);
  a22=arg[1]? arg[1][7] : 0;
  a24=(a23*a22);
  a24=(a4*a24);
  a25=sin(a0);
  a26=arg[1]? arg[1][6] : 0;
  a27=(a25*a26);
  a27=(a3*a27);
  a28=sin(a2);
  a29=arg[1]? arg[1][8] : 0;
  a30=(a28*a29);
  a30=(a1*a30);
  a27=(a27+a30);
  a27=(a6*a27);
  a24=(a24-a27);
  a27=cos(a0);
  a30=(a27*a26);
  a30=(a9*a30);
  a31=cos(a2);
  a32=(a31*a29);
  a32=(a8*a32);
  a30=(a30+a32);
  a24=(a24+a30);
  a24=(a10*a24);
  if (res[1]!=0) res[1][3]=a24;
  a24=cos(a5);
  a30=(a24*a22);
  a30=(a11*a30);
  a32=sin(a0);
  a33=(a32*a26);
  a33=(a12*a33);
  a30=(a30-a33);
  a30=(a14*a30);
  a33=cos(a2);
  a34=(a33*a29);
  a34=(a13*a34);
  a30=(a30+a34);
  a34=cos(a0);
  a35=(a34*a26);
  a35=(a16*a35);
  a36=sin(a2);
  a29=(a36*a29);
  a29=(a17*a29);
  a35=(a35-a29);
  a30=(a30-a35);
  a30=(a10*a30);
  if (res[1]!=0) res[1][4]=a30;
  a30=sin(a5);
  a35=(a30*a22);
  a35=(a20*a35);
  a29=sin(a0);
  a37=(a29*a26);
  a37=(a19*a37);
  a35=(a35+a37);
  a35=(a10*a35);
  a35=(-a35);
  if (res[1]!=0) res[1][5]=a35;
  a35=2.0964360587002098e+00;
  a26=(a35*a26);
  a26=(-a26);
  if (res[1]!=0) res[1][6]=a26;
  a22=(a35*a22);
  a22=(-a22);
  if (res[1]!=0) res[1][7]=a22;
  if (res[1]!=0) res[1][8]=a18;
  a22=arg[1]? arg[1][12] : 0;
  if (res[1]!=0) res[1][9]=a22;
  a22=arg[1]? arg[1][13] : 0;
  if (res[1]!=0) res[1][10]=a22;
  a22=arg[1]? arg[1][14] : 0;
  if (res[1]!=0) res[1][11]=a22;
  a22=arg[1]? arg[1][16] : 0;
  a26=(a23*a22);
  a26=(a4*a26);
  a37=arg[1]? arg[1][15] : 0;
  a38=(a25*a37);
  a38=(a3*a38);
  a39=arg[1]? arg[1][17] : 0;
  a40=(a28*a39);
  a40=(a1*a40);
  a38=(a38+a40);
  a38=(a6*a38);
  a26=(a26-a38);
  a38=(a27*a37);
  a38=(a9*a38);
  a40=(a31*a39);
  a40=(a8*a40);
  a38=(a38+a40);
  a26=(a26+a38);
  a26=(a10*a26);
  if (res[1]!=0) res[1][12]=a26;
  a26=(a24*a22);
  a26=(a11*a26);
  a38=(a32*a37);
  a38=(a12*a38);
  a26=(a26-a38);
  a26=(a14*a26);
  a38=(a33*a39);
  a38=(a13*a38);
  a26=(a26+a38);
  a38=(a34*a37);
  a38=(a16*a38);
  a39=(a36*a39);
  a39=(a17*a39);
  a38=(a38-a39);
  a26=(a26-a38);
  a26=(a10*a26);
  if (res[1]!=0) res[1][13]=a26;
  a26=(a30*a22);
  a26=(a20*a26);
  a38=(a29*a37);
  a38=(a19*a38);
  a26=(a26+a38);
  a26=(a10*a26);
  a26=(-a26);
  if (res[1]!=0) res[1][14]=a26;
  a37=(a35*a37);
  a37=(-a37);
  if (res[1]!=0) res[1][15]=a37;
  a22=(a35*a22);
  a22=(-a22);
  if (res[1]!=0) res[1][16]=a22;
  if (res[1]!=0) res[1][17]=a18;
  a22=arg[1]? arg[1][21] : 0;
  if (res[1]!=0) res[1][18]=a22;
  a22=arg[1]? arg[1][22] : 0;
  if (res[1]!=0) res[1][19]=a22;
  a22=arg[1]? arg[1][23] : 0;
  if (res[1]!=0) res[1][20]=a22;
  a22=arg[1]? arg[1][25] : 0;
  a37=(a23*a22);
  a37=(a4*a37);
  a26=arg[1]? arg[1][24] : 0;
  a38=(a25*a26);
  a38=(a3*a38);
  a39=arg[1]? arg[1][26] : 0;
  a40=(a28*a39);
  a40=(a1*a40);
  a38=(a38+a40);
  a38=(a6*a38);
  a37=(a37-a38);
  a38=(a27*a26);
  a38=(a9*a38);
  a40=(a31*a39);
  a40=(a8*a40);
  a38=(a38+a40);
  a37=(a37+a38);
  a37=(a10*a37);
  if (res[1]!=0) res[1][21]=a37;
  a37=(a24*a22);
  a37=(a11*a37);
  a38=(a32*a26);
  a38=(a12*a38);
  a37=(a37-a38);
  a37=(a14*a37);
  a38=(a33*a39);
  a38=(a13*a38);
  a37=(a37+a38);
  a38=(a34*a26);
  a38=(a16*a38);
  a39=(a36*a39);
  a39=(a17*a39);
  a38=(a38-a39);
  a37=(a37-a38);
  a37=(a10*a37);
  if (res[1]!=0) res[1][22]=a37;
  a37=(a30*a22);
  a37=(a20*a37);
  a38=(a29*a26);
  a38=(a19*a38);
  a37=(a37+a38);
  a37=(a10*a37);
  a37=(-a37);
  if (res[1]!=0) res[1][23]=a37;
  a26=(a35*a26);
  a26=(-a26);
  if (res[1]!=0) res[1][24]=a26;
  a22=(a35*a22);
  a22=(-a22);
  if (res[1]!=0) res[1][25]=a22;
  if (res[1]!=0) res[1][26]=a18;
  a22=arg[1]? arg[1][30] : 0;
  if (res[1]!=0) res[1][27]=a22;
  a22=arg[1]? arg[1][31] : 0;
  if (res[1]!=0) res[1][28]=a22;
  a22=arg[1]? arg[1][32] : 0;
  if (res[1]!=0) res[1][29]=a22;
  a22=arg[1]? arg[1][34] : 0;
  a26=(a23*a22);
  a26=(a4*a26);
  a37=arg[1]? arg[1][33] : 0;
  a38=(a25*a37);
  a38=(a3*a38);
  a39=arg[1]? arg[1][35] : 0;
  a40=(a28*a39);
  a40=(a1*a40);
  a38=(a38+a40);
  a38=(a6*a38);
  a26=(a26-a38);
  a38=(a27*a37);
  a38=(a9*a38);
  a40=(a31*a39);
  a40=(a8*a40);
  a38=(a38+a40);
  a26=(a26+a38);
  a26=(a10*a26);
  if (res[1]!=0) res[1][30]=a26;
  a26=(a24*a22);
  a26=(a11*a26);
  a38=(a32*a37);
  a38=(a12*a38);
  a26=(a26-a38);
  a26=(a14*a26);
  a38=(a33*a39);
  a38=(a13*a38);
  a26=(a26+a38);
  a38=(a34*a37);
  a38=(a16*a38);
  a39=(a36*a39);
  a39=(a17*a39);
  a38=(a38-a39);
  a26=(a26-a38);
  a26=(a10*a26);
  if (res[1]!=0) res[1][31]=a26;
  a26=(a30*a22);
  a26=(a20*a26);
  a38=(a29*a37);
  a38=(a19*a38);
  a26=(a26+a38);
  a26=(a10*a26);
  a26=(-a26);
  if (res[1]!=0) res[1][32]=a26;
  a37=(a35*a37);
  a37=(-a37);
  if (res[1]!=0) res[1][33]=a37;
  a22=(a35*a22);
  a22=(-a22);
  if (res[1]!=0) res[1][34]=a22;
  if (res[1]!=0) res[1][35]=a18;
  a22=arg[1]? arg[1][39] : 0;
  if (res[1]!=0) res[1][36]=a22;
  a22=arg[1]? arg[1][40] : 0;
  if (res[1]!=0) res[1][37]=a22;
  a22=arg[1]? arg[1][41] : 0;
  if (res[1]!=0) res[1][38]=a22;
  a22=arg[1]? arg[1][43] : 0;
  a37=(a23*a22);
  a37=(a4*a37);
  a26=arg[1]? arg[1][42] : 0;
  a38=(a25*a26);
  a38=(a3*a38);
  a39=arg[1]? arg[1][44] : 0;
  a40=(a28*a39);
  a40=(a1*a40);
  a38=(a38+a40);
  a38=(a6*a38);
  a37=(a37-a38);
  a38=(a27*a26);
  a38=(a9*a38);
  a40=(a31*a39);
  a40=(a8*a40);
  a38=(a38+a40);
  a37=(a37+a38);
  a37=(a10*a37);
  if (res[1]!=0) res[1][39]=a37;
  a37=(a24*a22);
  a37=(a11*a37);
  a38=(a32*a26);
  a38=(a12*a38);
  a37=(a37-a38);
  a37=(a14*a37);
  a38=(a33*a39);
  a38=(a13*a38);
  a37=(a37+a38);
  a38=(a34*a26);
  a38=(a16*a38);
  a39=(a36*a39);
  a39=(a17*a39);
  a38=(a38-a39);
  a37=(a37-a38);
  a37=(a10*a37);
  if (res[1]!=0) res[1][40]=a37;
  a37=(a30*a22);
  a37=(a20*a37);
  a38=(a29*a26);
  a38=(a19*a38);
  a37=(a37+a38);
  a37=(a10*a37);
  a37=(-a37);
  if (res[1]!=0) res[1][41]=a37;
  a26=(a35*a26);
  a26=(-a26);
  if (res[1]!=0) res[1][42]=a26;
  a22=(a35*a22);
  a22=(-a22);
  if (res[1]!=0) res[1][43]=a22;
  if (res[1]!=0) res[1][44]=a18;
  a22=arg[1]? arg[1][48] : 0;
  if (res[1]!=0) res[1][45]=a22;
  a22=arg[1]? arg[1][49] : 0;
  if (res[1]!=0) res[1][46]=a22;
  a22=arg[1]? arg[1][50] : 0;
  if (res[1]!=0) res[1][47]=a22;
  a22=arg[1]? arg[1][52] : 0;
  a26=(a23*a22);
  a26=(a4*a26);
  a37=arg[1]? arg[1][51] : 0;
  a38=(a25*a37);
  a38=(a3*a38);
  a39=arg[1]? arg[1][53] : 0;
  a40=(a28*a39);
  a40=(a1*a40);
  a38=(a38+a40);
  a38=(a6*a38);
  a26=(a26-a38);
  a38=(a27*a37);
  a38=(a9*a38);
  a40=(a31*a39);
  a40=(a8*a40);
  a38=(a38+a40);
  a26=(a26+a38);
  a26=(a10*a26);
  if (res[1]!=0) res[1][48]=a26;
  a26=(a24*a22);
  a26=(a11*a26);
  a38=(a32*a37);
  a38=(a12*a38);
  a26=(a26-a38);
  a26=(a14*a26);
  a38=(a33*a39);
  a38=(a13*a38);
  a26=(a26+a38);
  a38=(a34*a37);
  a38=(a16*a38);
  a39=(a36*a39);
  a39=(a17*a39);
  a38=(a38-a39);
  a26=(a26-a38);
  a26=(a10*a26);
  if (res[1]!=0) res[1][49]=a26;
  a26=(a30*a22);
  a26=(a20*a26);
  a38=(a29*a37);
  a38=(a19*a38);
  a26=(a26+a38);
  a26=(a10*a26);
  a26=(-a26);
  if (res[1]!=0) res[1][50]=a26;
  a37=(a35*a37);
  a37=(-a37);
  if (res[1]!=0) res[1][51]=a37;
  a22=(a35*a22);
  a22=(-a22);
  if (res[1]!=0) res[1][52]=a22;
  if (res[1]!=0) res[1][53]=a18;
  a22=arg[1]? arg[1][57] : 0;
  if (res[1]!=0) res[1][54]=a22;
  a22=arg[1]? arg[1][58] : 0;
  if (res[1]!=0) res[1][55]=a22;
  a22=arg[1]? arg[1][59] : 0;
  if (res[1]!=0) res[1][56]=a22;
  a22=arg[1]? arg[1][61] : 0;
  a37=(a23*a22);
  a37=(a4*a37);
  a26=arg[1]? arg[1][60] : 0;
  a38=(a25*a26);
  a38=(a3*a38);
  a39=arg[1]? arg[1][62] : 0;
  a40=(a28*a39);
  a40=(a1*a40);
  a38=(a38+a40);
  a38=(a6*a38);
  a37=(a37-a38);
  a38=(a27*a26);
  a38=(a9*a38);
  a40=(a31*a39);
  a40=(a8*a40);
  a38=(a38+a40);
  a37=(a37+a38);
  a37=(a10*a37);
  if (res[1]!=0) res[1][57]=a37;
  a37=(a24*a22);
  a37=(a11*a37);
  a38=(a32*a26);
  a38=(a12*a38);
  a37=(a37-a38);
  a37=(a14*a37);
  a38=(a33*a39);
  a38=(a13*a38);
  a37=(a37+a38);
  a38=(a34*a26);
  a38=(a16*a38);
  a39=(a36*a39);
  a39=(a17*a39);
  a38=(a38-a39);
  a37=(a37-a38);
  a37=(a10*a37);
  if (res[1]!=0) res[1][58]=a37;
  a37=(a30*a22);
  a37=(a20*a37);
  a38=(a29*a26);
  a38=(a19*a38);
  a37=(a37+a38);
  a37=(a10*a37);
  a37=(-a37);
  if (res[1]!=0) res[1][59]=a37;
  a26=(a35*a26);
  a26=(-a26);
  if (res[1]!=0) res[1][60]=a26;
  a22=(a35*a22);
  a22=(-a22);
  if (res[1]!=0) res[1][61]=a22;
  if (res[1]!=0) res[1][62]=a18;
  a22=arg[1]? arg[1][66] : 0;
  if (res[1]!=0) res[1][63]=a22;
  a22=arg[1]? arg[1][67] : 0;
  if (res[1]!=0) res[1][64]=a22;
  a22=arg[1]? arg[1][68] : 0;
  if (res[1]!=0) res[1][65]=a22;
  a22=arg[1]? arg[1][70] : 0;
  a26=(a23*a22);
  a26=(a4*a26);
  a37=arg[1]? arg[1][69] : 0;
  a38=(a25*a37);
  a38=(a3*a38);
  a39=arg[1]? arg[1][71] : 0;
  a40=(a28*a39);
  a40=(a1*a40);
  a38=(a38+a40);
  a38=(a6*a38);
  a26=(a26-a38);
  a38=(a27*a37);
  a38=(a9*a38);
  a40=(a31*a39);
  a40=(a8*a40);
  a38=(a38+a40);
  a26=(a26+a38);
  a26=(a10*a26);
  if (res[1]!=0) res[1][66]=a26;
  a26=(a24*a22);
  a26=(a11*a26);
  a38=(a32*a37);
  a38=(a12*a38);
  a26=(a26-a38);
  a26=(a14*a26);
  a38=(a33*a39);
  a38=(a13*a38);
  a26=(a26+a38);
  a38=(a34*a37);
  a38=(a16*a38);
  a39=(a36*a39);
  a39=(a17*a39);
  a38=(a38-a39);
  a26=(a26-a38);
  a26=(a10*a26);
  if (res[1]!=0) res[1][67]=a26;
  a26=(a30*a22);
  a26=(a20*a26);
  a38=(a29*a37);
  a38=(a19*a38);
  a26=(a26+a38);
  a26=(a10*a26);
  a26=(-a26);
  if (res[1]!=0) res[1][68]=a26;
  a37=(a35*a37);
  a37=(-a37);
  if (res[1]!=0) res[1][69]=a37;
  a22=(a35*a22);
  a22=(-a22);
  if (res[1]!=0) res[1][70]=a22;
  if (res[1]!=0) res[1][71]=a18;
  a22=arg[1]? arg[1][75] : 0;
  if (res[1]!=0) res[1][72]=a22;
  a22=arg[1]? arg[1][76] : 0;
  if (res[1]!=0) res[1][73]=a22;
  a22=arg[1]? arg[1][77] : 0;
  if (res[1]!=0) res[1][74]=a22;
  a22=arg[1]? arg[1][79] : 0;
  a23=(a23*a22);
  a23=(a4*a23);
  a37=arg[1]? arg[1][78] : 0;
  a25=(a25*a37);
  a25=(a3*a25);
  a26=arg[1]? arg[1][80] : 0;
  a28=(a28*a26);
  a28=(a1*a28);
  a25=(a25+a28);
  a25=(a6*a25);
  a23=(a23-a25);
  a27=(a27*a37);
  a27=(a9*a27);
  a31=(a31*a26);
  a31=(a8*a31);
  a27=(a27+a31);
  a23=(a23+a27);
  a23=(a10*a23);
  if (res[1]!=0) res[1][75]=a23;
  a24=(a24*a22);
  a24=(a11*a24);
  a32=(a32*a37);
  a32=(a12*a32);
  a24=(a24-a32);
  a24=(a14*a24);
  a33=(a33*a26);
  a33=(a13*a33);
  a24=(a24+a33);
  a34=(a34*a37);
  a34=(a16*a34);
  a36=(a36*a26);
  a36=(a17*a36);
  a34=(a34-a36);
  a24=(a24-a34);
  a24=(a10*a24);
  if (res[1]!=0) res[1][76]=a24;
  a30=(a30*a22);
  a30=(a20*a30);
  a29=(a29*a37);
  a29=(a19*a29);
  a30=(a30+a29);
  a30=(a10*a30);
  a30=(-a30);
  if (res[1]!=0) res[1][77]=a30;
  a37=(a35*a37);
  a37=(-a37);
  if (res[1]!=0) res[1][78]=a37;
  a22=(a35*a22);
  a22=(-a22);
  if (res[1]!=0) res[1][79]=a22;
  if (res[1]!=0) res[1][80]=a18;
  a22=arg[2]? arg[2][3] : 0;
  if (res[2]!=0) res[2][0]=a22;
  a22=arg[2]? arg[2][4] : 0;
  if (res[2]!=0) res[2][1]=a22;
  a22=arg[2]? arg[2][5] : 0;
  if (res[2]!=0) res[2][2]=a22;
  a22=cos(a5);
  a37=arg[2]? arg[2][7] : 0;
  a30=(a22*a37);
  a30=(a4*a30);
  a29=sin(a0);
  a24=arg[2]? arg[2][6] : 0;
  a34=(a29*a24);
  a34=(a3*a34);
  a36=sin(a2);
  a26=arg[2]? arg[2][8] : 0;
  a33=(a36*a26);
  a33=(a1*a33);
  a34=(a34+a33);
  a34=(a6*a34);
  a30=(a30-a34);
  a34=cos(a0);
  a33=(a34*a24);
  a33=(a9*a33);
  a32=cos(a2);
  a23=(a32*a26);
  a23=(a8*a23);
  a33=(a33+a23);
  a30=(a30+a33);
  a30=(a10*a30);
  if (res[2]!=0) res[2][3]=a30;
  a30=cos(a5);
  a33=(a30*a37);
  a33=(a11*a33);
  a23=sin(a0);
  a27=(a23*a24);
  a27=(a12*a27);
  a33=(a33-a27);
  a33=(a14*a33);
  a27=cos(a2);
  a31=(a27*a26);
  a31=(a13*a31);
  a33=(a33+a31);
  a31=cos(a0);
  a25=(a31*a24);
  a25=(a16*a25);
  a2=sin(a2);
  a26=(a2*a26);
  a26=(a17*a26);
  a25=(a25-a26);
  a33=(a33-a25);
  a33=(a10*a33);
  if (res[2]!=0) res[2][4]=a33;
  a5=sin(a5);
  a33=(a5*a37);
  a33=(a20*a33);
  a0=sin(a0);
  a25=(a0*a24);
  a25=(a19*a25);
  a33=(a33+a25);
  a33=(a10*a33);
  a33=(-a33);
  if (res[2]!=0) res[2][5]=a33;
  a33=3.0964360587002102e+00;
  a24=(a35*a24);
  a24=(a33-a24);
  if (res[2]!=0) res[2][6]=a24;
  a37=(a35*a37);
  a37=(-a37);
  if (res[2]!=0) res[2][7]=a37;
  if (res[2]!=0) res[2][8]=a18;
  a37=arg[2]? arg[2][12] : 0;
  if (res[2]!=0) res[2][9]=a37;
  a37=arg[2]? arg[2][13] : 0;
  if (res[2]!=0) res[2][10]=a37;
  a37=arg[2]? arg[2][14] : 0;
  if (res[2]!=0) res[2][11]=a37;
  a37=arg[2]? arg[2][16] : 0;
  a24=(a22*a37);
  a24=(a4*a24);
  a25=arg[2]? arg[2][15] : 0;
  a26=(a29*a25);
  a26=(a3*a26);
  a28=arg[2]? arg[2][17] : 0;
  a38=(a36*a28);
  a38=(a1*a38);
  a26=(a26+a38);
  a26=(a6*a26);
  a24=(a24-a26);
  a26=(a34*a25);
  a26=(a9*a26);
  a38=(a32*a28);
  a38=(a8*a38);
  a26=(a26+a38);
  a24=(a24+a26);
  a24=(a10*a24);
  if (res[2]!=0) res[2][12]=a24;
  a24=(a30*a37);
  a24=(a11*a24);
  a26=(a23*a25);
  a26=(a12*a26);
  a24=(a24-a26);
  a24=(a14*a24);
  a26=(a27*a28);
  a26=(a13*a26);
  a24=(a24+a26);
  a26=(a31*a25);
  a26=(a16*a26);
  a28=(a2*a28);
  a28=(a17*a28);
  a26=(a26-a28);
  a24=(a24-a26);
  a24=(a10*a24);
  if (res[2]!=0) res[2][13]=a24;
  a24=(a5*a37);
  a24=(a20*a24);
  a26=(a0*a25);
  a26=(a19*a26);
  a24=(a24+a26);
  a24=(a10*a24);
  a24=(-a24);
  if (res[2]!=0) res[2][14]=a24;
  a25=(a35*a25);
  a25=(-a25);
  if (res[2]!=0) res[2][15]=a25;
  a37=(a35*a37);
  a33=(a33-a37);
  if (res[2]!=0) res[2][16]=a33;
  if (res[2]!=0) res[2][17]=a18;
  a33=arg[2]? arg[2][21] : 0;
  if (res[2]!=0) res[2][18]=a33;
  a33=arg[2]? arg[2][22] : 0;
  if (res[2]!=0) res[2][19]=a33;
  a33=arg[2]? arg[2][23] : 0;
  if (res[2]!=0) res[2][20]=a33;
  a33=arg[2]? arg[2][25] : 0;
  a22=(a22*a33);
  a4=(a4*a22);
  a22=arg[2]? arg[2][24] : 0;
  a29=(a29*a22);
  a3=(a3*a29);
  a29=arg[2]? arg[2][26] : 0;
  a36=(a36*a29);
  a1=(a1*a36);
  a3=(a3+a1);
  a6=(a6*a3);
  a4=(a4-a6);
  a34=(a34*a22);
  a9=(a9*a34);
  a32=(a32*a29);
  a8=(a8*a32);
  a9=(a9+a8);
  a4=(a4+a9);
  a4=(a10*a4);
  a7=(a7+a4);
  if (res[2]!=0) res[2][21]=a7;
  a30=(a30*a33);
  a11=(a11*a30);
  a23=(a23*a22);
  a12=(a12*a23);
  a11=(a11-a12);
  a14=(a14*a11);
  a27=(a27*a29);
  a13=(a13*a27);
  a14=(a14+a13);
  a31=(a31*a22);
  a16=(a16*a31);
  a2=(a2*a29);
  a17=(a17*a2);
  a16=(a16-a17);
  a14=(a14-a16);
  a14=(a10*a14);
  a15=(a15+a14);
  if (res[2]!=0) res[2][22]=a15;
  a5=(a5*a33);
  a20=(a20*a5);
  a0=(a0*a22);
  a19=(a19*a0);
  a20=(a20+a19);
  a10=(a10*a20);
  a21=(a21-a10);
  if (res[2]!=0) res[2][23]=a21;
  a22=(a35*a22);
  a22=(-a22);
  if (res[2]!=0) res[2][24]=a22;
  a35=(a35*a33);
  a35=(-a35);
  if (res[2]!=0) res[2][25]=a35;
  if (res[2]!=0) res[2][26]=a18;
  return 0;
}

CASADI_SYMBOL_EXPORT int quadrotor_expl_vde_forw(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int quadrotor_expl_vde_forw_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int quadrotor_expl_vde_forw_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quadrotor_expl_vde_forw_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int quadrotor_expl_vde_forw_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quadrotor_expl_vde_forw_release(int mem) {
}

CASADI_SYMBOL_EXPORT void quadrotor_expl_vde_forw_incref(void) {
}

CASADI_SYMBOL_EXPORT void quadrotor_expl_vde_forw_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int quadrotor_expl_vde_forw_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int quadrotor_expl_vde_forw_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real quadrotor_expl_vde_forw_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quadrotor_expl_vde_forw_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quadrotor_expl_vde_forw_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quadrotor_expl_vde_forw_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    case 4: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quadrotor_expl_vde_forw_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int quadrotor_expl_vde_forw_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif