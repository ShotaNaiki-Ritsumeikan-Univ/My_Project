 #define CHIP_6713 1

#include <stdio.h>
#include <stdint.h>

#include <csl_chiphal.h>
#include <csl_irq.h>
#include <csl_emif.h>
#include <dsk6713.h>
#include "dsk6713if.h"

void DSKIFCHECK(void);
void emif_init(void);

/* ---------------------------------------------------- */
//				Parameters and variables
/* ---------------------------------------------------- */
#define	FS		16000// Sampling frequency [Hz]
#define	TAP		300// Tap length of noise control filter (*) Tap len. of Sec. Path should be equal to TAP.
#define STP		0.5// Step size parameter (NLMS)
#define REG		0.00001// Regularization parameter

#define TIME	1// Until start ANC

//Add delay to primary signal
#define Delay 1 //if Delay = 1: On, if Delay = 0, Off
#define Delay_idx 0

//Choose updating algorythm
#define Mode 2 // if Mode=1: NLMS, if Mode = 2: VSS-NLMS

// HPF (cutoff freq. is 60 Hz)
#define a0	0.983477203353836// Coef of HPF (a[0])
#define a1	-1.96695440670767// Coef of HPF (a[1])
#define a2	0.983477203353836// Coef of HPF (a[2])
#define b1	-1.96668138526349// Coef of HPF (b[1])
#define b2	0.967227428151861// Coef of HPF (b[2])

#define C_pre 0.00 //0.97  //For high frequency emphasis

int32_t n, k, cnt; // k is index for reduction
volatile short sw_continue, sw_interrupt, sw_process;
short i, j;

float IIR_BUF1[3], IIR_BUF2[3];
float X[TAP], ADF[TAP], r_ex[TAP], temp[TAP];
volatile float output, error, norm, grad, sigma_e, sigma_x, norm_r, gamma, lambda, prim_pre, ch1_prev;
float stp;

void main(void){
    DSK6713_init();         //initialize all bord APIs(for TI C6713DSK)

    emif_init();            //Bus Timing Setting(for HEG DSK6713IF use)

    DSKIFCHECK();

    *(volatile int *)DSKIF_SETREG1 = 0x0000000f;  // Sampling clock set (8kHz)
	*(volatile int *)DSKIF_SETREG2 = 0x00000011;  // Interrupt set

    /* ---------------------------------------- */
	//				Initialization
	/* ---------------------------------------- */
    n = 0; k = 0; cnt = 0;
    sigma_e = 0.0; sigma_x = 0.0;
    norm_r = 0.0;
    lambda = (1 - (1 / (20 * TAP)));

    for(i = 0; i < TAP; i++){
        X[i] = 0.0; ADF[i] = 0.0; r_ex[i] = 0.0; temp[i] = 0.0;
    }

    for(i = 0; i < 3; i++){
        IIR_BUF1[i] = 0.0;
        IIR_BUF2[i] = 0.0;
    }

    output = 0.0;
    error = 0.0;
    norm = 0.0;
    prim_pre = 0.0;
    ch1_prev = 0.0;

 
    /* ------------------------------------- */

    IRQ_map(IRQ_EVT_EXTINT4, 4);		// IRQ_EVT_EXTINT4 is allocated in INT4. 
	IRQ_enable(IRQ_EVT_EXTINT4);		// EXT_INT4 Inttrupt Enable

	IRQ_nmiEnable();					// enable NMI(Non Maskable Interrupt)
	IRQ_globalEnable();					// set GIE(Global Interrupt Enable) 

	printf("Start\n");

	/*for(;;){
										// main routine (waiting for interrupt)
	}*/

	while( 1 )
	{
		;
	}
}

interrupt void int4(void){
    short ch1, ch2;

    // Obtain each signal
	ch1 = *(volatile short *)DSKIF_AD1;// Primary signal
	ch2 = *(volatile short *)DSKIF_AD2;// Reference signal

    // Filtering by HPF
	IIR_BUF1[2] = IIR_BUF1[1];	IIR_BUF1[1] = IIR_BUF1[0];
	IIR_BUF2[2] = IIR_BUF2[1];	IIR_BUF2[1] = IIR_BUF2[0];

	IIR_BUF1[0] = (float)ch1 - b1 * IIR_BUF1[1] - b2 * IIR_BUF1[2];
	IIR_BUF2[0] = (float)ch2 - b1 * IIR_BUF2[1] - b2 * IIR_BUF2[2];
	
	ch1 = (short)( a0 * IIR_BUF1[0] + a1 * IIR_BUF1[1] + a2 * IIR_BUF1[2] );
	ch2 = (short)( a0 * IIR_BUF2[0] + a1 * IIR_BUF2[1] + a2 * IIR_BUF2[2] );

    //Output Obtained signal
    *(volatile short *)DSKIF_DA0 = ch1; // Primary signal
    *(volatile short *)DSKIF_DA1 = ch2; // Reference signal

    //Update norm of filtered reference signal

    norm -= X[TAP-1] * X[TAP-1];

    //Shift reference signal and filtered one
    for(i = TAP-1; i > 0; i--){
        temp[i] = temp[i-1];
    }
    temp[0] = ch2 / 3276.8; //Reference

    //Shift
    for(i = TAP-1; i > 0; i--){
        X[i] = X[i-1];
    }
    X[0] = temp[Delay_idx];

    *(volatile short *)DSKIF_DA2 = 3276.8 * X[0];

    output = ADF[0] * X[0];
    for(i = 1; i < TAP; i++){
        output += ADF[i] * X[i];
    }

    error = ch1 /3276.8 - output;

    *(volatile short *)DSKIF_DA3 = 3276.8 * error;

    norm += X[0] * X[0];


    sigma_e = lambda * sigma_e + (1 - lambda) * (error * error);
    sigma_x = lambda * sigma_x + (1 - lambda) * (X[0] * X[0]);

    norm_r = 0.0;
    for(i = 0.0; i < TAP; i++){
        r_ex[i] = lambda * r_ex[i] + (1 - lambda) * X[0] * error;
        norm_r = r_ex[i] * r_ex[i];
    }
    gamma = sigma_e - ((1 / sigma_x) * norm_r);
    if(gamma < 0.0){
        gamma = 0.0;
    }

    if(sigma_e >= gamma){
        stp = 0.1 * error * (1 / (REG + norm)) * (1 - sqrt(gamma / (REG + sigma_e)));
    }else{
        stp = 0.0;
    }
    /* -------------------------------------------- */
    /*              Update ADF                      */
    /* -------------------------------------------- */
    if(n==FS*TIME){
       for(i = 0; i < TAP; i++){
            ADF[i] += stp * X[i];
        }
    }else{
        n++;
    }
}

void emif_init(void)
{
	/* DSK6713IF(HEG)  BusTimingSet */
	EMIF_FSET(CECTL2,TA,3);
	EMIF_FSET(CECTL2,MTYPE,2);

	EMIF_FSET(CECTL2,WRSETUP,0);
	EMIF_FSET(CECTL2,WRSTRB,3);
	EMIF_FSET(CECTL2,WRHLD,2);

	EMIF_FSET(CECTL2,RDSETUP,0);
	EMIF_FSET(CECTL2,RDSTRB,4);
	EMIF_FSET(CECTL2,RDHLD,1);
}

// Check Toggle Bit
void DSKIFCHECK(void)
{
	int i;
	for(i=0;i<3;i++){
		while((*(volatile int *)DSKIF_TOGGLE & 0x01) !=0);
		while((*(volatile int *)DSKIF_TOGGLE & 0x01) ==0);
	}
}
