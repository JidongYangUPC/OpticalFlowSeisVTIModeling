/* Copyright (c) China university of Petroelum, 2024.*/
/* All rights reserved.                       */


void ptprs(int nx,int nz,float dx,float dz,float dt,float**p,float**pp,float**pm,float**Ep,float**ud,float **E,float **u,float **v,float **dpdx,float**dpdz,float**vx,float**vz,float**dpdxx,float**dpdzz,float**dvv,float**epsilon,float**delta,float**s,int order,int ssize,float tol,float lambda,float gama,int iteration,float**coeff,int pml_thick,int pml_thickness);

void gausssmooth_2d(float**in,int n1,int n2, float r1);


void ptprs(int nx,int nz,float dx,float dz,float dt,float**p,float**pp,float**pm,float**Ep,float**ud,float **E,float **u,float **v,float**dpdx,float**dpdz,float**vx,float**vz,float **dpdxx,float**dpdzz,
float**dvv,float**epsilon,float**delta,float**s,int order,int ssize,float tol,float lambda,float gama,int iteration,float**coeff,int pml_thick,int pml_thickness)
{
	register int i,j,ix,iz,m,n;
	float tscl2,txscl,tzscl;
	float pc,sum;
	float maxl,Emax,abn;
	float ux,uz,ud1,uvm,uit,vit;
	float **Ex,**Ez,**Et,**um,**vm,**vxx,**vzz;
	float *uinline,*udepth;
	float *vinline,*vdepth;

	Ex = alloc2float(nz+pml_thickness, nx+pml_thickness);
	Ez = alloc2float(nz+pml_thickness, nx+pml_thickness);
	Et = alloc2float(nz+pml_thickness, nx+pml_thickness);
	um = alloc2float(nz+pml_thickness, nx+pml_thickness);
	vm = alloc2float(nz+pml_thickness, nx+pml_thickness);
	vxx = alloc2float(nz+pml_thickness, nx+pml_thickness);
	vzz = alloc2float(nz+pml_thickness, nx+pml_thickness);
	uinline = alloc1float(nz+pml_thickness);
	udepth = alloc1float(nx+pml_thickness);
	vinline = alloc1float(nz+pml_thickness);
	vdepth = alloc1float(nx+pml_thickness);
        memset(Ex[0],0,sizeof(float)*(nx+pml_thickness)*(nz+pml_thickness));
        memset(Ez[0],0,sizeof(float)*(nx+pml_thickness)*(nz+pml_thickness));
        memset(Et[0],0,sizeof(float)*(nx+pml_thickness)*(nz+pml_thickness));
        memset(um[0],0,sizeof(float)*(nx+pml_thickness)*(nz+pml_thickness));
        memset(vm[0],0,sizeof(float)*(nx+pml_thickness)*(nz+pml_thickness));
        memset(u[0],0,sizeof(float)*(nx+pml_thickness)*(nz+pml_thickness));
        memset(v[0],0,sizeof(float)*(nx+pml_thickness)*(nz+pml_thickness));
        memset(vxx[0],0,sizeof(float)*(nx+pml_thickness)*(nz+pml_thickness));
        memset(vzz[0],0,sizeof(float)*(nx+pml_thickness)*(nz+pml_thickness));

	tscl2 = dt*dt;
	txscl = 1/dx;
	tzscl = 1/dz;
//velocity
      #pragma omp parallel for default(shared) private(ix,iz,ud1) 
	for(ix=0;ix<nx+pml_thickness;ix++)
	   for(iz=0;iz<nz+pml_thickness;iz++)
	   {
		ud1=(1 + 2*epsilon[ix][iz]) + 1;
          	ud[ix][iz]=1+sqrt(1-8*(epsilon[ix][iz]-delta[ix][iz])/(ud1*ud1));

	   }

	#pragma omp parallel for default(shared) private(ix,iz,ux,uz)
	for(ix=order-1;ix<nx+pml_thickness-order;ix++)
	   for(iz=order-1;iz<nz+pml_thickness-order;iz++)
	   {
		ux =  coeff[3][0]*(p[ix+1][iz]-p[ix][iz])+
		      coeff[3][1]*(p[ix+2][iz]-p[ix-1][iz])+
			coeff[3][2]*(p[ix+3][iz]-p[ix-2][iz])+
			coeff[3][3]*(p[ix+4][iz]-p[ix-3][iz]);
		
		uz =  coeff[order-1][0]*(p[ix][iz+1]-p[ix][iz])+
			coeff[order-1][1]*(p[ix][iz+2]-p[ix][iz-1])+
			coeff[order-1][2]*(p[ix][iz+3]-p[ix][iz-2])+
			coeff[order-1][3]*(p[ix][iz+4]-p[ix][iz-3]);

                dpdx[ix][iz] = b_x[ix] * dpdx[ix][iz] + a_x[ix] * ux;
                ux = ux / K_x[ix] + dpdx[ix][iz];
		vx[ix][iz] = txscl*ux;

                dpdz[ix][iz] = b_z[iz] * dpdz[ix][iz] + a_z[iz] * uz;
                uz = uz / K_z[iz] + dpdz[ix][iz];
		vz[ix][iz] = tzscl*uz;

	   }
      #pragma omp parallel for default(shared) private(ix,iz,ux,uz,pc) 
	for(ix=order;ix<nx+pml_thickness-order+1;++ix)
	   for(iz=order;iz<nz+pml_thickness-order+1;++iz)
	   {
		ux = coeff[order-1][0]*(vx[ix][iz]-vx[ix-1][iz])+
			coeff[order-1][1]*(vx[ix+1][iz]-vx[ix-2][iz])+
			coeff[order-1][2]*(vx[ix+2][iz]-vx[ix-3][iz])+
			coeff[order-1][3]*(vx[ix+3][iz]-vx[ix-4][iz]);		

		uz = coeff[order-1][0]*(vz[ix][iz]-vz[ix][iz-1])+
			coeff[order-1][1]*(vz[ix][iz+1]-vz[ix][iz-2])+
			coeff[order-1][2]*(vz[ix][iz+2]-vz[ix][iz-3])+
			coeff[order-1][3]*(vz[ix][iz+3]-vz[ix][iz-4]);
		
                dpdxx[ix][iz] = b_x[ix] * dpdxx[ix][iz] + a_x[ix] * ux;
                dpdzz[ix][iz] = b_z[iz] * dpdzz[ix][iz] + a_z[iz] * uz;

                ux = ux / K_x[ix] + dpdxx[ix][iz];
                uz = uz / K_z[iz] + dpdzz[ix][iz];
		vxx[ix][iz] = txscl*ux;
		vzz[ix][iz] = tzscl*uz;

	        pc = 0.5*tscl2*dvv[ix][iz]*((1+2*epsilon[ix][iz])*vxx[ix][iz]+vzz[ix][iz])*ud[ix][iz]+2*p[ix][iz]-pm[ix][iz];

                E[ix][iz]=pc*pc;

	   }
		Emax=0;
	for(ix=order-1;ix<nx+pml_thickness-order;ix++)
	   for(iz=order-1;iz<nz+pml_thickness-order;iz++)
	   {
		
		if(E[ix][iz]>Emax)
		{
			Emax=E[ix][iz];
		}
	   }

	if(Emax>1e-20)
	{
	#pragma omp parallel for default(shared) private(ix,iz)
	for(ix=order-1;ix<nx+pml_thickness-order;ix++)
	   for(iz=order-1;iz<nz+pml_thickness-order;iz++)
	   {
		
		E[ix][iz]=E[ix][iz]/Emax;
	   }
	}
	else
	{
	Emax=Emax+1e-20;
	#pragma omp parallel for default(shared) private(ix,iz)
	for(ix=order-1;ix<nx+pml_thickness-order;ix++)
	   for(iz=order-1;iz<nz+pml_thickness-order;iz++)
	   {
		
		E[ix][iz]=E[ix][iz]/Emax;
	   }
	}
	
//#if 0
//optical flow
	#pragma omp parallel for default(shared) private(ix,iz,ux,uz)
	for(ix=order-1;ix<nx+pml_thickness-order;ix++)
	   for(iz=order-1;iz<nz+pml_thickness-order;iz++)
	   {
		
		ux = -0.08333333*(E[ix+2][iz]-E[ix-2][iz])+0.66666667*(E[ix+1][iz]-E[ix-1][iz]);
		uz = -0.08333333*(E[ix][iz+2]-E[ix][iz-2])+0.66666667*(E[ix][iz+1]-E[ix][iz-1]);

		Ex[ix][iz] = txscl*ux;
		Ez[ix][iz] = tzscl*uz;

         	Et[ix][iz]= (E[ix][iz]-Ep[ix][iz])/dt;
		Ep[ix][iz]= E[ix][iz];
           }


	int ii=0;
	float uk;	
	float vk;	

	while(ii < iteration)
	{
		ii++;

	#pragma omp parallel for default(shared) private(ix,iz,uk,vk)
	for(ix=order-1;ix<nx+pml_thickness-order;ix++)
	   for(iz=order-1;iz<nz+pml_thickness-order;iz++)
       {               
	       uk=u[ix][iz];
	       vk=v[ix][iz];		

               um[ix][iz]=(u[ix+1][iz]+u[ix][iz]+u[ix-1][iz]+u[ix+1][iz+1]+u[ix][iz+1]+u[ix-1][iz+1]+u[ix+1][iz-1]+u[ix][iz-1]+u[ix-1][iz-1]+u[ix][iz-1]+u[ix][iz+1]+u[ix+1][iz]+u[ix-1][iz]-u[ix][iz])/12;
               vm[ix][iz]=(v[ix+1][iz]+v[ix][iz]+v[ix-1][iz]+v[ix+1][iz+1]+v[ix][iz+1]+v[ix-1][iz+1]+v[ix+1][iz-1]+v[ix][iz-1]+v[ix-1][iz-1]+v[ix][iz-1]+v[ix][iz+1]+v[ix+1][iz]+v[ix-1][iz]-v[ix][iz])/12;

               u[ix][iz]=um[ix][iz]-(Ex[ix][iz]*(Ex[ix][iz]*um[ix][iz]+Ez[ix][iz]*vm[ix][iz]+Et[ix][iz]))/(lambda*lambda+Ex[ix][iz]*Ex[ix][iz]+Ez[ix][iz]*Ez[ix][iz]);
               v[ix][iz]=vm[ix][iz]-(Ez[ix][iz]*(Ex[ix][iz]*um[ix][iz]+Ez[ix][iz]*vm[ix][iz]+Et[ix][iz]))/(lambda*lambda+Ex[ix][iz]*Ex[ix][iz]+Ez[ix][iz]*Ez[ix][iz]);

        }

		gausssmooth_2d(u,nz+pml_thickness,nx+pml_thickness,ssize);
		gausssmooth_2d(v,nz+pml_thickness,nx+pml_thickness,ssize);

//#if 0
	}
     

	//#pragma omp parallel for default(shared) private(ix,iz,uvm)
	for(ix=order;ix<nx+pml_thickness-order+1;++ix)
	   for(iz=order;iz<nz+pml_thickness-order+1;++iz)
	   {
		
                uvm=u[ix][iz]*u[ix][iz]+v[ix][iz]*v[ix][iz];
		if(uvm>maxl)
		{
			maxl=uvm;
		}

           }
	//printf("maxuv=%f\n",maxl);

//#endif
//#if 0
      #pragma omp parallel for default(shared) private(ix,iz,uit,vit,ud1,uvm) 
	//for(ix=order;ix<nx+pml_thickness-order+1;++ix)
	//   for(iz=order;iz<nz+pml_thickness-order+1;++iz)
	for(ix=pml_thick;ix<nx+pml_thick;++ix)
	   for(iz=pml_thick;iz<nz+pml_thick;++iz)
	   {
                uvm=u[ix][iz]*u[ix][iz]+v[ix][iz]*v[ix][iz];
		if (uvm>maxl*gama)
		{
             	uit=u[ix][iz]*u[ix][iz]/uvm;
             	vit=v[ix][iz]*v[ix][iz]/uvm;     
		ud1=(1 + 2*epsilon[ix][iz])*uit + vit;
          	ud[ix][iz]=1+sqrt(1-8*(epsilon[ix][iz]-delta[ix][iz])*uit*vit/(ud1*ud1));
//		printf("uit=%.20f vit=%.20f \n",uit,vit);
		}

	   }

//#endif
//#if 0
	abn=1.0/(pml_thick*pml_thick);
//A area
	#pragma omp parallel for default(shared) private(ix,iz,ux,uz)
	for(ix=pml_thick;ix<nx+pml_thick;ix++)
	   for(iz=order;iz<pml_thick;iz++)
	  { 
		ux=iz*iz*abn;
		uz=ux*(ud[ix][pml_thick]-ud[ix][iz])+ud[ix][iz];
		ud[ix][iz]=uz;
	  }
//C area
	#pragma omp parallel for default(shared) private(ix,iz,ux,uz)
	for(ix=pml_thick;ix<nx+pml_thick;ix++)
	   for(iz=nz+pml_thick;iz<pml_thickness+nz-order;iz++)
	   {
		ux=(nz+pml_thickness-iz)*(nz+pml_thickness-iz)*abn;
		uz=ux*(ud[ix][nz+pml_thick-1]-ud[ix][iz])+ud[ix][iz];
		ud[ix][iz]=uz;
	   }

//B area

	#pragma omp parallel for default(shared) private(ix,iz,ux,uz)
	for(ix=order;ix<pml_thick;ix++)
	   for(iz=order;iz<pml_thickness+nz;iz++)
	   {
		ux=ix*ix*abn;
		uz=ux*(ud[pml_thick][iz]-ud[ix][iz])+ud[ix][iz];
		ud[ix][iz]=uz;
	   }
//D area

	#pragma omp parallel for default(shared) private(ix,iz,ux,uz)
	for(ix=nx+pml_thick;ix<nx+pml_thickness-order;ix++)
	   for(iz=order;iz<pml_thickness+nz;iz++)
	   {
		ux=(nx+pml_thickness-ix)*(nx+pml_thickness-ix)*abn;
		uz=ux*(ud[nx+pml_thick-1][iz]-ud[ix][iz])+ud[ix][iz];
		ud[ix][iz]=uz;
	   }
//#endif

      #pragma omp parallel for default(shared) private(ix,iz,pc) 
	for(ix=order;ix<nx+pml_thickness-order+1;++ix)
	   for(iz=order;iz<nz+pml_thickness-order+1;++iz)
	   {
	        pc = 0.5*tscl2*dvv[ix][iz]*((1+2*epsilon[ix][iz])*vxx[ix][iz]+vzz[ix][iz])*ud[ix][iz]+2*p[ix][iz]-pm[ix][iz];

	        pp[ix][iz] = pc+s[ix][iz];
                pm[ix][iz] = p[ix][iz];
                p[ix][iz] = pp[ix][iz];
	   }
	free2float(Et);
	free2float(Ex);
	free2float(Ez);
	free2float(um);
	free2float(vm);
	free2float(vxx);
	free2float(vzz);
	free(uinline);
	free(udepth);
	free(vinline);
	free(vdepth);

}

void gausssmooth_2d(float**in,int n1,int n2, float r1)
{
    int ix,iz,i1,j1;
    float *mat1,*mat2,*kernel;
    float sigma,s,tmp,sum,conv;
    int nw,hfs;
    /* define filter size */
    nw = round(r1);
    if (nw==0)	return;
	if (!(nw%2)) {
		if (nw > 0)
			nw += 1;
		else
			nw -= 1;
	}
    nw = abs(nw);
    /* parameters */
    hfs = abs(nw)/2;
	sigma = hfs/2.0;
	s = 2.0*sigma*sigma;

    kernel = alloc1float(2*hfs+1);

    /* create filter kernel */
    sum = 0.0;
	for (i1=0;i1<2*hfs+1;i1++)
	{
		tmp = 1.0*(i1-hfs);
	    kernel[i1] = exp(-(tmp*tmp)/s);
		sum += kernel[i1];
        }
    /* normalize kernel */
    for (i1=0;i1<2*hfs+1;i1++)
    {
        kernel[i1] /= sum;
    }

    #pragma omp parallel for default(shared) private(ix,mat1,i1,j1,conv)
    for(ix=0;ix<n2;ix++)	
    {
    /* copy input to mat */
    mat1 = alloc1float(n2+2*hfs);
    for(i1=0;i1<n1;i1++)
    {
    	mat1[i1+hfs] = in[ix][i1];
    }
    /* extend boundary */
    for(i1=0;i1<hfs;i1++)
    {
        mat1[i1] = mat1[hfs];
        mat1[i1+n1+hfs] = mat1[n1+hfs-1];
    }
    /* apply Gaussian filter */
    //#pragma omp parallel for default(shared) private(i1,j1,conv)
    for (i1=hfs;i1<n1+hfs;i1++)
    {
        /* loop over kernel*/
        conv = 0.0;
	  	for (j1=0;j1<2*hfs+1;j1++)
	  	{
	         conv += mat1[i1+j1-hfs]*kernel[j1];
                }
        /* output of filtered gradient */
        in[ix][i1-hfs] = conv;
    }
   free1float(mat1);
    }
	

    #pragma omp parallel for default(shared) private(iz,mat2,i1,j1,conv)
    for(iz=0;iz<n1;iz++)	
    {

    mat2 = alloc1float(n2+2*hfs);
    /* copy input to mat */
    for(i1=0;i1<n2;i1++)
    {
    	mat2[i1+hfs] = in[i1][iz];
    }
    /* extend boundary */
    for(i1=0;i1<hfs;i1++)
    {
        mat2[i1] = mat2[hfs];
        mat2[i1+n2+hfs] = mat2[n2+hfs-1];
    }
    /* apply Gaussian filter */
    for (i1=hfs;i1<n2+hfs;i1++)
    {
        /* loop over kernel*/
        conv = 0.0;
	  	for (j1=0;j1<2*hfs+1;j1++)
	  	{
	         conv += mat2[i1+j1-hfs]*kernel[j1];
                }
        /* output of filtered gradient */
        in[i1-hfs][iz] = conv;
    }
    free1float(mat2);
    }

   free1float(kernel);
}
