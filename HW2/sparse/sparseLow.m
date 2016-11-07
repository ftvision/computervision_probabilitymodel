load('patchsin_lowfre.mat');

dim1 = [16*16 4];
dim2 = [16*16 36];
dim3 = [16*16 100];

A1 = rand(dim1)-0.5;
A1 = A1*diag(1./sqrt(sum(A1.*A1)));
figure(1), colormap(gray)
A = A1;
sparseFromGabor

A2 = rand(dim2)-0.5;
A2 = A2*diag(1./sqrt(sum(A2.*A2)));
figure(1), colormap(gray)
A = A2;
sparseFromGabor

A3 = rand(dim3)-0.5;
A3 = A3*diag(1./sqrt(sum(A3.*A3)));
figure(1), colormap(gray)
A = A3
sparseFromGabor
