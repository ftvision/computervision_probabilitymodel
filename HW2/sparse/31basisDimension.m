load IMAGES

dim1 = [64, 25];
dim2 = [64, 64];
dim3 = [64, 100];

A1 = rand(dim1)-0.5;
A1 = A1*diag(1./sqrt(sum(A1.*A1)));
figure(1), colormap(gray)
A = A1;
sparsenet

A2 = rand(dim2)-0.5;
A2 = A2*diag(1./sqrt(sum(A2.*A2)));
figure(1), colormap(gray)
A = A2;
sparsenet

A3 = rand(dim3)-0.5;
A3 = A3*diag(1./sqrt(sum(A3.*A3)));
figure(1), colormap(gray)
A = A3
sparsenet
