% sparsefromGabor.m - simulates the sparse coding algorithm
% 
% Adapted from sparsenet.m
%
% Before running you must first define A and load Garbor Images.
% See the README file for further instructions.


gabor_sz=[16 16];

num_trials=10000;
batch_size=100; 

num_images=size(patchsin,2);
%image_size=size(patchsin,1);
image_size = gabor_sz(1);
BUFF=4;

[L M]=size(A);
sz=sqrt(L);

eta = 1.0;
noise_var= 0.01;
beta= 2.2;
sigma=0.316;
tol=.01;

VAR_GOAL=0.1;
S_var=VAR_GOAL*ones(M,1);
var_eta=.001;
alpha=.02;
gain=sqrt(sum(A.*A))';

X=zeros(L,batch_size);

display_every=100;

h=display_network(A,S_var);

% main loop

for t=1:num_trials

    % choose an image for this batch
    
    selection = randperm(num_images, batch_size);
    X = patchsin(:,selection);
    
    % calculate coefficients for these data via conjugate gradient routine
    
    S=cgf_fitS(A,X,noise_var,beta,sigma,tol);
    
    % calculate residual error
    
    E=X-A*S;
    
    % update bases
    
    dA=zeros(L,M);
    for i=1:batch_size
        dA = dA + E(:,i)*S(:,i)';
    end
    dA = dA/batch_size;
    
    A = A + eta*dA;
    
    % normalize bases to match desired output variance
    
    for i=1:batch_size
        S_var = (1-var_eta)*S_var + var_eta*S(:,i).*S(:,i);
    end
    gain = gain .* ((S_var/VAR_GOAL).^alpha);
    normA=sqrt(sum(A.*A));
    for i=1:M
        A(:,i)=gain(i)*A(:,i)/normA(i);
    end
    
    % display
    
    if (mod(t,display_every)==0)
        display_network(A,S_var,h);
    end
    
end
