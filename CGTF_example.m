function CGTF_example()
maxIter=100;
R_cov=5;
init();
v_mu=10^1*[1; 1; 0];
v_crho=10^2*[1; 1; 0];
v_prhoA=10^2*[1; 1; 0];
v_prhoD=10^2*[1; 1; 0];
mis_probX=0.1;
mis_probG1=0.3;
mis_probG2=0.95;
mis_probG3=0;
tol=10^-5;

% [bool_mis_init_X,tX,bool_mis_G1,bool_mis_G2,tG1,tG2,tG3]=generate_UserLocAct_dataset;
load('Digg.mat');
SizeTen = size(UserStoryTime);
I1 = SizeTen(1);
I2 = SizeTen(2);
I3 = SizeTen(3);

tX = double(full(UserStoryTime));
tG1 = UserUser;
tG1 = tG1 + diag(ones(size(tG1,1),1));
tG2 = StoryStory;
tG3 = zeros(I3);
[iX,bool_mis_X]=gen_mis_pattern(tX,mis_probX);
[iG1,bool_mis_G1]=gen_mis_pattern(tG1,mis_probG1);
[iG2,bool_mis_G2]=gen_mis_pattern(tG2,mis_probG2);
[iG3,bool_mis_G3]=gen_mis_pattern(tG3,mis_probG3);
ninit=1;
if ~ninit
    facs{1}=symnmf_newton(iG1,R);
    facs{2}=symnmf_newton(iG2,R);
    facs{3}=symnmf_newton(iG3,R);
else
    facs={};
end
%% CGTF
[X_CGTF,G1_CGTF,G2_CGTF,G3_CGTF,A1,A2,A3,D1,D2,D3]=CGTF_wrapper(iX,bool_mis_X,iG1,bool_mis_G1,iG2,bool_mis_G2,iG3,bool_mis_G3,v_mu,v_crho,v_prhoA,v_prhoD,R_cov,maxIter,tol,ninit,facs);


end
function init()
pth=which('init');
path_left=pth(1:end-(length('/CGTF_example')+2));
addpath(genpath([path_left '/tensor_toolbox']));
addpath(genpath([path_left '/tensorlab']));

end