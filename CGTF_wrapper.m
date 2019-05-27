function [X,G1,G2,G3,A1,A2,A3,d1,d2,d3]= CGTF_wrapper(X,bool_mis_X,G1,bool_mis_G1,G2,bool_mis_G2,G3,bool_mis_G3,v_mu,v_crho,v_prhoA,v_prhoD,R,maxIter,tol,notinit,facs)
%{
The references point to the paper https://arxiv.org/abs/1809.08353
-X is an I1x I2x I3 tensor
-bool_mis_X is an I1x I2x I3 boolean tensor with 1 at the available entries
location and 0 in the missing ones
-Gi is an Iix Ii adjacency matrix 
-bool_mis_Gi is an Iix Ii matrix with 1 at the available links location and 
0 in the missing ones
-v_mu is a vector of regularizers for the optimization problem in (6)
-v_mu(i) corresponds to the regularizer for the i-th Graph
-v_crho is a vector of regularizers for the optimization problem in (7)
weighting the frobenious norm of ||A_n-\bar{A}_n||_F
-v_crho(i) corresponds to the regularizer for the i-th factor
-v_prhoA is a vector of regularizers for the optimization problem in (7)
weighting the frobenious norm of ||A_n-\tilde{A}_n||_F
-v_prhoA(i) corresponds to the regularizer for the i-th factor
-v_prhoD is a vector of regularizers for the optimization problem in (7)
weighting the frobenious norm of ||d_n-\tilde{d}_n||_F
-v_prhoD(i) corresponds to the regularizer for the i-th factor
-R is the rank of the tensor and the graphs
-maxIter is the maximum number of outer iterations for the ADMM algorithm
-tol is the prefered difference among the iterates of ADMM to terminate
-notinit is a boolean if it is set to 0 then the factors are initialized to 0 
-facs is a dictionary containing initializations for the three factor matrices
%}
plot_bool=1;
mu1=v_mu(1);
mu2=v_mu(2);
mu3=v_mu(3);

crho1=v_crho(1);
crho2=v_crho(2);
crho3=v_crho(3);

prhoA1=v_prhoA(1);
prhoA2=v_prhoA(2);
prhoA3=v_prhoA(3);

prhoD1=v_prhoD(1);
prhoD2=v_prhoD(2);
prhoD3=v_prhoD(3);

X1 = tens2mat(X,[],1)';
X2 = tens2mat(X,[],2)';
X3 = tens2mat(X,[],3)';

[I1, I2, I3] = size(X);

%% Initialization
if ~notinit
A1 = facs{1};
A2 = facs{2};
A3  = facs{3};
else
   A1 =rand(I1,R);
   A2 =rand(I2,R);
   A3 =rand(I3,R);
end


pA1 = A1;
pA2 = A2;
pA3 = A3;

pA1dual = zeros(size(A1));
pA2dual = zeros(size(A2));
pA3dual = zeros(size(A3));

cA1 = A1;
cA2 = A2;
cA3 = A3;

cA1dual = zeros(size(A1));
cA2dual = zeros(size(A2));
cA3dual = zeros(size(A3));

d1 = ones(R,1);
d2 = ones(R,1);
d3 = ones(R,1);


pd1 = d1;
pd2 = d2;
pd3 = d3;

pd1dual = zeros(size(d1));
pd2dual = zeros(size(d2));
pd3dual = zeros(size(d3));

A1_=A1;
A2_=A2;
A3_=A3;
d1_=d1;
d2_=d2;
d3_=d3;

flag=1;
s_it=0;
X = tensor_withmiss_update(A1,A2,A3,X,bool_mis_X);

while flag && s_it < maxIter
    s_it=s_it+1;
    
    A1=factor_update(A2_,A3_,X1,G1,pA1,cA1,crho1,prhoA1,mu1,pA1dual,cA1dual,diag(d1_));
    A2=factor_update(A1,A3_,X2,G2,pA2,cA2,crho2,prhoA2,mu2,pA2dual,cA2dual,diag(d2_));
    A3=factor_update(A1,A2,X3,G3,pA3,cA3,crho3,prhoA3,mu3,pA3dual,cA3dual,diag(d3_));
    
    d1=diag_update(A1,cA1,pd1,G1,prhoD1,mu1,pd1dual);
    d2=diag_update(A2,cA2,pd2,G2,prhoD2,mu2,pd2dual);
    d3=diag_update(A3,cA3,pd3,G3,prhoD3,mu3,pd3dual);
    
    cA1=cfactor_update(A1,G1,crho1,mu1,cA1dual,diag(d1));
    cA2=cfactor_update(A2,G2,crho2,mu2,cA2dual,diag(d2));
    cA3=cfactor_update(A3,G3,crho3,mu3,cA3dual,diag(d3));
   
    
    A1=real(A1);
    A2=real(A2);
    A3=real(A3);
    d1=real(d1);
    d2=real(d2);
    d3=real(d3);
    
    pA1=pfactor_update(A1,prhoA1,pA1dual);
    pA2=pfactor_update(A2,prhoA2,pA2dual);
    pA3=pfactor_update(A3,prhoA3,pA3dual);
    
    pd1=pdiag_update(d1,prhoD1,pd1dual);
    pd2=pdiag_update(d2,prhoD2,pd2dual);
    pd3=pdiag_update(d3,prhoD3,pd3dual);
    
    pd1dual=dual_update(pd1dual,prhoD1,d1,pd1);
    pd2dual=dual_update(pd2dual,prhoD2,d2,pd2);
    pd3dual=dual_update(pd3dual,prhoD3,d3,pd3);
    
    pA1dual=dual_update(pA1dual,prhoA1,A1,pA1);
    pA2dual=dual_update(pA2dual,prhoA2,A2,pA2);
    pA3dual=dual_update(pA3dual,prhoA3,A3,pA3);
    
    cA1dual=dual_update(cA1dual,crho1,A1,cA1);
    cA2dual=dual_update(cA2dual,crho2,A2,cA2);
    cA3dual=dual_update(cA3dual,crho3,A3,cA3);
    
    
    err(s_it,:) =calc_conv(A1,A2,A3,pA1,pA2,pA3,cA1,cA2,cA3,...
        d1,d2,d3,pd1,pd2,pd3,A1_,A2_,A3_,d1_,d2_,d3_);
    obj_fun(s_it)=f(A1,A2,A3,G1,mu1,G2,mu2,G3,mu3,X,diag(d1),diag(d2),diag(d3));
    
    X = tensor_withmiss_update(A1,A2,A3,X,bool_mis_X);
    G1 = graph_withmiss_update(A1,d1,cA1,G1,bool_mis_G1);
    G2 = graph_withmiss_update(A2,d2,cA2,G2,bool_mis_G2);
    G3 = graph_withmiss_update(A3,d3,cA3,G3,bool_mis_G3);
%     if s_it>maxIter-1
%         
%     end
    flag=check_error(err(s_it,:),tol);
    
    A1_=A1;
    A2_=A2;
    A3_=A3;
    d1_=d1;
    d2_=d2;
    d3_=d3;
    
end
if plot_bool
    figure(1)
    clf
    err=downsample(err,100);
    x=(0:100:2700);
    plot(err);
    figure(2)
    clf
    plot(obj_fun);
end

end
function bool_check=check_error(err,tol)
bool_check=1;

if (sum(err>tol))==0
    bool_check=0;
end

end

function err=calc_conv(A1,A2,A3,pA1,pA2,pA3,cA1,cA2,cA3,...
    d1,d2,d3,pd1,pd2,pd3,A1_,A2_,A3_,d1_,d2_,d3_);
err(1) = norm(A1-pA1,'fro');
err(2) = norm(A2-pA2,'fro');
err(3) = norm(A3-pA3,'fro');
err(4) = norm(A1-cA1,'fro');
err(5) = norm(A2-cA2,'fro');
err(6) = norm(A3-cA3,'fro');
err(7) = norm(d1-pd1,'fro');
err(8) = norm(d2-pd2,'fro');
err(9) = norm(d3-pd3,'fro');
err(10) = norm(A1-A1_,'fro');
err(11) = norm(A2-A2_,'fro');
err(12) = norm(A3-A3_,'fro');
err(13) = norm(d1-d1_,'fro');
err(14) = norm(d2-d2_,'fro');
err(15) = norm(d3-d3_,'fro');
end

function X=tensor_withmiss_update(A1,A2,A3,X,bool_mis_X)
mis_X=outprod1(A1,A2,A3).*bool_mis_X;
X=(X.*~bool_mis_X)+mis_X;
end

function G=graph_withmiss_update(A,d,cA,G,bool_mis_G)
mis_G=(A*diag(d)*cA').*bool_mis_G;
G=(G.*~bool_mis_G)+mis_G;
end

function y=dual_update(yprev,rho,pr,tpr)
y=yprev+rho*(pr-tpr);
end

function d=diag_update(A,cA,pd,G,rho_p,mu,pddual)
    I = numel(G);
    g=reshape(G,I,1);
    if norm(g)<eps
        d = pd;
    else
        M=krp(cA,A);
        s=rho_p/(mu+eps);
        R=size(A,2);
        d=(M'*M+s*eye(R))\(s*pd+M'*g-pddual);
    end
end

function pd=pdiag_update(d,rho_p,pddual)
if rho_p == 0
    pd =d;
else
    pd=max(d+(1/rho_p)*pddual, 0);
end
end


function A=factor_update(Ar1,Ar2,X,G,pA,cA,rho_c,rho_p,mu,pAdual,cAdual,D)
M=krp(Ar2,Ar1); %for A1  krp(A3,A2) for A2 krp(A3,A1) for A3 krp(A2,A1)
H=D*cA';
I=size(pA,2);
A=(X*M+mu*G*H'+rho_c*cA+rho_p*pA-pAdual-cAdual)/(M'*M+mu*(H*H')+(rho_c+rho_p)*eye(I));
end

function cA=cfactor_update(A,G,rho_c,mu,cAdual,D)
H=A*D;
I=size(A,2);
cA=(mu*G'*H+rho_c*A+cAdual)/((mu+eps)*(H'*H)+(rho_c+eps)*eye(I));
end

function pA=pfactor_update(A,rho_p,pAdual)
pA=max(A+(1/rho_p)*pAdual, 0);
end

function C=krp(A,B)
C=khatrirao(A,B);
end


function pM=pos_penalty_fun(M)
pM=zeros(size(M));
pM(M<0)=10^6;
end


function s_fval=f(A1,A2,A3,G1,mu1,G2,mu2,G3,mu3,X,D1,D2,D3)
s_fval=norm(tensor((X - outprod1(A1,A2,A3))))^2 + mu1*norm(G1 - A1*D1*A1','fro')^2 +...
    mu2*norm(G2 - A2*D2*A2','fro')^2 + mu3*norm(G3 - A3*D3*A3','fro')^2;
end

function s_fval=f_wmis(A,B,C,Wa,Ga,muA,Wb,Gb,muB,Gc,muC,X,W,Da,Db,Dc)
s_fval=norm(tensor(W.*((X - outprod1(A,B,C)))))^2 + muA*norm(Wa.*(Ga - A*Da*A'))^2 +...
    muB*norm(Wb.*(Gb - B*Db*B'))^2 + muC*norm(Gc - C*Dc*C')^2;
end