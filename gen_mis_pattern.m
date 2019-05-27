function [iA,bool_mis_A]=gen_mis_pattern(A,mis_propA)
bool_mis_A=rand(size(A));
bool_mis_A=bool_mis_A<mis_propA;
iA=A.*(~bool_mis_A);
end