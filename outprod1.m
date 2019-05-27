function X = outprod1( A, B, C )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
[I, F] = size(A);
[J, ~] = size(B);
[K, ~] = size(C);

X = zeros(I,J,K);
for i = 1:I
    for j = 1:J
        for k = 1:K
        X(i,j,k) = 0;
        for f = 1:F
            X(i,j,k) = X(i,j,k) + A(i,f)*B(j,f)*C(k,f);
        end
        end
    end
end


end

