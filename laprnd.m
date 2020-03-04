
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 
%%% This function generates i.i.d. laplacian distributed random numbers
%%% with mean mu and standard deviation sigma.
%%% The dimension of the output vector is [m, n].
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y  = laprnd(m, n, mu, sigma)

if nargin == 2
    mu = 0; sigma = 1;
end

if nargin == 3
    sigma = 1;
end

u = rand(m, n)-0.5;
b = sigma / sqrt(2);
y = mu - b * sign(u).* log(1- 2* abs(u));