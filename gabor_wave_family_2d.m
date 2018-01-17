% Produce gabor wavelet family
%
% Input:
%   n: wavelet size
%   K,S,Q: total rotation and scale factors
%   sigma, zeta, eta: coefficients
%   a: scale base
%
% Ouput:
%   psi: a family of 2d array which consist gabor wavelet family. First and
%   second dimension are row and column, third dimension relates to
%   corresponding rotation factor, fourth to coresponding scale factor.

function psi = gabor_wave_family_2d(n,K,Q,S,sigma,zeta,eta,a)
psi = zeros(n,n,K,Q*S);
for i = 1:K
    for j = 0:S
        for l = 0:Q-1
            psi(:,:,i,j*Q+l+1) = gabor_wave_2d(n,sigma,zeta,eta,i*pi/K,a,j+l/Q);
        end
    end
end