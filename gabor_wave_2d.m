%Compute 2d Gabor wavelet.
%
% Input:
%   n: wavelet size
%   sigma, zeta, eta: coefficients
%   theta: rotation angle
%   a: scale base
%   m: scale parameter
% Ouput:
%   psi: a 2d array of 2d gabor wavelet
function psi = gabor_wave_2d(n,sigma,zeta,eta,theta,a,m)

x = [-n/2:1:n/2-1];
y = x;
[X,Y] = meshgrid(x,y);
psi = zeros(n,n);

X_new = a^(-m) * (X * cos(theta) + Y * sin(theta)); %rotate x and y
Y_new = a^(-m) * (- X * sin(theta) + Y * cos(theta));
%calculate gabor wavelet
psi = a^(-2 * m) / (2 * pi * sigma^2 * zeta) * exp((- X_new.^2 - (Y_new/zeta).^2)/(2 * sigma^2)) .* exp(i * eta * X_new);

       
