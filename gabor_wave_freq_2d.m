%j:scale factor, n:angle factor, K:total number of angle
function psi_hat = gabor_wave_freq_2d(n, sigma, zeta, eta, a,j,theta)

omega1 = [-3*pi:(2*pi)/n:3*pi-(2*pi)/n];
omega2 = omega1;

[omega1,omega2] = meshgrid(omega1,omega2);
omega1_new = omega1 * cos(theta) + omega2 * sin(theta);
omega2_new = - omega1 * sin(theta) + omega2 * cos(theta);

psi_hat = exp(- 1/2 * sigma^2 * (a^j * omega1_new - eta).^2) .* exp(- 1/2 * sigma^2 * zeta^2 * (a^j * omega2_new).^2);

psi_add = zeros(n,n);
for i = 1:3
    for k = 1:3
        psi_add = psi_add + psi_hat((i - 1) * n + 1:i * n,(k - 1) * n + 1:k * n);
    end
end
psi_hat = psi_add;