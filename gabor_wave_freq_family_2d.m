function psi_hat = gabor_wave_freq_family_2d(n,K,S,Q,sigma, zeta, eta,a)

psi_hat = zeros(n,n,K,S*Q);
for i = 1:K
    for j = 0:S-1
        for l = 0:Q-1
            psi_hat(:,:,i,j*Q+l+1) = gabor_wave_freq_2d(n,sigma,zeta,eta,a,j*Q+l,i*pi/K);
        end
    end
end