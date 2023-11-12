function f_lamba = f_lamba(lamba,alfa,beta,tmin,tmax,deltaI,phi, S, P, I)

prod = 0;
for i = tmin:tmax
    k = kappa(S(i),phi,P,lamba,I(i));

    prod=prod+gammaln(k+deltaI(i))-sum(log(1:deltaI(i)))-gammaln(k)+log((1-phi)^k);
end
f_lamba = log((lamba^(alfa - 1)*exp(-beta*lamba)))+prod;
end