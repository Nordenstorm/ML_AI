function f_t = f_t(lamba1,lamba2,tmin,tbreak,tmax,deltaI,phi, S, P, I)



prod1 = 0;
for i = tmin:tbreak
    k = kappa(S(i),phi,P,lamba1,I(i));
    prod1 = prod1 +  gammaln(deltaI(i)+k) - sum(log(1:deltaI(i))) - gammaln(k) + log((1-phi)^k);

end

prod2 = 0;
for i = tbreak:tmax
    k = kappa(S(i),phi,P,lamba2,I(i));
    prod2= prod2 + gammaln(deltaI(i)+k) -sum(log(1:deltaI(i))) - gammaln(k) + log((1-phi)^k);

end

f_t = prod1 + prod2;


end
