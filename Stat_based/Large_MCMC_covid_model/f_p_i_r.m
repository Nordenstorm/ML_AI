function f_p_i_r = f_p_i_r(deltaR,I,p_i_r,a,b)
exp1 = sum(deltaR)+a-1;
exp2 = sum(I-deltaR)+b-1;
f_p_i_r = p_i_r^exp1 * p_i_r^exp2;
end