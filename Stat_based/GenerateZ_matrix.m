function [] = GenerateZ_matrix(N,m)
%GENERATEZ_MATRIX Summary of this function goes here
%   Detailed explanation goes here
Z_choose=[[0,0]',[3.5,0]',[0,3.5]',[0,-3.5]',[-3.5,0]'];
Z_all_array=[];
Z_all_array_digit=zeros(N,m);
P=(1/20)*(ones(5)+15*eye(5));
mc=dtmc(P);
for i=1:N %This is moded from problem 1 i create
    
    S=simulate(mc,m-1);
    Z_all_array_digit(i,:)=S;
    Z_all_array=[Z_all_array;Z_choose(:,S)];
    %I create to array to hold all the relevant data for the future simulation of X
    

end   
save('Z_all_array.mat','Z_all_array')
save('Z_all_array_digit.mat','Z_all_array_digit')


