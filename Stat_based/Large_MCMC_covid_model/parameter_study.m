function [data] = parameter_study(sigma,beta,M,a,b,N,IG,breakpoints)
%PARAMETER_STUDY Summary of this function goes here
%   Detailed explanation goes here



if IG==1
%GERMANY
    I=load('germany_infected.csv')';
    R=load('germany_removed.csv')';
    P = 83 * 10^6;


else
%IRAN
    I=load('iran_infected.csv')';
    R=load('iran_removed.csv')';
    P = 84 * 10^6;
end
%calculations form data
S = ones(1,length(I))*P - R - I;
deltaI = S(1:end-1)-S(2:end);
deltaR = R(2:end)-R(1:end-1);
I=I(1:end-1);
R=R(1:end-1);
S=S(1:end-1);

%parameters:
alfa = 2;
phi = 0.995;
T = length(I);

%hyperparameters to be tuned

Set_break=(1:(2+breakpoints));
Time_break=(6:(5+breakpoints));
Set_break=[Set_break,Time_break];


%% hybdrid sampler 
% Indexs (prob_i_r,lambda,lambda,lambda,lambda,time,time,time) First prob, 4 lambda is max, 3 time break out
data = zeros(8,N);
data(:,1) = data(:,1) + [0.5 ,0.2, 0.2, 0.2, 0.2, 10, 50, 80]'; % initial guess
for i=1:N

    for j = Set_break
        
        if j==1 %Step for p_i_r //gibbs
            
            data(1,i) = betarnd(sum(deltaR)+a-1,sum(I-deltaR)+b-1);
   
        elseif j <= (2+breakpoints) %Step for Lambda // Metropolis

            if j==(2+breakpoints) %Select t max and min for a specific lambda
                tmax=T;
            else
                tmax=data(j+4,i);
            end
            
            if j==2
                tmin=1;
            else
                tmin=data(j+3,i);
            end
            
            %Here i have done all pre process and is ready for Metropolis Hastings step
            %First suggest 
            lamba_proposal = data(j,i) + normrnd(0,sigma);
            if lamba_proposal<0
                data(j,i+1) = data(j,i);
            %Calc the metro alpha
            else
                Metropolis_alpha=min(0,f_lamba(lamba_proposal,alfa,beta,tmin,tmax,deltaI,phi,S,P,I)-f_lamba(data(j,i),alfa,beta,tmin,tmax,deltaI,phi,S,P,I));
                u=log(unifrnd(0,1));
                %Compare rand with metro

                if u <= Metropolis_alpha
                    data(j,i+1) = lamba_proposal;
                else
                    data(j,i+1) = data(j,i);
                end
            end
            
        elseif j > 5 %Step for t // Metropolis
            %Select right lambda_R and lambda_L
            lambda_R=data(j-3,i);
            lambda_L=data(j-4,i);
            
            if j==(max(Set_break)) %Select t max and min for a specific time
                tmax=T;
            else
                tmax=data(j+1,i);
            end
            
            if j==6
                tmin=1;
            else
                tmin=data(j-1,i);
            end
            
           %Here i have done all pre process and is ready for Metropolis Hastings step
           %First suggest 
           
           t_proposal = data(j,i) + randi(M)* (-1)^(randi(2));
            
           if t_proposal<tmin%First i deal with the two cases where the proposal hit a boundary of any form
               data(j,i+1) = data(j,i);
           elseif t_proposal>tmax

               data(j,i+1) = data(j,i);
           else
               %Next we are ready for metropolis

               Metropolis_alpha= min(0,f_t(lambda_L,lambda_R,tmin,t_proposal,tmax,deltaI,phi,S,P,I) - f_t(lambda_L,lambda_R,tmin,data(j,i),tmax,deltaI,phi,S,P,I));
               u=log(unifrnd(0,1));

               %Compare rand with metro
               if u <= Metropolis_alpha
                   data(j,i+1) = t_proposal;
               else
                   data(j,i+1) = data(j,i);
               end
           end
            

            
        end
    
    
    end
end
end

