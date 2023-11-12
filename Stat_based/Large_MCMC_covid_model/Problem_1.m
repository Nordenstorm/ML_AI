%% Load data and set parameters

%Imported data and post process
IG=1;
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
sigma = 0.1;
beta = 1;
M = 3;
a = 1;
b = 1; 
breakpoints =3; % = 1, 2, 3
Set_break=(1:(2+breakpoints));
Time_break=(6:(5+breakpoints));
Set_break=[Set_break,Time_break];
N = 10000;
burn_in=500;
% lambda,t,p
%% hybdrid sampler 
% Indexs (prob_i_r,lambda,lambda,lambda,lambda,time,time,time) First prob, 4 lambda is max, 3 time break out
data = zeros(8,N);
data(:,1) = data(:,1) + [0.08 ,0.2, 0.2, 0.2, 0.2, 26, 32, 38]'; % initial guess
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
data(:,burn_in:N);
%%
figure
Width=0.001;
h1=histogram(data(1,:));
h1.Normalization = 'probability';
h1.BinWidth = Width;
hold on
legend('probability transistion')
hold off
%%
figure
hold on
Width=0.01;
h2=histogram(data(2,:))
h3=histogram(data(3,:))
h4=histogram(data(4,:))
h5=histogram(data(5,:))
h2.Normalization = 'probability';
h3.Normalization = 'probability';
h4.Normalization = 'probability';
h5.Normalization = 'probability';
h2.BinWidth = Width;
h3.BinWidth = Width;
h4.BinWidth = Width;
h5.BinWidth = Width;
hold off
legend('\lambda_1 ','\lambda_2 ','\lambda_3 ','\lambda_4 ')
%%
figure
hold on
Width=1;
h6=histogram(data(6,:))
h7=histogram(data(7,:))
h8=histogram(data(8,:))
h6.Normalization = 'probability';
h7.Normalization = 'probability';
h8.Normalization = 'probability';

h6.BinWidth = Width;
h7.BinWidth = Width;
h8.BinWidth = Width;

hold off
legend('time break 1','time break 2','time break 3')

%%
mean(data(6,:))-14
mean(data(7,:))-14
mean(data(8,:))-14












