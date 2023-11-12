%Simulation_homework_1

%% HERE I SET ALL MODEL PARAMETERS AND MATRICES
%Define all the model paramaters 
delta_t=0.5;
sigma=0.5;
sigma_list=[500,5,5,200,5,5]';
alpha=0.6;
v=90;%Decibel
eta=3;
sigma_Y=1.5;
eps=0;

%Define all vector and matrices that are set
psi_z=[(delta_t^2)/2,delta_t,0]';
psi_w=[(delta_t^2)/2,delta_t,1]';
fi_line=[1,delta_t,(delta_t^2)/2;0,1,delta_t;0,0,alpha];
I=eye(2);
P=(1/20)*(ones(5)+15*eye(5));
Z_choose=[[0,0]',[3.5,0]',[0,3.5]',[0,-3.5]',[-3.5,0]'];

%Inital data
X_inital=[normrnd([0,0,0,0,0,0]',sigma_list)];
Z_c=[0,0,0,0,0]';
R=randperm(5,1);
Z_c(R)=1;

%Use all relevant tensor product
psiZ=(kron(I,psi_z));
psiW=(kron(I,psi_w));
fi=kron(I,fi_line);

%Simulation parameter
m=500;%Time steps
N=10000;%Particles

%Tensors
big_Tensor=speye(N);
fi_big=sparse((kron(big_Tensor,fi)));
psiW_big=sparse((kron(big_Tensor,psiW)));
psiZ_big=sparse(kron(big_Tensor,psiZ));

%Imported data and post process
S=load('stations.mat');
pos_vec=getfield(S,'pos_vec');
S1=load('RSSI-measurements.mat');
Y=getfield(S1,'Y');



%% Calculate Z results 

%GenerateZ_matrix(N,m)
W_all_array=normrnd(0,sigma,N*2,m);%I create to array to hold all the relevant data for the future simulation of X
Z_all_array_digit=load('Z_all_array_digit').Z_all_array_digit;
Z_all_array=load('Z_all_array').Z_all_array;
Z_all_array_digit=Z_all_array_digit(1:N,1:m);
Z_all_array=Z_all_array(1:2*N,1:m);


%%
Y=Y__;%Only for testing
X_all_array=zeros(N*6,m);
Weight=zeros(N,1);
Weight_all_array=zeros(N,m);
Y_vec=Y(:,1);

for i=0:N-1

    X_all_array(1+i*6 : (i+1)*6 ,1)=[normrnd([0,0,0,0,0,0]',sigma_list)];
    Weight(i+1,1)=p(X_all_array(1+i*6,1),X_all_array(4+i*6,1),Y_vec,v,eta,sigma_Y,pos_vec);%Note this might be wrong if tau is wrong/log
    
end

Weight_all_array(:,1)=Weight;
L=[5,4,3,2,1,0]';
diff_X=kron(ones(N,1),L);
Q=[1,0]';
diff_Z=kron(ones(N,1),Q);

for i=1:m

    ind = randsample(N,N,true,Weight_all_array(:,i)); % selection
    indX = 6*kron(ind,ones(6,1))-diff_X;
    indZ = 2*kron(ind,ones(2,1))-diff_Z;
    X_all_array(:,i)=X_all_array(indX,i);
    Z_all_array(:,i:end)=Z_all_array(indZ,i:end);
    Z_all_array_digit(:,i:end)=Z_all_array_digit(ind,i:end);

    Y_vec=Y(:,i+1);
    X_all_array(:,i+1)=fi_big*X_all_array(:,i) + psiW_big*W_all_array(:,i) + psiZ_big*Z_all_array(:,i);
    a=[];
    
    for l=0:N-1
        a=[a, p(X_all_array(1+l*6,i),X_all_array(4+l*6,i),Y_vec,v,eta,sigma_Y,pos_vec)];
    end
     
    Weight_all_array(:,i+1) = a'; 
    [M,I]=max(Weight_all_array(:,i+1));
    Weight_all_array(:,i+1) = Weight_all_array(:,i+1)/M;
    Weight_all_array(:,i+1) = Weight_all_array(:,i+1) + eps;
    Weight_all_array(:,i+1)=Weight_all_array(:,i+1)/sum(Weight_all_array(:,i+1));
end

tau=zeros(2,m);

for i=1:m

    Omega=sum(Weight_all_array(:,i));
    x=0;
    y=0;
    for l=0:N-1
          x=x+(X_all_array(1+l*6,i)*Weight_all_array(l+1,i))/Omega;
          y=y+(X_all_array(4+l*6,i)*Weight_all_array(l+1,i))/Omega;
        
    end

    tau(1,i)=x;
    tau(2,i)=y;
end


%% plot estimated trajectory
f1 = figure;
plot(tau(1,:),tau(2,:))
hold on
plot(pos_vec(1,:),pos_vec(2,:),'*')
hold on
plot(X_array(1,:),X_array(4,:))% For testing only
legend('Estimation',"Stations","Generated path")
xlim([-4000,4000])
xlabel('X¹')
ylabel('X²')
%% calculate most probable drive command
tau_z=zeros(5,m);

for i=1:m
    
    Omega=sum(Weight_all_array(:,i));
    
    for l=1:N
        tau_z(Z_all_array_digit(l,i),i)=tau_z(Z_all_array_digit(l,i),i)+Weight_all_array(l,i)/Omega;
    end
    
end


%% plot probability of most probable drive command
f2 = figure;
[M, I ]= max(tau_z);
plot(M)
ylabel('Probability of the most probable drive command')
xlabel('Timestep')
%% plot the most probable drive command
f3 = figure;
[M, I ]= max(tau_z);
plot(I)
ylabel('Most probable drive command')
xlabel('Timestep')
%% compare most probable to actual Z
f4 = figure;
plot(I)
hold on
plot(Z_array)
%% plot ESS for SISR
f5 = figure;
ESS = zeros(0,m+1);
for i=1:m+1
    ESS(i) = 1/sum((Weight_all_array(:,i)./sum(Weight_all_array(:,i))).^2);
end
temp = ESS(1,1:500)
plot(temp)
xlabel('Timestep')
ylabel('ESS')



