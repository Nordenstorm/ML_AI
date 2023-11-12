
N=10000;
burn_in=500;

parameter_choose_index=5;

sigma=[0.1,0.5,1,1.5,2.5];%Parameter 1
M=[3,5,10,15,20];%Parameter 2
beta=[1,5,10,15,20];%Parameter 3
a=[1,5,10,15,20];%Parameter 4
b=[1,5,10,15,20];%Parameter 5

breakpoints =1; % = 1, 2, 3

Parameter_investigate=[sigma ; M ; beta ; a ; b];

IG=1;

Parameter_choosen=[sigma(1),M(1),beta(1),a(1),b(1)];

averages__=zeros(8,5);
median__=zeros(8,5);
std__=zeros(8,5);


n = 5 ;
M = cell(n, 1) ;
for i=1:5

    
    Parameter_choosen(parameter_choose_index)=Parameter_investigate(parameter_choose_index,i);
    sigma_temp=Parameter_choosen(1);
    M_temp=Parameter_choosen(2);
    beta_temp=Parameter_choosen(3);
    a_temp=Parameter_choosen(4);
    b_temp=Parameter_choosen(5);
    
    data=parameter_study(sigma_temp,beta_temp,M_temp,a_temp,b_temp,N,IG,breakpoints);
    data=data(:,burn_in:N);
    M{i}=data;

end

%%

%%



hold off
%%
figure
hold on
plot_quanti=2%This plot we want p = 1 lambda= 2,3,4,5 time=6,7,8
Width=0.001;
h1=histogram(M{1}(plot_quanti,:),10)
h2=histogram(M{2}(plot_quanti,:),10)
h3=histogram(M{3}(plot_quanti,:),10)
h4=histogram(M{4}(plot_quanti,:),10)
h5=histogram(M{5}(plot_quanti,:),10)
h1.Normalization = 'probability';
h2.Normalization = 'probability';
h3.Normalization = 'probability';
h4.Normalization = 'probability';
h5.Normalization = 'probability';
h1.BinWidth = Width;
h2.BinWidth = Width;
h3.BinWidth = Width;
h4.BinWidth = Width;
h5.BinWidth = Width;
xlabel('Value of \lambda_1')

ylabel('Probability')

hold off
legend('\lambda_1 ,b=1 ','\lambda_1 ,b=5 ','\lambda_1 ,b=10 ','\lambda_1 ,b=15 ','\lambda_1 ,b=20 ')


figure
hold on
plot_quanti=1%This plot we want p = 1 lambda= 2,3,4,5 time=6,7,8
Width=0.0001;
h1=histogram(M{1}(plot_quanti,:),10)
h2=histogram(M{2}(plot_quanti,:),10)
h3=histogram(M{3}(plot_quanti,:),10)
h4=histogram(M{4}(plot_quanti,:),10)
h5=histogram(M{5}(plot_quanti,:),10)
h1.Normalization = 'probability';
h2.Normalization = 'probability';
h3.Normalization = 'probability';
h4.Normalization = 'probability';
h5.Normalization = 'probability';
h1.BinWidth = Width;
h2.BinWidth = Width;
h3.BinWidth = Width;
h4.BinWidth = Width;
h5.BinWidth = Width;
xlabel('Value of p_{i \rightarrow r}')

ylabel('Probability')

hold off
legend('p_{i \rightarrow r} ,b=1 ','p_{i \rightarrow r} ,b=5 ','p_{i \rightarrow r} ,b=10 ','p_{i \rightarrow r} ,b=15 ','p_{i \rightarrow r} ,b=20 ')


figure
hold on
plot_quanti=6%This plot we want p = 1 lambda= 2,3,4,5 time=6,7,8
Width=1;
h1=histogram(M{1}(plot_quanti,:),10)
h2=histogram(M{2}(plot_quanti,:),10)
h3=histogram(M{3}(plot_quanti,:),10)
h4=histogram(M{4}(plot_quanti,:),10)
h5=histogram(M{5}(plot_quanti,:),10)
h1.Normalization = 'probability';
h2.Normalization = 'probability';
h3.Normalization = 'probability';
h4.Normalization = 'probability';
h5.Normalization = 'probability';
h1.BinWidth = Width;
h2.BinWidth = Width;
h3.BinWidth = Width;
h4.BinWidth = Width;
h5.BinWidth = Width;
xlabel('Value of t_1')

ylabel('Probability')

hold off
legend('t_1 , b=1 ','t_1 , b=5 ','t_1 ,b=10 ','t_1 ,b=15 ','t_1 ,b=20 ')




