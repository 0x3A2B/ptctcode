%YuQi
%NUid:001304349
%qi.yu1@husky.neu.edu
clear all;
m(:,1) = [-1;0]; Sigma(:,:,1) = 0.1*[10 -4;-4,5]; % mean and covariance of data pdf conditioned on label 1
m(:,2) = [1;0]; Sigma(:,:,2) = 0.1*[5 0;0,2]; % mean and covariance of data pdf conditioned on label 2
m(:,3) = [0;1]; Sigma(:,:,3) = 0.1*eye(2); % mean and covariance of data pdf conditioned on label 3
classPriors = [0.15,0.35,0.5];
thr = [0,cumsum(classPriors)];
N = 10000; u = rand(1,N); L = zeros(1,N); x = zeros(2,N); data = zeros(4,N);% store data
%figure(1),clf, colorList = 'rbg';
for l = 1:3
    indices = find(thr(l)<=u & u<thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices));
    x(:,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
    %figure(1), plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
end

data = [L;x];% my favorite data format
% the 1st line are true labels
% the 2nd line are x1
% the 3rd line are x2
% the 4th line are decision labels

class1 = data(:,find(data(1,:)==1));% each class
class2 = data(:,find(data(1,:)==2));
class3 = data(:,find(data(1,:)==3));
cnt(1,1) = length(class1);% actual number of each class
cnt(1,2) = length(class2);
cnt(1,3) = length(class3);

figure(1);
plot(class1(2,:),class1(3,:),'+r');% plot each class
hold on;
plot(class2(2,:),class2(3,:),'ob');
hold on;
plot(class3(2,:),class3(3,:),'sg');
hold on;
axis equal;
legend('Class 1','Class 2','Class 3'), 
title('Data and their true labels'),
xlabel('x1'), ylabel('x2'), 

de1 = evalGaussian(data(2:3,:),m(:,1),Sigma(:,:,1)) *classPriors(1);% ready to make decision
de2 = evalGaussian(data(2:3,:),m(:,2),Sigma(:,:,2)) *classPriors(2);
de3 = evalGaussian(data(2:3,:),m(:,3),Sigma(:,:,3)) *classPriors(3);

for i = 1:N% add decision label
    demax = de1(i);% find which map is largest
    tmpindex = 1;
    if(de2(i)>demax)
        demax = de2(i);
        tmpindex = 2;
    end
    if(de3(i)>demax)
        demax = de3(i);
        tmpindex = 3;
    end
    data(4,i) = tmpindex;
end

class1true = data(:,find(data(1,:)==1 & data(4,:)==data(1,:)));% calculate true-false of each class
class1false = data(:,find(data(1,:)==1 & data(4,:)~=data(1,:)));
class2true = data(:,find(data(1,:)==2 & data(4,:)==data(1,:)));
class2false = data(:,find(data(1,:)==2 & data(4,:)~=data(1,:)));
class3true = data(:,find(data(1,:)==3 & data(4,:)==data(1,:)));
class3false = data(:,find(data(1,:)==3 & data(4,:)~=data(1,:)));

figure(2);% plot true-false of each class
plot(class1false(2,:),class1false(3,:),'+r');
hold on;
plot(class1true(2,:),class1true(3,:),'+g');
hold on;
plot(class2false(2,:),class2false(3,:),'or');
hold on;
plot(class2true(2,:),class2true(3,:),'og');
hold on;
plot(class3false(2,:),class3false(3,:),'sr');
hold on;
plot(class3true(2,:),class3true(3,:),'sg');
hold on;
axis equal;
legend('Correct decisions for data from Class 1','Wrong decisions for data from Class 1','Correct decisions for data from Class 2','Wrong decisions for data from Class 2','Correct decisions for data from Class 3','Wrong decisions for data from Class 3','location','southwest'), 
title('Data and their classifier decisions versus true labels'),
xlabel('x1'), ylabel('x2'),

errcnt(1,1)= length(class1false);% error count of each class
errcnt(1,2)= length(class2false);
errcnt(1,3)= length(class3false);
perr = sum(errcnt(1,:))/N;% p error
r = data(4,:);% my decision labels
c = data(1,:);% true labels

% print the answers
fprintf('actual sample of each class: %d  %d  %d\n',cnt);
fprintf('each misclassified: %d  %d  %d\n',errcnt);
fprintf('total misclassified: %d\n',sum(errcnt));
fprintf('P(error): %d\n',perr);

%confusion matrix
%                           true            decision
cm11 = length(data(:,find(data(1,:)==1 & data(4,:)==1)));
cm12 = length(data(:,find(data(1,:)==2 & data(4,:)==1)));
cm13 = length(data(:,find(data(1,:)==3 & data(4,:)==1)));
cm21 = length(data(:,find(data(1,:)==1 & data(4,:)==2)));
cm22 = length(data(:,find(data(1,:)==2 & data(4,:)==2)));
cm23 = length(data(:,find(data(1,:)==3 & data(4,:)==2)));
cm31 = length(data(:,find(data(1,:)==1 & data(4,:)==3)));
cm32 = length(data(:,find(data(1,:)==2 & data(4,:)==3)));
cm33 = length(data(:,find(data(1,:)==3 & data(4,:)==3)));
cm = [cm11,cm12,cm13;cm21,cm22,cm23;cm31,cm32,cm33]

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end