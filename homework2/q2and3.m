% Yu Qi
% NUid: 001304349
% qi.yu1@husky.neu.edu
% ==============instruction by Yu==============
% This script contains the function of PDF with label0 and label1
% Use the data generated in question 2 to achieve MAP and Fisher LDA
% The codes of case 1 are well commented, others are the same except for
% parameters.
%==============================================
clear all;

% basic parameters and setup
n = 2;% number of feature dimensions
N = 400;% number of samples-idd
x = zeros(n,N);% store the result

% Question 2 and 3
% case 1
% parameters about label
prior = [0.5,0.5];% set prior
label = rand(1,N) >= prior(1);% create random labels
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
% mu and sigma
mu(:,1) = [0;0]; mu(:,2) = [3;3];
Sigma(:,:,1) = eye(n); Sigma(:,:,2) = eye(n);
% create data
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
% plot
figure(1);
subplot(4,1,1);
plot(x(1,label==0),x(2,label==0),'o');
hold on;
plot(x(1,label==1),x(2,label==1),'+');
axis equal;
legend('Class 0','Class 1');
title('Data and their true labels');
xlabel('x_1'), ylabel('x_2');
hold on;
% draw the correct and incorrect - MAP
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * prior(1)/prior(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10);%/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01);%/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
case1error = (p10+p01)/N % print P(error)
case1errorcount = p10+p01

figure(1); % class 0 circle, class 1 +, correct green, incorrect red
subplot(4,1,2);
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','location','northeastoutside'); 
axis equal;
% Fisher LDA
class0 = x(:,label==0)';% seperate the data set by label
class1 = x(:,label==1)';
m1 = mean(class0);% mean of each class
m2 = mean(class1);% 
m = mean([class0;class1]);% mean of class0 and class1
n1 = size(class0,1);% size of each class
n2 = size(class1,1);

s1 = 0;
for i = 1:n1
    s1 = s1+(class0-m1)'*(class0-m1);% calculate s1
end

s2 = 0;
for i = 1:n2
    s2 = s2+(class1-m2)'*(class1-m2);% calculate s2
end

Sw = (n1*s1+n2*s2)/(n1+n2);% calculate sw
Sb = (n1*(m-m1)'*(m-m1)+n2*(m-m2)'*(m-m2))/(n1+n2);% calculate sb

A = repmat(0.1,[1,size(Sw,1)]);
B = diag(A);
[V,D]=svd(inv(Sw + B)*Sb);% Andrew Ng said that svd is better than eig.
[~,b]=max(max(D));
W1=V(:,b);% vector
W1 = abs(W1);
y1 = class0*W1;
y2 = class1*W1;
y1 = y1';
y2 = y2';

figure(1);% plot the result of Fisher LDA
subplot(4,1,3);
plot(y1(1,:),zeros(1,n1),'o');
hold on;
plot(y2(1,:),zeros(1,n2),'+');
axis equal;
legend('Class 0','Class 1');

y1_l = zeros(1,n1);
y2_l = ones(1,n2);
y1 = [y1_l;y1];
y2 = [y2_l;y2];
y = zeros(2,n1+n2);
y = [y1 y2];% sort the data set like before
temp = y(2,:);
[c,pos]=sort(temp);
y(1,:)=y(1,pos);
y(2,:)=c;

% draw the correct and incorrect - MAP
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * prior(1)/prior(2); %threshold
discriminantScore = log(evalGaussian(y(2,:),mu(:,2),Sigma(:,:,2)))-log(evalGaussian(y(2,:),mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

d = zeros(1,n1+n2);
y = [y;d];
for i = 1:(n1+n2)% add correct and incorrect indexs
    if (y(1,i)==0 && decision(i)==0)
        y(3,i)=1;% true negative
        cnt(i) = 0;
    elseif (y(1,i)==0 && decision(i)==1)
        y(3,i)=2;% false positive
        cnt(i) = 1;
    elseif (y(1,i)==1 && decision(i)==0)
        y(3,i)=3;% false negative
        cnt(i) = 2;
    elseif (y(1,i)==1 && decision(i)==1)
        y(3,i)=4;% true positive
        cnt(i) = 0;
    end
end
 
figure(1); % class 0 circle, class 1 +, correct green, incorrect red
subplot(4,1,4);
for i = 1:(n1+n2)
    if(y(3,i)==1)
        plot(y(2,i),0,'og');hold on;
    elseif(y(3,i)==4)
        plot(y(2,i),0,'+g');hold on;
    elseif(y(3,i)==2)
        plot(y(2,i),0,'or');hold on;
    elseif(y(3,i)==3)
        plot(y(2,i),0,'+r');hold on;
    end
end
if(isempty(find(cnt==2)) && isempty(find(cnt==1)))
    hold on;
elseif(isempty(find(cnt==2)))
    fp = min(find(cnt==1));
    de = fp+2;
    xline(y(2,de));hold on;
elseif(isempty(find(cnt==1)))
    fn = max(find(cnt==2));
    de = fn-2;
    xline(y(2,de));hold on;
else
    fn = max(find(cnt==2));
    fp = min(find(cnt==1));
    de = round((fn+fp)/2);
    xline(y(2,de));hold on;
end


% case 2
% parameters about label
prior = [0.5,0.5];% set prior
label = rand(1,N) >= prior(1);% create random labels
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
% mu and sigma
mu(:,1) = [0;0]; mu(:,2) = [3;3];
Sigma(:,:,1) = [3,1;1,0.8]; Sigma(:,:,2) = [3,1;1,0.8];
% create data
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
% plot
figure(2);
subplot(4,1,1);
plot(x(1,label==0),x(2,label==0),'o');
hold on;
plot(x(1,label==1),x(2,label==1),'+');
axis equal;
legend('Class 0','Class 1');
title('Data and their true labels');
xlabel('x_1'), ylabel('x_2');
hold on;
% draw the correct and incorrect
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * prior(1)/prior(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10);%/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01);%/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
case2error = (p10+p01)/N
case2errorcount = p10+p01

figure(2); % class 0 circle, class 1 +, correct green, incorrect red
subplot(4,1,2);
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','location','northeastoutside'); 
axis equal;
% Fisher LDA
class0 = x(:,label==0)';
class1 = x(:,label==1)';
m1 = mean(class0);% class 0
m2 = mean(class1);% class 1
m = mean([class0;class1]);
n1 = size(class0,1);
n2 = size(class1,1);

s1 = 0;
for i = 1:n1
    s1 = s1+(class0-m1)'*(class0-m1);
end

s2 = 0;
for i = 1:n2
    s2 = s2+(class1-m2)'*(class1-m2);
end

Sw = (n1*s1+n2*s2)/(n1+n2);
Sb = (n1*(m-m1)'*(m-m1)+n2*(m-m2)'*(m-m2))/(n1+n2);

A = repmat(0.1,[1,size(Sw,1)]);
B = diag(A);
[V,D]=svd(inv(Sw + B)*Sb);
[~,b]=max(max(D));
W2=V(:,b);
W2 = abs(W2);
y1 = class0*W2;
y2 = class1*W2;
y1 = y1';
y2 = y2';

figure(2);
subplot(4,1,3);
plot(y1(1,:),zeros(1,n1),'o');
hold on;
plot(y2(1,:),zeros(1,n2),'+');
axis equal;
legend('Class 0','Class 1');

y1_l = zeros(1,n1);
y2_l = ones(1,n2);
y1 = [y1_l;y1];
y2 = [y2_l;y2];
y = zeros(2,n1+n2);
y = [y1 y2];
temp = y(2,:);
[c,pos]=sort(temp);
y(1,:)=y(1,pos);
y(2,:)=c;

% draw the correct and incorrect
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * prior(1)/prior(2); %threshold
discriminantScore = log(evalGaussian(y(2,:),mu(:,2),Sigma(:,:,2)))-log(evalGaussian(y(2,:),mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

d = zeros(1,n1+n2);
y = [y;d];
for i = 1:(n1+n2)
    if (y(1,i)==0 && decision(i)==0)
        y(3,i)=1;% true negative
        cnt(i) = 0;
    elseif (y(1,i)==0 && decision(i)==1)
        y(3,i)=2;% false positive
        cnt(i) = 1;
    elseif (y(1,i)==1 && decision(i)==0)
        y(3,i)=3;% false negative
        cnt(i) = 2;
    elseif (y(1,i)==1 && decision(i)==1)
        y(3,i)=4;% true positive
        cnt(i) = 0;
    end
end
 
figure(2); % class 0 circle, class 1 +, correct green, incorrect red
subplot(4,1,4);
for i = 1:(n1+n2)
    if(y(3,i)==1)
        plot(y(2,i),0,'og');hold on;
    elseif(y(3,i)==4)
        plot(y(2,i),0,'+g');hold on;
    elseif(y(3,i)==2)
        plot(y(2,i),0,'or');hold on;
    elseif(y(3,i)==3)
        plot(y(2,i),0,'+r');hold on;
    end
end
if(isempty(find(cnt==2)) && isempty(find(cnt==1)))
    hold on;
elseif(isempty(find(cnt==2)))
    fp = min(find(cnt==1));
    de = fp+2;
    xline(y(2,de));hold on;
elseif(isempty(find(cnt==1)))
    fn = max(find(cnt==2));
    de = fn-2;
    xline(y(2,de));hold on;
else
    fn = max(find(cnt==2));
    fp = min(find(cnt==1));
    de = round((fn+fp)/2);
    xline(y(2,de));hold on;
end

% case 3
% parameters about label
prior = [0.5,0.5];% set prior
label = rand(1,N) >= prior(1);% create random labels
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
% mu and sigma
mu(:,1) = [0;0]; mu(:,2) = [2;2];
Sigma(:,:,1) = [2,0.5;0.5,1]; Sigma(:,:,2) = [2,-1.9;-1.9,5];
% create data
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
% plot
figure(3);
subplot(4,1,1);
plot(x(1,label==0),x(2,label==0),'o');
hold on;
plot(x(1,label==1),x(2,label==1),'+');
axis equal;
legend('Class 0','Class 1');
title('Data and their true labels');
xlabel('x_1'), ylabel('x_2');
hold on;
% draw the correct and incorrect
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * prior(1)/prior(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10);%/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01);%/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
case3error = (p10+p01)/N
case3errorcount = p10+p01

figure(3); % class 0 circle, class 1 +, correct green, incorrect red
subplot(4,1,2);
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','location','northeastoutside'); 
axis equal;
% Fisher LDA
class0 = x(:,label==0)';
class1 = x(:,label==1)';
m1 = mean(class0);% class 0
m2 = mean(class1);% class 1
m = mean([class0;class1]);
n1 = size(class0,1);
n2 = size(class1,1);

s1 = 0;
for i = 1:n1
    s1 = s1+(class0-m1)'*(class0-m1);
end

s2 = 0;
for i = 1:n2
    s2 = s2+(class1-m2)'*(class1-m2);
end

Sw = (n1*s1+n2*s2)/(n1+n2);
Sb = (n1*(m-m1)'*(m-m1)+n2*(m-m2)'*(m-m2))/(n1+n2);

A = repmat(0.1,[1,size(Sw,1)]);
B = diag(A);
[V,D]=svd(inv(Sw + B)*Sb);
[~,b]=max(max(D));
W3=V(:,b);
W3 = abs(W3);
y1 = class0*W3;
y2 = class1*W3;
y1 = y1';
y2 = y2';

figure(3);
subplot(4,1,3);
plot(y1(1,:),zeros(1,n1),'o');
hold on;
plot(y2(1,:),zeros(1,n2),'+');
axis equal;
legend('Class 0','Class 1');

y1_l = zeros(1,n1);
y2_l = ones(1,n2);
y1 = [y1_l;y1];
y2 = [y2_l;y2];
y = zeros(2,n1+n2);
y = [y1 y2];
temp = y(2,:);
[c,pos]=sort(temp);
y(1,:)=y(1,pos);
y(2,:)=c;

% draw the correct and incorrect
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * prior(1)/prior(2); %threshold
discriminantScore = log(evalGaussian(y(2,:),mu(:,2),Sigma(:,:,2)))-log(evalGaussian(y(2,:),mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

d = zeros(1,n1+n2);
y = [y;d];
for i = 1:(n1+n2)
    if (y(1,i)==0 && decision(i)==0)
        y(3,i)=1;% true negative
        cnt(i) = 0;
    elseif (y(1,i)==0 && decision(i)==1)
        y(3,i)=2;% false positive
        cnt(i) = 1;
    elseif (y(1,i)==1 && decision(i)==0)
        y(3,i)=3;% false negative
        cnt(i) = 2;
    elseif (y(1,i)==1 && decision(i)==1)
        y(3,i)=4;% true positive
        cnt(i) = 0;
    end
end
 
figure(3); % class 0 circle, class 1 +, correct green, incorrect red
subplot(4,1,4);
for i = 1:(n1+n2)
    if(y(3,i)==1)
        plot(y(2,i),0,'og');hold on;
    elseif(y(3,i)==4)
        plot(y(2,i),0,'+g');hold on;
    elseif(y(3,i)==2)
        plot(y(2,i),0,'or');hold on;
    elseif(y(3,i)==3)
        plot(y(2,i),0,'+r');hold on;
    end
end
if(isempty(find(cnt==2)) && isempty(find(cnt==1)))
    hold on;
elseif(isempty(find(cnt==2)))
    fp = min(find(cnt==1));
    de = fp+2;
    xline(y(2,de));hold on;
elseif(isempty(find(cnt==1)))
    fn = max(find(cnt==2));
    de = fn-2;
    xline(y(2,de));hold on;
else
    fn = max(find(cnt==2));
    fp = min(find(cnt==1));
    de = round((fn+fp)/2);
    xline(y(2,de));hold on;
end

% case 4 - ONLY prior different from 1
% parameters about label
prior = [0.05,0.95];% set prior
label = rand(1,N) >= prior(1);% create random labels
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
% mu and sigma
mu(:,1) = [0;0]; mu(:,2) = [3;3];
Sigma(:,:,1) = eye(n); Sigma(:,:,2) = eye(n);
% create data
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
% plot
figure(4);
subplot(4,1,1);
plot(x(1,label==0),x(2,label==0),'o');
hold on;
plot(x(1,label==1),x(2,label==1),'+');
axis equal;
legend('Class 0','Class 1');
title('Data and their true labels');
xlabel('x_1'), ylabel('x_2');
hold on;
% draw the correct and incorrect
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * prior(1)/prior(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10);%/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01);%/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
case4error = (p10+p01)/N
case4errorcount = p10+p01

figure(4); % class 0 circle, class 1 +, correct green, incorrect red
subplot(4,1,2);
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','location','northeastoutside'); 
axis equal;
% Fisher LDA
class0 = x(:,label==0)';
class1 = x(:,label==1)';
m1 = mean(class0);% class 0
m2 = mean(class1);% class 1
m = mean([class0;class1]);
n1 = size(class0,1);
n2 = size(class1,1);

s1 = 0;
for i = 1:n1
    s1 = s1+(class0-m1)'*(class0-m1);
end

s2 = 0;
for i = 1:n2
    s2 = s2+(class1-m2)'*(class1-m2);
end

Sw=(n1*s1+n2*s2)/(n1+n2);
Sb=(n1*(m-m1)'*(m-m1)+n2*(m-m2)'*(m-m2))/(n1+n2);

A = repmat(0.1,[1,size(Sw,1)]);
B = diag(A);
[V,D]=svd(inv(Sw + B)*Sb);
[~,b]=max(max(D));
W4=V(:,b);
W4 = abs(W4);
y1 = class0*W4;
y2 = class1*W4;
y1 = y1';
y2 = y2';

figure(4);
subplot(4,1,3);
plot(y1(1,:),zeros(1,n1),'o');
hold on;
plot(y2(1,:),zeros(1,n2),'+');
axis equal;
legend('Class 0','Class 1');

y1_l = zeros(1,n1);
y2_l = ones(1,n2);
y1 = [y1_l;y1];
y2 = [y2_l;y2];
y = zeros(2,n1+n2);
y = [y1 y2];
temp = y(2,:);
[c,pos]=sort(temp);
y(1,:)=y(1,pos);
y(2,:)=c;

% draw the correct and incorrect
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * prior(1)/prior(2); %threshold
discriminantScore = log(evalGaussian(y(2,:),mu(:,2),Sigma(:,:,2)))-log(evalGaussian(y(2,:),mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

d = zeros(1,n1+n2);
y = [y;d];
for i = 1:(n1+n2)
    if (y(1,i)==0 && decision(i)==0)
        y(3,i)=1;% true negative
        cnt(i) = 0;
    elseif (y(1,i)==0 && decision(i)==1)
        y(3,i)=2;% false positive
        cnt(i) = 1;
    elseif (y(1,i)==1 && decision(i)==0)
        y(3,i)=3;% false negative
        cnt(i) = 2;
    elseif (y(1,i)==1 && decision(i)==1)
        y(3,i)=4;% true positive
        cnt(i) = 0;
    end
end

figure(4); % class 0 circle, class 1 +, correct green, incorrect red
subplot(4,1,4);
for i = 1:(n1+n2)
    if(y(3,i)==1)
        plot(y(2,i),0,'og');hold on;
    elseif(y(3,i)==4)
        plot(y(2,i),0,'+g');hold on;
    elseif(y(3,i)==2)
        plot(y(2,i),0,'or');hold on;
    elseif(y(3,i)==3)
        plot(y(2,i),0,'+r');hold on;
    end
end
if(isempty(find(cnt==2)) && isempty(find(cnt==1)))
    hold on;
elseif(isempty(find(cnt==2)))
    fp = min(find(cnt==1));
    de = fp;
    xline(y(2,de));hold on;
elseif(isempty(find(cnt==1)))
    fn = max(find(cnt==2));
    de = fn;
    xline(y(2,de));hold on;
else
    fn = max(find(cnt==2));
    fp = min(find(cnt==1));
    de = round((fn+fp)/2);
    xline(y(2,de));hold on;
end

% case 5 - ONLY prior different from 2
% parameters about label
prior = [0.05,0.95];% set prior
label = rand(1,N) >= prior(1);% create random labels
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
% mu and sigma
mu(:,1) = [0;0]; mu(:,2) = [3;3];
Sigma(:,:,1) = [3,1;1,0.8]; Sigma(:,:,2) = [3,1;1,0.8];
% create data
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
% plot
figure(5);
subplot(4,1,1);
plot(x(1,label==0),x(2,label==0),'o');
hold on;
plot(x(1,label==1),x(2,label==1),'+');
axis equal;
legend('Class 0','Class 1');
title('Data and their true labels');
xlabel('x_1'), ylabel('x_2');
hold on;
% draw the correct and incorrect
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * prior(1)/prior(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10);%/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01);%/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
case5error = (p10+p01)/N
case5errorcount = p10+p01

figure(5); % class 0 circle, class 1 +, correct green, incorrect red
subplot(4,1,2);
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','location','northeastoutside'); 
axis equal;
% Fisher LDA
class0 = x(:,label==0)';
class1 = x(:,label==1)';
m1 = mean(class0);% class 0
m2 = mean(class1);% class 1
m = mean([class0;class1]);
n1 = size(class0,1);
n2 = size(class1,1);

s1 = 0;
for i = 1:n1
    s1 = s1+(class0-m1)'*(class0-m1);
end

s2 = 0;
for i = 1:n2
    s2 = s2+(class1-m2)'*(class1-m2);
end

Sw=(n1*s1+n2*s2)/(n1+n2);
Sb=(n1*(m-m1)'*(m-m1)+n2*(m-m2)'*(m-m2))/(n1+n2);

A = repmat(0.1,[1,size(Sw,1)]);
B = diag(A);
[V,D]=svd(inv(Sw + B)*Sb);
[~,b]=max(max(D));
W5=V(:,b);
W5 = abs(W5);
y1 = class0*W5;
y2 = class1*W5;
y1 = y1';
y2 = y2';

figure(5);
subplot(4,1,3);
plot(y1(1,:),zeros(1,n1),'o');
hold on;
plot(y2(1,:),zeros(1,n2),'+');
axis equal;
legend('Class 0','Class 1');

y1_l = zeros(1,n1);
y2_l = ones(1,n2);
y1 = [y1_l;y1];
y2 = [y2_l;y2];
y = zeros(2,n1+n2);
y = [y1 y2];
temp = y(2,:);
[c,pos]=sort(temp);
y(1,:)=y(1,pos);
y(2,:)=c;

% draw the correct and incorrect
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * prior(1)/prior(2); %threshold
discriminantScore = log(evalGaussian(y(2,:),mu(:,2),Sigma(:,:,2)))-log(evalGaussian(y(2,:),mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

d = zeros(1,n1+n2);
y = [y;d];
for i = 1:(n1+n2)
    if (y(1,i)==0 && decision(i)==0)
        y(3,i)=1;% true negative
        cnt(i) = 0;
    elseif (y(1,i)==0 && decision(i)==1)
        y(3,i)=2;% false positive
        cnt(i) = 1;
    elseif (y(1,i)==1 && decision(i)==0)
        y(3,i)=3;% false negative
        cnt(i) = 2;
    elseif (y(1,i)==1 && decision(i)==1)
        y(3,i)=4;% true positive
        cnt(i) = 0;
    end
end
 
figure(5); % class 0 circle, class 1 +, correct green, incorrect red
subplot(4,1,4);
for i = 1:(n1+n2)
    if(y(3,i)==1)
        plot(y(2,i),0,'og');hold on;
    elseif(y(3,i)==4)
        plot(y(2,i),0,'+g');hold on;
    elseif(y(3,i)==2)
        plot(y(2,i),0,'or');hold on;
    elseif(y(3,i)==3)
        plot(y(2,i),0,'+r');hold on;
    end
end
if(isempty(find(cnt==2)) && isempty(find(cnt==1)))
    hold on;
elseif(isempty(find(cnt==2)))
    fp = min(find(cnt==1));
    de = fp+2;
    xline(y(2,de));hold on;
elseif(isempty(find(cnt==1)))
    fn = max(find(cnt==2));
    de = fn-2;
    xline(y(2,de));hold on;
else
    fn = max(find(cnt==2));
    fp = min(find(cnt==1));
    de = round((fn+fp)/2);
    xline(y(2,de));hold on;
end

% case 6 - ONLY prior different from 3
% parameters about label
prior = [0.05,0.95];% set prior
label = rand(1,N) >= prior(1);% create random labels
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
% mu and sigma
mu(:,1) = [0;0]; mu(:,2) = [2;2];
Sigma(:,:,1) = [2,0.5;0.5,1]; Sigma(:,:,2) = [2,-1.9;-1.9,5];
% create data
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
% plot
figure(6);
subplot(4,1,1);
plot(x(1,label==0),x(2,label==0),'o');
hold on;
plot(x(1,label==1),x(2,label==1),'+');
axis equal;
legend('Class 0','Class 1');
title('Data and their true labels');
xlabel('x_1'), ylabel('x_2');
hold on;
% draw the correct and incorrect
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * prior(1)/prior(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10);%/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01);%/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
case6error = (p10+p01)/N
case6errorcount = p10+p01

figure(6); % class 0 circle, class 1 +, correct green, incorrect red
subplot(4,1,2);
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','location','northeastoutside'); 
axis equal;
% Fisher LDA
class0 = x(:,label==0)';
class1 = x(:,label==1)';
m1 = mean(class0);% class 0
m2 = mean(class1);% class 1
m = mean([class0;class1]);
n1 = size(class0,1);
n2 = size(class1,1);

s1 = 0;
for i = 1:n1
    s1 = s1+(class0-m1)'*(class0-m1);
end

s2 = 0;
for i = 1:n2
    s2 = s2+(class1-m2)'*(class1-m2);
end

Sw=(n1*s1+n2*s2)/(n1+n2);
Sb=(n1*(m-m1)'*(m-m1)+n2*(m-m2)'*(m-m2))/(n1+n2);

A = repmat(0.1,[1,size(Sw,1)]);
B = diag(A);
[V,D]=svd(inv(Sw + B)*Sb);
[~,b]=max(max(D));
W6=V(:,b);
W6 = abs(W6);
y1 = class0*W6;
y2 = class1*W6;
y1 = y1';
y2 = y2';

figure(6);
subplot(4,1,3);
plot(y1(1,:),zeros(1,n1),'o');
hold on;
plot(y2(1,:),zeros(1,n2),'+');
axis equal;
legend('Class 0','Class 1');

y1_l = zeros(1,n1);
y2_l = ones(1,n2);
y1 = [y1_l;y1];
y2 = [y2_l;y2];
y = zeros(2,n1+n2);
y = [y1 y2];
temp = y(2,:);
[c,pos]=sort(temp);
y(1,:)=y(1,pos);
y(2,:)=c;

% draw the correct and incorrect
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * prior(1)/prior(2); %threshold
discriminantScore = log(evalGaussian(y(2,:),mu(:,2),Sigma(:,:,2)))-log(evalGaussian(y(2,:),mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

d = zeros(1,n1+n2);
y = [y;d];
for i = 1:(n1+n2)
    if (y(1,i)==0 && decision(i)==0)
        y(3,i)=1;% true negative
        cnt(i) = 0;
    elseif (y(1,i)==0 && decision(i)==1)
        y(3,i)=2;% false positive
        cnt(i) = 1;
    elseif (y(1,i)==1 && decision(i)==0)
        y(3,i)=3;% false negative
        cnt(i) = 2;
    elseif (y(1,i)==1 && decision(i)==1)
        y(3,i)=4;% true positive
        cnt(i) = 0;
    end
end
 
figure(6); % class 0 circle, class 1 +, correct green, incorrect red
subplot(4,1,4);
axis equal;
for i = 1:(n1+n2)
    if(y(3,i)==1)
        plot(y(2,i),0,'og');hold on;
    elseif(y(3,i)==4)
        plot(y(2,i),0,'+g');hold on;
    elseif(y(3,i)==2)
        plot(y(2,i),0,'or');hold on;
    elseif(y(3,i)==3)
        plot(y(2,i),0,'+r');hold on;
    end
end
if(isempty(find(cnt==2)) && isempty(find(cnt==1)))
    hold on;
elseif(isempty(find(cnt==2)))
    fp = min(find(cnt==1));
    de = fp+2;
    xline(y(2,de));hold on;
elseif(isempty(find(cnt==1)))
    fn = max(find(cnt==2));
    de = fn-2;
    xline(y(2,de));hold on;
else
    fn = max(find(cnt==2));
    fp = min(find(cnt==1));
    de = round((fn+fp)/2);
    xline(y(2,de));hold on;
end


function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end