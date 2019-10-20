%YuQi
%NUid:001304349
%qi.yu1@husky.neu.edu
clear all;
n_sigma = 0.3;% noise sigma
xtrue = 0.5;% true point in the unit circle
ytrue = 0.5;
sigmax = 0.25;% sigma of prior
sigmay = 0.25;

% K landmarks on a unit r circle at origin point
%      l1    l2             l3                                  l4
x_l = [1,   1,-1,   1,cos(120*pi/180),cos(240*pi/180),      1,-1,0, 0];
y_l = [0,   0, 0,   0,sin(120*pi/180),sin(240*pi/180),      0, 0,1,-1];
%      1    2  3    4        5              6               7  8 9  10

for i = 1:10
    n(i) = normrnd(0,n_sigma);% make each ri's noise different
    r(i) = sqrt((x_l(i)-xtrue)^2+(y_l(i)-ytrue)^2) + n(i);% ri =dti + n
end
clear i;

x = linspace(-2,2,100);
y = linspace(-2,2,100);
% I first select landmarks on unit r circle and the true point
% and I calculate the ri = dti + noise
% so the ri ~ N (dti, sigma of noise), and I know the likelihood and prior
% prior and likelihood contain all constant
[X,Y] = meshgrid(x,y);% X and Y are estimate value, so it can become contours with equal value on each contour
prior = 1/2*(X(:).^2/sigmax^2 + Y(:).^2/sigmay^2) - log(inv(2*pi*sigmax*sigmay));% calculate prior,after loge, see math in my report
likelihood1 = ((r(1)-sqrt((x_l(1)-X(:)).^2+(y_l(1)-Y(:)).^2)).^2)/(2*n_sigma^2) - log(1/(sqrt(2*pi)*n_sigma));% calculate likelihood, after loge, see math in my report
likelihood2 = ((r(2)-sqrt((x_l(2)-X(:)).^2+(y_l(2)-Y(:)).^2)).^2)/(2*n_sigma^2) - log(1/(sqrt(2*pi)*n_sigma)) + ((r(3)-sqrt((x_l(3)-X(:)).^2+(y_l(3)-Y(:)).^2)).^2)/(2*n_sigma^2) - log(1/(sqrt(2*pi)*n_sigma));
likelihood3 = ((r(4)-sqrt((x_l(4)-X(:)).^2+(y_l(4)-Y(:)).^2)).^2)/(2*n_sigma^2) - log(1/(sqrt(2*pi)*n_sigma)) + ((r(5)-sqrt((x_l(5)-X(:)).^2+(y_l(5)-Y(:)).^2)).^2)/(2*n_sigma^2) - log(1/(sqrt(2*pi)*n_sigma)) + ((r(6)-sqrt((x_l(6)-X(:)).^2+(y_l(6)-Y(:)).^2)).^2)/(2*n_sigma^2) - log(1/(sqrt(2*pi)*n_sigma));
likelihood4 = ((r(7)-sqrt((x_l(7)-X(:)).^2+(y_l(7)-Y(:)).^2)).^2)/(2*n_sigma^2) - log(1/(sqrt(2*pi)*n_sigma)) + ((r(8)-sqrt((x_l(8)-X(:)).^2+(y_l(8)-Y(:)).^2)).^2)/(2*n_sigma^2) - log(1/(sqrt(2*pi)*n_sigma)) + ((r(9)-sqrt((x_l(9)-X(:)).^2+(y_l(9)-Y(:)).^2)).^2)/(2*n_sigma^2) - log(1/(sqrt(2*pi)*n_sigma)) + ((r(10)-sqrt((x_l(10)-X(:)).^2+(y_l(10)-Y(:)).^2)).^2)/(2*n_sigma^2) - log(1/(sqrt(2*pi)*n_sigma));

map1 = likelihood1 + prior;% calculate posterior
map2 = likelihood2 + prior;
map3 = likelihood3 + prior;
map4 = likelihood4 + prior;

map1contour = reshape(map1,100,100);% reshape them
map2contour = reshape(map2,100,100);
map3contour = reshape(map3,100,100);
map4contour = reshape(map4,100,100);

figure(1);
contour(X,Y,map1contour,'Showtext', 'on');% draw contour
hold on;
plot(x_l(1),y_l(1),'or');% draw landmarks
hold on;
plot(xtrue,ytrue,'hb');% draw true point
hold on;
legend('MAP contour','Landmark Points','Actual Point'), 
title('MAP with one Landmark');
xlabel('x'), ylabel('y');
axis equal;

figure(2);
contour(X,Y,map2contour,'Showtext', 'on');% draw contour
hold on;
plot(x_l(2:3),y_l(2:3),'or');% draw landmarks
hold on;
plot(xtrue,ytrue,'hb');% draw true point
hold on;
legend('MAP contour','Landmark Points','Actual Point'), 
title('MAP with two Landmarks');
xlabel('x'), ylabel('y');
axis equal;

figure(3);
contour(X,Y,map3contour,'Showtext', 'on');% draw contour
hold on;
plot(x_l(4:6),y_l(4:6),'or');% draw landmarks
hold on;
plot(xtrue,ytrue,'hb');% draw true point
hold on;
legend('MAP contour','Landmark Points','Actual Point'), 
title('MAP with three Landmarks');
xlabel('x'), ylabel('y');
axis equal;

figure(4);
contour(X,Y,map4contour,'Showtext', 'on');% draw contour
hold on;
plot(x_l(7:10),y_l(7:10),'or');% draw landmarks
hold on;
plot(xtrue,ytrue,'hb');% draw true point
hold on;
legend('MAP contour','Landmark Points','Actual Point'), 
title('MAP with four Landmarks');
xlabel('x'), ylabel('y');
axis equal;