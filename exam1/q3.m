%YuQi
%NUid:001304349
%qi.yu1@husky.neu.edu
clear all;
v_mu = 0;% noise
v_sigma = 1;
wtrue = [1,-0.13,-0.65,0.12];% w
n = 1;% dimentional
N = 10;% ten iid samples
x = zeros(N,n);% store Xn
y = zeros(N,n);% store Yn
v = zeros(N,n);% store noise
gamma2 = [10^-20, 10^-19, 10^-18, 10^-17, 10^-16, 10^-15, 10^-14, 10^-13, 10^-12, 10^-11, 10^-10, 10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1, 0.1 ,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 3, 5, 7, 10^1, 10^2, 10^3, 10^4, 10^5, 10^6, 10^7, 10^8, 10^9, 10^10, 10^11, 10^12, 10^13, 10^14, 10^15, 10^16, 10^17, 10^18, 10^19, 10^20, 10^21, 10^22, 10^23, 10^24, 10^25];% gamma square

for k = 1:length(gamma2)% 60 gamma
    for i =1:100% each gamma test 100 times
        
        for j = 1:10% generate D
            x(j) = (2*rand-1)*1;
            v(j) = normrnd(v_mu,v_sigma);% different noise
            y(j) = wtrue*[x(j).^3;x(j).^2;x(j).^1;x(j).^0] + v(j);
        end
        clear j;
    
        for j = 1:10% cal b(x)*b(x)' and sum1 <- see math expression in my report
            bxT(:,:,j) = [(x(j))^3; (x(j))^2; (x(j))^1; 1] * [(x(j))^3; (x(j))^2; (x(j))^1; 1]';
        end
        clear j;
        sum1 = bxT(:,:,1)+bxT(:,:,2)+bxT(:,:,3)+bxT(:,:,4)+bxT(:,:,5)+bxT(:,:,6)+bxT(:,:,7)+bxT(:,:,8)+bxT(:,:,9)+bxT(:,:,10);
    
        for j = 1:10% cal b(x)*y(i)' and sum3 <- see math expression in my report
            bxyi(:,:,j) = [(x(j))^3; (x(j))^2; (x(j))^1; 1] * y(j);
        end
        clear j;
        sum3 = bxyi(:,:,1)+bxyi(:,:,2)+bxyi(:,:,3)+bxyi(:,:,4)+bxyi(:,:,5)+bxyi(:,:,6)+bxyi(:,:,7)+bxyi(:,:,8)+bxyi(:,:,9)+bxyi(:,:,10);
        
        % cal w prediction - MAP
        w(:,i) = inv(sum1+(v_sigma)^2/gamma2(k)*eye(4))*sum3;
        % cal w prediction - ML
        wml(:,i) = inv(sum1+(v_sigma)^2)*sum3;
        % cal SE - MAP
        mse(k,i) = (wtrue*wtrue'-w(:,i)'*w(:,i))^2;
        % cal SE - ML
        mseml(k,i) = (wtrue*wtrue'-wml(:,i)'*wml(:,i))^2;
    end
    wmean(:,k) = mean(w,2);% get mean of 100 times
end

% MAP
msesorted = sort(mse,2);% sort and get each percentage sample point to plot the error
error0 = msesorted(:,1);
error25 = msesorted(:,25);
error50 = msesorted(:,50);
error75 = msesorted(:,75);
error100 = msesorted(:,100);

% ML
msemlsorted = sort(mseml,2);% sort and get each percentage sample point to plot the error
errorml0 = msemlsorted(:,1);
errorml25 = msemlsorted(:,25);
errorml50 = msemlsorted(:,50);
errorml75 = msemlsorted(:,75);
errorml100 = msemlsorted(:,100);

% plot MAP
figure(1);
for k = 1:length(gamma2)% plot error with each gamma
    plot(gamma2(k),error0(k),'*m');
    hold on;
    set(gca,'xscale','log')
    set(gca,'yscale','log')
    plot(gamma2(k),error25(k),'ob');
    hold on;
    set(gca,'xscale','log')
    set(gca,'yscale','log')
    plot(gamma2(k),error50(k),'*r');
    hold on;
    set(gca,'xscale','log')
    set(gca,'yscale','log')
    plot(gamma2(k),error75(k),'og');
    hold on;
    set(gca,'xscale','log')
    set(gca,'yscale','log')
    plot(gamma2(k),error100(k),'*k');
    hold on;
    set(gca,'xscale','log')
    set(gca,'yscale','log')
end
legend('Minimum Errors','25% Errors','50% Errors','75% Errors','Maximum Errors','location','southwest');
title('MAP - Squared Errors (60 different gamma)');
xlabel('gamma'), ylabel('Squared Errors')

% plot ML
% figure(2);
% for k = 1:length(gamma2)% plot error with each gamma
%     plot(gamma2(k),errorml0(k),'*m');
%     hold on;
%     set(gca,'xscale','log')
%     set(gca,'yscale','log')
%     plot(gamma2(k),errorml25(k),'ob');
%     hold on;
%     set(gca,'xscale','log')
%     set(gca,'yscale','log')
%     plot(gamma2(k),errorml50(k),'*r');
%     hold on;
%     set(gca,'xscale','log')
%     set(gca,'yscale','log')
%     plot(gamma2(k),errorml75(k),'og');
%     hold on;
%     set(gca,'xscale','log')
%     set(gca,'yscale','log')
%     plot(gamma2(k),errorml100(k),'*k');
%     hold on;
%     set(gca,'xscale','log')
%     set(gca,'yscale','log')
% end
% legend('Minimum Errors','25% Errors','50% Errors','75% Errors','Maximum Errors','location','southwest');
% title('ML - Squared Errors (60 different gamma)');
% xlabel('gamma'), ylabel('Squared Errors')
