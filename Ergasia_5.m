clear all
close all
% initialization
%iris data set
[x , y] = iris_dataset;
N = length(x);
% 3 layered Neural Network
w_1 = rand(4,4);
w_2 = rand(5,4);  
w_3 = rand(3,5);
seasons = 500;
J_it = zeros(seasons,1);
for l = 1:seasons
    accu_element_1 = zeros();
    accu_element_2 = zeros();
    accu_element_3 = zeros();
for k = 1:2:N
    
d = y(:,k);
% first layer input
u_1 = (x(:,k));
n_1 = length(u_1);
% FORWARD PATH
%first layer activation
v_1 = w_1*u_1;
%first layer output second layer input
u_2 = 1./(1+exp(-v_1));
n_2 = length(u_2);
%second layer activation
v_2 = w_2*(u_2);
%second layer output third layer input
u_3 = 1./(1+exp(-(v_2)));
n_3 = length(u_3);
%third layer activation
v_3 = w_3*(u_3);
%third layer output which is the final
u_4 = 1./(1+exp(-(v_3)));
n_4 = length(u_4);
%error
e = d - (u_4);
%norm of errors
J_t(k) = (transpose(e))*e;
J_it(l) = J_it(l) + J_t(k);
% BACKWARD PATH
h_diff_3= exp(-v_3)./((exp(-v_3)+1).*(exp(-v_3)+1));
%errors for all neurons
delta_3 = e .* (h_diff_3);
h_diff_2= exp(-v_2)./((exp(-v_2)+1).*(exp(-v_2)+1));
delta_2 = h_diff_2 .* (w_3'*delta_3);
h_diff_1= exp(-v_1)./((exp(-v_1)+1).*(exp(-v_1)+1));
delta_1 = (h_diff_1) .* (w_2'*delta_2) ;

% elements of gradients
%accumulation of the elements
element_1 = -(u_1 * (delta_1)');
accu_element_1 = accu_element_1 + element_1;
element_2 =  -(u_2 * (delta_2)');
accu_element_2 = accu_element_2 + element_2;
element_3 = -(u_3 * (delta_3)');
accu_element_3 = accu_element_3 + element_3;
end
%weight modification at the end of the season
accu_element_1 = accu_element_1 ./ max(abs(accu_element_1(:)));
accu_element_2 = accu_element_2 ./ max(abs(accu_element_2(:)));
accu_element_3 = accu_element_3 ./ max(abs(accu_element_3(:)));
w_1 = w_1 - (0.8 .* accu_element_1);
w_2 = w_2 - (0.8 .* accu_element_2');
w_3 = w_3 - (0.8 .* accu_element_3');
end
figure(1)
plot (J_it);
xlabel('season')
ylabel('J_i_t');
title('Learning curve');

