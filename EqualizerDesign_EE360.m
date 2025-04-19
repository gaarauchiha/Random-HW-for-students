% Equalizer Design
% Part (a)
a = randsrc(100, 1, [-1 1], 0);
t = -3:0.1:3;
p = sin(pi * t) ./ (pi * t);
p(31) = 1;

tt = -3:0.1:103;
y = zeros(size(tt));

for k = 1:100
    idx_k = 10 * k + 31;
    start_idx = idx_k - 30;
    end_idx = idx_k + 30;
    y(start_idx:end_idx) = y(start_idx:end_idx) + a(k) * p;
end

idx_plot = find(tt >= 0 & tt <= 10);
figure;
plot(tt(idx_plot), y(idx_plot));
xlabel('Time in seconds');
ylabel('Magnitude');
title('Line code y in MATLAB using the symbols a and the pulse p');
grid on;



% Part (b)

figure;
eyediagram(y, 20, 10);
title('Eye Diagram of Line Code y(t)');