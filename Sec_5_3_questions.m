% Q3 x' = -3x + 2y, y' = -3x + 4y
[x, y] = meshgrid(-10:1:10, -10:1:10);
dx = -3*x + 2*y; 
dy = -3*x + 4*y; 

figure;
quiver(x, y, dx, dy, 'b'); 
xlabel('x'); ylabel('y'); title('Direction Field for Problem 3');
axis equal;
hold on;

t = linspace(0, 0.5, 50);
x_sol = (-4/5)*exp(-2*t) + (4/5)*exp(3*t); 
y_sol = (-2/5)*exp(-2*t) + (12/5)*exp(3*t); 
plot(x_sol, y_sol, 'r', 'LineWidth', 2); 
legend('Direction Field', 'Solution Curve');
hold off;

%% 

% Q7 x' = 4x + y + 2t, y' = -2x + y
clc; clear;

tspan = [0, 10];
z0 = [1, 0]; 

f = @(t, z) [4*z(1) + z(2) + 2*t; -2*z(1) + z(2)];

[t, sol] = ode45(f, tspan, z0);

[x, y] = meshgrid(-5:0.5:5, -5:0.5:5); 
dx = 4*x + y; 
dy = -2*x + y; 
L = sqrt(dx.^2 + dy.^2)
dx = dx ./ L;
dy = dy ./ L;

figure;
quiver(x, y, dx, dy, 'r'); hold on;
xlabel('x'); ylabel('y');
title('Direction Field and Solution Curve for Problem 7');

plot(sol(:,1), sol(:,2), 'b', 'LineWidth', 2);
legend('Direction Field', 'Solution Curve');
grid on;


%% 

% Q9 x' = 2x - 3y + 2sin(2t), y' = x - 2y - cos(2t)

tspan = [0, 10];
z0 = [1, 0]; 

f = @(t, z) [2*z(1) - 3*z(2) + 2*sin(2*t); z(1) - 2*z(2) - cos(2*t)];

[t, sol] = ode45(f, tspan, z0);

[x, y] = meshgrid(-5:0.5:5, -5:0.5:5); 
dx = 2*x - 3*y; 
dy = x - 2*y;
L = sqrt(dx.^2 + dy.^2); 
dx = dx ./ L;
dy = dy ./ L;

figure;
quiver(x, y, dx, dy, 'r'); hold on;
xlabel('x'); ylabel('y');
title('Direction Field and Solution Curve for Problem 9');
hold on;


tspan = [0, 10];
z0 = [1, 0]; 

f = @(t, z) [2*z(1) - 3*z(2) + 2*sin(2*t); z(1) - 2*z(2) - cos(2*t)];

[t, sol] = ode45(f, tspan, z0);

[x, y] = meshgrid(-5:0.5:5, -5:0.5:5); 
dx = 2*x - 3*y;
dy = x - 2*y; 
L = sqrt(dx.^2 + dy.^2); 
dx = dx ./ L;
dy = dy ./ L;

figure;
quiver(x, y, dx, dy, 'r'); hold on;
xlabel('x'); ylabel('y');
title('Direction Field and Solution Curve for Problem 9');





