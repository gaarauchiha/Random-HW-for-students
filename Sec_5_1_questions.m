% Q12 x' = y, y' = x
[x, y] = meshgrid(-2:0.2:2, -2:0.2:2);
dx = y; 
dy = x;
quiver(x, y, dx, dy); 
xlabel('x'); ylabel('y');
title('Direction Field for x'' = y, y'' = x');
hold on;

t = linspace(-2, 2, 100);
for A = -2:1:2
    for B = -2:1:2
        x_sol = A * exp(t) + B * exp(-t);
        y_sol = A * exp(t) - B * exp(-t);
        plot(x_sol, y_sol, 'r');
    end
end
hold off;
%% 

% Q13 x' = -2y, y' = 2x
[x, y] = meshgrid(-2:0.2:2, -2:0.2:2); 
dx = -2 * y; 
dy = 2 * x; 
quiver(x, y, dx, dy);
xlabel('x'); ylabel('y');
title('Direction Field for x'' = -2y, y'' = 2x');
hold on;

t = linspace(0, 2*pi, 100);
x_sol = cos(2*t);
y_sol = sin(2*t);
plot(x_sol, y_sol, 'r', 'LineWidth', 2);
hold off;

%% 
% Q14 x' = 10y, y' = -10x
[x, y] = meshgrid(-5:0.5:5, -5:0.5:5); 
dx = 10 * y;
dy = -10 * x; 
quiver(x, y, dx, dy); 
xlabel('x'); ylabel('y');
title('Direction Field for x'' = 10y, y'' = -10x');
hold on;

t = linspace(0, 2*pi, 100);
x_sol = 3*cos(10*t) + 4*sin(10*t); % x(t)
y_sol = -3*sin(10*t) + 4*cos(10*t); % y(t)
plot(x_sol, y_sol, 'r', 'LineWidth', 2);
hold off;

%% 

% Q15 x' = y/2, y' = -8x
[x, y] = meshgrid(-5:0.2:5, -5:0.2:5);

dx = y / 2; 
dy = -8 * x; 

figure;
quiver(x, y, dx, dy, 'b'); 
xlabel('x'); ylabel('y');
title('Vector Field and Solution Curve for Problem 15');
axis equal;
hold on;

c1 = 1; 
c2 = 1; 

t = linspace(0, 2*pi, 500);

x_sol = (1/4) * c2 * sin(2*t) + c1 * cos(2*t);
y_sol = c2 * cos(2*t) - 4 * c1 * sin(2*t);

plot(x_sol, y_sol, 'r', 'LineWidth', 2); 
legend('Vector Field', 'Solution Curve');
hold off;







