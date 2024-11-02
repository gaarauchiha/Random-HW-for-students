function linearShooting

%Solves the BVP y'' = p(x)y' + q(x)y + r(x), for a<x<b, with the boundary
%conditions y(a)=alpha and y(b)=beta.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%INPUTS.  Change these to adjust for the problem you are solving.

a = 0;  b = 720;             %the endpoints of the interval, a<x<b.
h = 12;                    %space between points on x axis.
alpha = 0;  beta = 0;       %boundary values.  y(a)=alpha, y(b)=beta.
E = 5e7;                   % Modulus of Elasticity in lb/in^2
I = 60;                    % Moment of Inertia in in^4
Q = 50;                    % Uniform Load Intensity in lb/in
L = 720;
S = 900;                   % End Stress in lb
p = @(x) S / (E * I);    %continuous function
q = @(x) 0;      %positive continuous function
r = @(x) Q * x * (x - L) / (2 * E * I);       %continuous function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Main part of the code.  Solves numerically the two IVP systems with
%ode45, and then combines the results to form the solution y to the BVP.

t = a:h:b;

[ ~, y1 ] = ode45( @odefun1, t, [alpha,0] );
[ ~, y2 ] = ode45( @odefun2, t,     [0,1] );

y1 = y1(:,1);  y2 = y2(:,1);

y = y1 + (beta-y1(end)) / y2(end) * y2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Plots the numerical solution y

figure(1), clf, hold('on')
plot( t, y, 'k', 'lineWidth', 2 )
plot( t, y, 'k.', 'markerSize', 20 )
set( gca, 'fontSize', 15 )
xlabel('x'), ylabel('y(x)')
grid('on')
title('Deflection of a Uniform Beam with doubled L')
drawnow, hold('off')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%The two ODE functions that are passed into ode45

    function u = odefun1(t,y)
        u = zeros(2,1);
        u(1) = y(2);
        u(2) = p(t)*y(2) + q(t)*y(1) + r(t);
    end

    function u = odefun2(t,y)
        u = zeros(2,1);
        u(1) = y(2);
        u(2) = p(t)*y(2) + q(t)*y(1);
    end

end