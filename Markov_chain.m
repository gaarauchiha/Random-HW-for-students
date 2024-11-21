% Part 1
P = [1/5, 1/5, 1/5, 1/5, 1/5, 0,   0;
     0,   1/4, 1/4, 1/4, 1/4, 0,   0;
     0,   0,   0,   1/3, 1/3, 1/3, 0;
     0,   0,   0,   0,   1/3, 1/3, 1/3;
     1/2, 0,   0,   0,   0,   0,   1/2;
     0,   1/2, 0,   0,   0,   1/2, 0;
     1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7];

P_transpose = P';
[eigenvectors, eigenvalues] = eig(P_transpose);

% Eigenvector of eigenvalue 1
[~, idx] = max(diag(eigenvalues)); % Index of eigenvalue 1
pi = eigenvectors(:, idx);     

% Sum(Pi) = 1
pi = pi / sum(pi);

disp('Stationary distribution (Pi) = ');
disp(pi)

% Part 2
p0 = [1/8, 1/4, 1/8, 0, 1/8, 3/8, 0];

% p2
p2 = p0 * (P^2);
disp('p2 = ');
disp(p2);

% p4
p4 = p0 * (P^4);
disp('p4 = ');
disp(p2);

% P(X2 = 3), P(X4 = 3)
P_X2_equals_3 = p2(3); 
P_X4_equals_3 = p4(3); 

disp('P(X2 = 3) = ');
disp(P_X2_equals_3);

disp('P(X4 = 3) = ');
disp(P_X4_equals_3);



% Part 3
P_Xn_equals_3 = pi(3);

disp(' P(Xn = 3) as n -> inf = ');
disp(P_Xn_equals_3);