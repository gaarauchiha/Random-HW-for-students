% Define the transition probability matrix P
P = [1/5 1/5 1/5 1/5 1/5 0   0;
     0   1/4 1/4 1/4 1/4 0   0;
     0   0   0   1/3 1/3 1/3 0;
     0   0   0   0   1/3 1/3 1/3;
     1/2 0   0   0   0   0   1/2;
     0   1/2 0   0   0   1/2 0;
     1/7 1/7 1/7 1/7 1/7 1/7 1/7];

% Find the eigenvectors and eigenvalues of P'
[V, D] = eig(P');

% Find the eigenvector corresponding to eigenvalue 1
% (This is the stationary distribution ?)
[eigenvalue, idx] = min(abs(diag(D) - 1)); % Find eigenvalue closest to 1
pi = V(:, idx); % Corresponding eigenvector

% Normalize the eigenvector to make it a probability distribution
pi = pi / sum(pi);

% Display the stationary distribution
disp('Stationary distribution (Pi):');
disp(pi');