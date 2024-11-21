% Define matrices
A = [-1, -8, -16; 2/3, 0, 5/3; 0, 2, 3];
B = [2; 1; -0.5];

% (a) Check controllability
C = ctrb(A, B);
controllability_rank = rank(C);

if controllability_rank == size(A, 1)
    disp('The system is controllable.');
else
    disp('The system is not controllable.');
end

% (b) Diagonalize A if possible
[eigVectors, eigValues] = eig(A);

% Check if A is diagonalizable
if rank(eigVectors) == size(A, 1)
    disp('A is diagonalizable.');
    P = eigVectors;
    A_diag = inv(P) * A * P;
else
    disp('A is not diagonalizable.');
end

% (c) Transform B
B_transformed = inv(P) * B;

% Display results
disp('Diagonal matrix A:');
disp(A_diag);
disp('Transformed B:');
disp(B_transformed);


% Define system matrices
A = [-1, -8, -16;
     2/3, 0, 5/3;
     0, 2, 3];
B = [2;
     1;
     -0.5];

% Controllability matrix
C = ctrb(A, B);

% Calculate rank
rank_C = rank(C);

% Null space of C (uncontrollable part)
null_space = null(C, 'r');

% Controllable part
controllable_part = orth(C);

fprintf('Uncontrollable Basis:\n');
disp(null_space);

fprintf('\nControllable Basis:\n');
disp(controllable_part);

% Feedback gain for desired eigenvalues
desired_eigenvalues = [-1, -2, -3];

% Place poles
K = place(A, B, desired_eigenvalues);

fprintf('\nFeedback Gain K:\n');
disp(K);