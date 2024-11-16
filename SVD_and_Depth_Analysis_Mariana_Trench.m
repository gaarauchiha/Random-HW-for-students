% Import data
depth = csvread('mariana_depth.csv');
latitude = csvread('mariana_latitude.csv');
longitude = csvread('mariana_longitude.csv');

% Convert depths to kilometers
depth_km = depth / 1000;

% Ensure longitude and latitude are vectors
longitude = longitude(:); % Column vector
latitude = latitude(:); % Column vector

% Create meshgrid for contour plotting
[lonGrid, latGrid] = meshgrid(longitude, latitude);

% Transpose depth_km to match grid orientation
depth_km = depth_km';

% Display as an image
figure;
imagesc(longitude, latitude, depth_km);
colormap('jet');
colorbar;
xlabel('Longitude');
ylabel('Latitude');
title('Mariana Trench Depth (km)');
set(gca, 'YDir', 'normal');

% Display as a contour map
figure;
contour(lonGrid, latGrid, depth_km, -11:1:11);
clabel(contour(lonGrid, latGrid, depth_km, -11:1:11));
colormap('jet');
colorbar;
xlabel('Longitude');
ylabel('Latitude');
title('Mariana Trench Contour Map (km)');
set(gca, 'YDir', 'normal');


%% 


% Find the deepest point
[deepest_depth, index] = min(depth(:));
[deepest_lat_idx, deepest_lon_idx] = ind2sub(size(depth), index);
deepest_latitude = latitude(deepest_lat_idx);
deepest_longitude = longitude(deepest_lon_idx);

fprintf('Deepest point: %.2f meters\n', deepest_depth);
fprintf('Latitude: %.2f\n', deepest_latitude);
fprintf('Longitude: %.2f\n', deepest_longitude);

%% 


% Define ocean floor depth in meters
ocean_floor_depth = -6000;
ocean_floor_depth_km = -6000/1000;
% Find all depths greater than ocean floor
deeper_points = depth(depth < ocean_floor_depth);

% Calculate mean depth of these points
average_depth = mean(deeper_points);

fprintf('Average depth below 6 km: %.2f meters\n', average_depth);

%% 

A = depth;

% Initialize a random unit vector of appropriate size
N = size(A, 2);
u = rand(N, 1);
u = u / norm(u);

% Set a tolerance level
tolerance = 1e-6;
difference = inf;
iteration = 0;
%% 


% Power iteration method to find the first eigenvector
while difference > tolerance
    iteration = iteration + 1;
    u_new = (A' * A) * u;
    u_new = u_new / norm(u_new);
    difference = norm(u_new - u);
    u = u_new;
end

% The eigenvector is u
V1 = u;

% Calculate the corresponding eigenvalue
eigenvalue = (u' * (A' * A) * u);

% Plot the eigenvector
figure;
plot(1:N, V1, '-o');
xlabel('Component Index');
ylabel('Value');
title('First Eigenvector of A^T A');

%% 


% Compute the i largest eigenvalues and eigenvectors using Gram-Schmidt
num_eigenvectors = 50;
V = zeros(N, num_eigenvectors);
eigenvalues = zeros(num_eigenvectors, 1);

for i = 1:num_eigenvectors
    u = rand(N, 1);
    u = u / norm(u);
    
    difference = inf;
    
    while difference > tolerance
        u_new = (A' * A) * u;
        
        for j = 1:i-1
            u_new = u_new - (u_new' * V(:, j)) * V(:, j);
        end
        
        u_new = u_new / norm(u_new);
        difference = norm(u_new - u);
        u = u_new;
    end
    
    V(:, i) = u;
    eigenvalues(i) = (u' * (A' * A) * u);
end

% Plot the eigenvalues
figure;
semilogy(1:num_eigenvectors, eigenvalues, '-o');
xlabel('Eigenvector Index');
ylabel('Eigenvalue');
title('Eigenvalues of A^T A');

%% 


% Construct the Sigma matrix
Sigma = diag(sqrt(eigenvalues));

% Calculate the U matrix
U = A * V;
for i = 1:size(Sigma, 1)
    U(:, i) = U(:, i) / Sigma(i, i);
end

% Calculate V transpose
VT = V';

% Display matrices using spy
figure;
subplot(1, 3, 1);
spy(U);
title('U Matrix');

subplot(1, 3, 2);
spy(Sigma);
title('\Sigma Matrix');

subplot(1, 3, 3);
spy(VT);
title('V^T Matrix');

%% 


% Calculate the number of elements to save
num_elements_A = numel(A);
num_elements_U = numel(U);
num_elements_Sigma = numel(Sigma);
num_elements_V = numel(V);
total_elements_SVD = num_elements_U + num_elements_Sigma + num_elements_V;

fprintf('Total elements in A: %d\n', num_elements_A);
fprintf('Total elements in U, Sigma, and V: %d\n', total_elements_SVD);

%% 


% Calculate the number of nonzero elements
nnz_A = nnz(A);
nnz_U = nnz(U);
nnz_Sigma = nnz(Sigma);
nnz_V = nnz(V);
total_nnz_SVD = nnz_U + nnz_Sigma + nnz_V;

fprintf('Nonzero elements in A: %d\n', nnz_A);
fprintf('Nonzero elements in U, Sigma, and V: %d\n', total_nnz_SVD);

%% 


% Reconstruct the matrix A_approx using the incomplete SVD
A_approx = U * Sigma * VT;

% Display the reconstructed matrix as an image
figure;
imagesc(longitude, latitude, A_approx / 1000);
colormap('jet');
colorbar;
xlabel('Longitude');
ylabel('Latitude');
title('Reconstructed Mariana Trench Depth (km)');
set(gca, 'YDir', 'normal');

%% 




%% 


% Calculate the maximum depth in the reconstructed matrix
[deepest_depth_approx, index_approx] = min(A_approx(:));
[deepest_lat_idx_approx, deepest_lon_idx_approx] = ind2sub(size(A_approx), index_approx);
deepest_latitude_approx = latitude(deepest_lat_idx_approx);
deepest_longitude_approx = longitude(deepest_lon_idx_approx);

fprintf('Deepest point (reconstructed): %.2f meters\n', deepest_depth_approx);
fprintf('Latitude: %.2f\n', deepest_latitude_approx);
fprintf('Longitude: %.2f\n', deepest_longitude_approx);

%% 


% Calculate the mean depth below 6 km in the reconstructed matrix
deeper_points_approx = A_approx(A_approx < -ocean_floor_depth);
average_depth_approx = mean(deeper_points_approx);

fprintf('Average depth below 6 km (reconstructed): %.2f meters\n', average_depth_approx);

%% 


% Use only the first 10 columns of U, Sigma, and V
k = 10;
U_k = U(:, 1:k);
Sigma_k = Sigma(1:k, 1:k);
VT_k = VT(1:k, :);

% Reconstruct the matrix with fewer singular values
A_approx_k = U_k * Sigma_k * VT_k;

% Display the reduced matrix as an image
figure;
imagesc(longitude, latitude, A_approx_k / 1000);
colormap('jet');
colorbar;
xlabel('Longitude');
ylabel('Latitude');
title('Reconstructed Depth with 10 Singular Values (km)');
set(gca, 'YDir', 'normal');

