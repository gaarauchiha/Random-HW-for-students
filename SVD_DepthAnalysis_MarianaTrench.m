depth = csvread('mariana_depth.csv');
latitude = csvread('mariana_latitude.csv');
longitude = csvread('mariana_longitude.csv');

% Convert to km
depth_km = depth / 1000;

longitude = longitude(:); 
latitude = latitude(:); 

[lonGrid, latGrid] = meshgrid(longitude, latitude);

depth_km = depth_km';

figure;
imagesc(longitude, latitude, depth_km);
colormap('jet');
colorbar;
xlabel('Longitude');
ylabel('Latitude');
title('Mariana Trench Depth Plot');
set(gca, 'YDir', 'normal');

figure;
contour(lonGrid, latGrid, depth_km, -11:1:11);
clabel(contour(lonGrid, latGrid, depth_km, -11:1:11));
colormap('jet');
colorbar;
xlabel('Longitude');
ylabel('Latitude');
title('Contour Map of the Mariana Trench');
set(gca, 'YDir', 'normal');



[deepest_depth, index] = min(depth(:));
[deepest_lat_idx, deepest_lon_idx] = ind2sub(size(depth), index);
deepest_latitude = latitude(deepest_lat_idx);
deepest_longitude = longitude(deepest_lon_idx);

fprintf('Deepest point is =  %.2f meters\n', deepest_depth);
fprintf('Latitude is = %.2f\n', deepest_latitude);
fprintf('Longitude is = %.2f\n', deepest_longitude);


ocean_floor_depth = -6000;
ocean_floor_depth_km = -6000/1000;
deeper_points = depth(depth < ocean_floor_depth);

average_depth = mean(deeper_points);

fprintf('Average depth below nominal ocean f;loor is = %.2f meters\n', average_depth);


A = depth;
N = size(A, 2);
u = rand(N, 1);
u = u / norm(u);

tolerance = 1e-6;
difference = inf;
iteration = 0;

while difference > tolerance
    iteration = iteration + 1;
    u_new = (A' * A) * u;
    u_new = u_new / norm(u_new);
    difference = norm(u_new - u);
    u = u_new;
end

V1 = u;

eigenvalue = (u' * (A' * A) * u);

figure;
plot(1:N, V1, '-o');
xlabel('Component Index');
ylabel('Value');
title('First Eigenvector of A^T A');


% Gram-Schmidt
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

figure;
semilogy(1:num_eigenvectors, eigenvalues, '-o');
xlabel('Eigenvector Index');
ylabel('Eigenvalue');
title('Eigenvalues of A^T A (Gram-Schmidt');




Sigma = diag(sqrt(eigenvalues));

U = A * V;
for i = 1:size(Sigma, 1)
    U(:, i) = U(:, i) / Sigma(i, i);
end

VT = V';

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

num_elements_A = numel(A);
num_elements_U = numel(U);
num_elements_Sigma = numel(Sigma);
num_elements_V = numel(V);
total_elements_SVD = num_elements_U + num_elements_Sigma + num_elements_V;

fprintf('Total elements in A is equal to =  %d\n', num_elements_A);
fprintf('Total elements in U, Sigma and V is equal to = %d\n', total_elements_SVD);


nnz_A = nnz(A);
nnz_U = nnz(U);
nnz_Sigma = nnz(Sigma);
nnz_V = nnz(V);
total_nnz_SVD = nnz_U + nnz_Sigma + nnz_V;

fprintf('Nonzero elements in A is = %d\n', nnz_A);
fprintf('Nonzero elements in U, Sigma and V =  %d\n', total_nnz_SVD);


A_approx = U * Sigma * VT;

figure;
imagesc(longitude, latitude, A_approx / 1000);
colormap('jet');
colorbar;
xlabel('Longitude');
ylabel('Latitude');
title('Reconstructed depth plot of Mariana Trench');
set(gca, 'YDir', 'normal');



[deepest_depth_approx, index_approx] = min(A_approx(:));
[deepest_lat_idx_approx, deepest_lon_idx_approx] = ind2sub(size(A_approx), index_approx);
deepest_latitude_approx = latitude(deepest_lat_idx_approx);
deepest_longitude_approx = longitude(deepest_lon_idx_approx);

fprintf('Deepest point (reconstructed) is = %.2f meters\n', deepest_depth_approx);
fprintf('Latitude is =  %.2f\n', deepest_latitude_approx);
fprintf('Longitude is = %.2f\n', deepest_longitude_approx);


deeper_points_approx = A_approx(A_approx < -ocean_floor_depth);
average_depth_approx = mean(deeper_points_approx);

fprintf('Average depth below ocean floor in reconstructed form is =: %.2f meters\n', average_depth_approx);



k = 10;
U_k = U(:, 1:k);
Sigma_k = Sigma(1:k, 1:k);
VT_k = VT(1:k, :);

A_approx_k = U_k * Sigma_k * VT_k;

figure;
imagesc(longitude, latitude, A_approx_k / 1000);
colormap('jet');
colorbar;
xlabel('Longitude');
ylabel('Latitude');
title('Reconstructed Depth with 10 Singular Values plot');
set(gca, 'YDir', 'normal');

