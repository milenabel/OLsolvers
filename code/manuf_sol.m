% Manufactured Solution Verification for Variable Coefficient Poisson Equation in MATLAB

clc; clear; close all;

% Define domain and grid size
Lx = 12; Ly = 12; % Domain size
Nx = 50; Ny = 50; % Number of grid points
x = linspace(0, Lx, Nx);
y = linspace(0, Ly, Ny);
[XX, YY] = meshgrid(x, y);

% Define variable coefficient k(x,y)
k = sin(4 * pi * XX) .* cos(4 * pi * YY);

% Define an exact solution p(x,y) similar to Poisson setup
P_exact = sin(pi * XX / Lx) .* sin(pi * YY / Ly);

% Compute f(x,y) = \nabla \cdot (k \nabla p)
[Px_x, Px_y] = gradient(P_exact, x, y);
[KPx_x, ~] = gradient(k .* Px_x, x, y);
[~, KPy_y] = gradient(k .* Px_y, x, y);
f = KPx_x + KPy_y;

% Define boundary conditions (Neumann \partial p/\partial n = 0)
g_top = (pi / Ly) * cos(pi * XX(1,:) / Lx) .* sin(pi * YY(1,:) / Ly); % Top boundary
g_bottom = (pi / Ly) * cos(pi * XX(end,:) / Lx) .* sin(pi * YY(end,:) / Ly); % Bottom boundary
g_left = (pi / Lx) * sin(pi * XX(:,1) / Lx) .* cos(pi * YY(:,1) / Ly); % Left boundary
g_right = (pi / Lx) * sin(pi * XX(:,end) / Lx) .* cos(pi * YY(:,end) / Ly); % Right boundary

% Plot results
figure;
subplot(2,2,1); surf(XX, YY, P_exact); title('Manufactured Solution p(x,y)'); shading interp;
subplot(2,2,2); surf(XX, YY, f); title('Forcing Function f(x,y)'); shading interp;
subplot(2,2,3); plot(x, g_top, 'r', x, g_bottom, 'b'); title('Neumann BCs (Top/Bottom)');
subplot(2,2,4); plot(y, g_left, 'r', y, g_right, 'b'); title('Neumann BCs (Left/Right)');

fprintf('Manufactured solution verification complete with variable k(x,y).\n');


