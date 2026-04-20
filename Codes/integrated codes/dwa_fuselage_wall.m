function plot_dwa_fuselage_wall()
% ISOLATED VISUALIZATION: THE 2.5D DWA FUSELAGE "WALL"
% Generates a clean 3D figure showing how the DWA algorithm 
% perceives the aircraft fuselage as an impenetrable boundary.

clc; clear; close all;

%% 1. Hangar Dimensions
hangar_length = 50; 
hangar_width = 50;  
hangar_height = 20;

%% 2. Load and Shift Aircraft
fprintf('Loading aircraft CAD...\n');
try
    model = stlread('airplane.STL');
    original_points = model.Points;
    min_coords = min(original_points);
    shifted_points = original_points - min_coords + [10, 10, 0];
catch
    error('airplane.STL not found in the current directory.');
end

%% 3. Define the DWA Fuselage Obstacles (The "Wall")
obs_x = linspace(10, 45, 15)'; 
obs_y = ones(15, 1) * 25;      
obs_r = ones(15, 1) * 2.5;     
obs_2d = [obs_x, obs_y, obs_r];

%% 4. Create the Figure
figure('Name', 'DWA Fuselage Obstacles', 'Color', 'w', 'Position', [150, 150, 1000, 750]); 
hold on;

% Plot Aircraft (Semi-transparent so we can see the rings)
patch('Faces', model.ConnectivityList, 'Vertices', shifted_points, ...
    'FaceColor', [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);

% Plot the Red Obstacle Rings
th = linspace(0, 2*pi, 40);
for i = 1:size(obs_2d, 1)
    cx = obs_2d(i,1);
    cy = obs_2d(i,2);
    cr = obs_2d(i,3);
    
    % Draw the main thick ring at cabin height
    plot3(cx + cr*cos(th), cy + cr*sin(th), ones(1,40) * 5, 'r-', 'LineWidth', 2.0);
    
    % Draw thinner helper rings to visually create a "wall" effect
    plot3(cx + cr*cos(th), cy + cr*sin(th), ones(1,40) * 8, 'r-', 'LineWidth', 1.0, 'Color', [1 0 0 0.5]);
    plot3(cx + cr*cos(th), cy + cr*sin(th), ones(1,40) * 2, 'r-', 'LineWidth', 1.0, 'Color', [1 0 0 0.5]);
    
    % Draw a faint vertical line connecting them to show it's a solid block
    plot3([cx, cx], [cy+cr, cy+cr], [2, 8], 'r--', 'Color', [1 0 0 0.3]);
    plot3([cx, cx], [cy-cr, cy-cr], [2, 8], 'r--', 'Color', [1 0 0 0.3]);
end

% 5. Lighting, View, and Formatting
camlight('headlight'); material('shiny'); lighting('gouraud');
xlim([0, hangar_length]); ylim([0, hangar_width]); zlim([0, hangar_height]);
daspect([1 1 1]); view(45, 30); grid on; rotate3d on;

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (Altitude)');
title('DWA - The Fuselage Keep-Out Zone', 'FontSize', 16, 'FontWeight', 'bold');

hold off;
end