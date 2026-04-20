% AIRCRAFT HANGAR INSPECTION: FORMULATED HUNGARIAN ALLOCATION & TSP
clc; clear; close all;

%% CONFIGURABLE HANGAR & WEIGHTS
hangar_length = 50; 
hangar_width = 50;  
hangar_height = 20;  

% Conversion Logic
w_dist = 1.0;
w_battery = 1.0;

%% LOAD AND SHIFT AIRCRAFT
model = stlread('airplane.STL');
original_points = model.Points;

min_coords = min(original_points);
shifted_points = original_points - min_coords;
offset = [10, 10, 0]; 
shifted_points = shifted_points + offset;

%% FILTER FOR TOP SURFACE
TR = triangulation(model.ConnectivityList, shifted_points);
normals = faceNormal(TR);
incenters = incenter(TR);

z_threshold = 0.25 * max(shifted_points(:,3)); 
valid_mask = (normals(:,3) > 0.2) & (incenters(:,3) > z_threshold);
valid_face_indices = find(valid_mask);
valid_points = incenters(valid_face_indices, :);

%% GENERATE FOIs (AERODYNAMIC SURFACE ANALYSIS)
y_min = min(valid_points(:,2)); y_max = max(valid_points(:,2));
y_center = (y_min + y_max) / 2; 
fuselage_margin = (y_max - y_min) * 0.125; 

% Standard Zone Filtering
left_wing_pts  = valid_points(valid_points(:,2) < y_center - fuselage_margin, :);
right_wing_pts = valid_points(valid_points(:,2) > y_center + fuselage_margin, :);
fuselage_pts   = valid_points(valid_points(:,2) >= y_center - fuselage_margin & ...
                              valid_points(:,2) <= y_center + fuselage_margin, :);

get_centroids = @(pts, k) kmeans(pts, k, 'Replicates', 3);

% Automated Clustering for Primary Surfaces
[~, C_left]  = get_centroids(left_wing_pts, 3); 
[~, C_right] = get_centroids(right_wing_pts, 4);
[~, C_fuse]  = get_centroids(fuselage_pts, 4);

target_removal_zone = [20, 25, 6];
dist_to_target = sum((C_fuse - target_removal_zone).^2,2);
[~, remove_idx] = min(dist_to_target);
C_fuse(remove_idx, :) = [];

% --- AUTOMATED EMPENNAGE DETECTION ---
% Logic: Filter for the rear-most section (low X) and isolate the horizontal stabilizers
tail_section_pts = valid_points(valid_points(:,1) < (min(valid_points(:,1)) + 5), :);
[~, left_tail_idx]  = min(tail_section_pts(:,2)); % Outboard Left Stabilizer
[~, right_tail_idx] = max(tail_section_pts(:,2)); % Outboard Right Stabilizer

C_empennage = [tail_section_pts(left_tail_idx, :); tail_section_pts(right_tail_idx, :)];

% Final Coordinate Integration (Total: 13)
[~, max_x_idx] = max(valid_points(:,1));
nose_point = valid_points(max_x_idx, :);

all_targets = [C_left; C_right; C_fuse; C_empennage; nose_point];

% Snap to surface for structural fidelity
final_fois = zeros(size(all_targets, 1), 3);
for i = 1:size(all_targets, 1)
    dists = sum((valid_points - all_targets(i,:)).^2, 2);
    [~, min_idx] = min(dists);
    final_fois(i,:) = valid_points(min_idx, :);
end

%% DEFINE AGENTS
num_drones = 5;
drones_pos = [0,  0, 0;
            50,  0, 0;
            0, 50, 0;   
            50, 50, 0;   
            25,  0, 0];
drone_battery = [80; 80; 83; 85; 79]; 

colors = lines(num_drones);

%% CLUSTERING
[foi_cluster_idx, cluster_centers] = kmeans(final_fois, num_drones, 'Start', drones_pos);

%% COST MATRIX (HUNGARIAN FORMULATION)
CostMatrix = zeros(num_drones, num_drones);
for i = 1:num_drones
    for j = 1:num_drones 
        % Formula: J_total = (w_dist * d_ij) + (w_batt * (100 - B_i))
        dist = norm(drones_pos(i,:) - cluster_centers(j,:));
        battery_penalty = (100 - drone_battery(i));
        CostMatrix(i, j) = (w_dist * dist) + (w_battery * battery_penalty);
    end
end

assignments = matchpairs(CostMatrix, 1000);

%% VISUALIZATION SETUP
figure('Name', 'Task Allocation', 'Color', 'w');
hold on;

patch('Faces', model.ConnectivityList, 'Vertices', shifted_points, ...
    'FaceColor', [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
camlight('headlight'); material('shiny'); lighting('gouraud');

xlim([0, hangar_length]); ylim([0, hangar_width]); zlim([0, hangar_height]);
daspect([1 1 1]); view(0, 90); grid on; rotate3d on;
xlabel('X (Length)'); ylabel('Y (Width)'); zlabel('Z (Height)');
title('Task Allocation');

% VORONOI TERRITORY VISUALIZATION (FIXED)
[X_grid, Y_grid] = meshgrid(0:1:hangar_length, 0:1:hangar_width);
voronoi_idx = zeros(size(X_grid));

% Assign each grid point to the nearest drone base (Voronoi Logic)
for i = 1:size(X_grid, 1)
    for j = 1:size(X_grid, 2)
        curr_p = [X_grid(i,j), Y_grid(i,j)];
        dists_to_bases = sqrt(sum((drones_pos(:,1:2) - curr_p).^2, 2));
        [~, nearest_base] = min(dists_to_bases);
        voronoi_idx(i,j) = nearest_base;
    end
end

% Plot the dotted boundaries on the floor
contour(X_grid, Y_grid, voronoi_idx, 'k:', 'LineWidth', 1.2, ...
    'DisplayName', 'Geographic Voronoi Limits');

% Plot semi-transparent floor regions matching Agent Colors
for d = 1:num_drones
    % FIX: Convert logical mask to double for CData compatibility
    mask_numeric = double(voronoi_idx == d);
    mask_numeric(mask_numeric == 0) = NaN; % Make non-region areas invisible
    
    surf(X_grid, Y_grid, zeros(size(X_grid)) - 0.1, ... % Slightly below 0 to avoid flickering
        'FaceColor', colors(d,:), ...
        'EdgeColor', 'none', ...
        'FaceAlpha', 0.1, ... % Very light transparency
        'HandleVisibility', 'off');
end

%% TSP ROUTING & COST RECORDING
colors = lines(num_drones); 
total_system_flight_dist = 0;
total_assignment_cost = 0;

fprintf('HUNGARIAN COST REPORT');

for k = 1:size(assignments, 1)
    d_id = assignments(k, 1);
    c_id = assignments(k, 2); 
    
    task_cost = CostMatrix(d_id, c_id);
    total_assignment_cost = total_assignment_cost + task_cost;
    
    home_pos = drones_pos(d_id, :);
    assigned_fois = final_fois(foi_cluster_idx == c_id, :);
    num_assigned = size(assigned_fois, 1);
    
    % Plot Base Station
    scatter3(home_pos(1), home_pos(2), home_pos(3), 150, colors(d_id,:), 's', 'filled', ...
        'DisplayName', sprintf('Agent %d Base', d_id));
    
    % Plot FOIs for this assignment
    if num_assigned > 0
        scatter3(assigned_fois(:,1), assigned_fois(:,2), assigned_fois(:,3), ...
            80, colors(d_id,:), 'filled', '^', 'HandleVisibility', 'off');
    end
    
    % TSP Calculation: J_total = d_ij + C_insp + d_RTL
    all_perms = perms(1:num_assigned);
    min_route_dist = inf;
    best_route = [];
    
    for p = 1:size(all_perms, 1)
        current_order = all_perms(p, :);
        route_dist = 0;
        current_pos = home_pos;
        
        for step = 1:num_assigned
            target_foi = assigned_fois(current_order(step), :);
            route_dist = route_dist + norm(target_foi - current_pos); % d_ij segment
            current_pos = target_foi;
        end
        route_dist = route_dist + norm(home_pos - current_pos); % d_RTL segment
        
        if route_dist < min_route_dist
            min_route_dist = route_dist;
            best_route = current_order;
        end
    end
    
    % Handle empty assignments
    if num_assigned == 0
        min_route_dist = 0;
    end
    
    total_system_flight_dist = total_system_flight_dist + min_route_dist;
    
    fprintf('Agent %d (Bat: %d%%) -> Cluster %d | FOIs: %d | Assign Cost: %5.2f | Flight Dist: %5.2f m\n', ...
        d_id, drone_battery(d_id), c_id, num_assigned, task_cost, min_route_dist);
        
    % Draw TSP Path
    if num_assigned > 0
        ordered_points = zeros(num_assigned, 3);
        for step = 1:num_assigned
            ordered_points(step, :) = assigned_fois(best_route(step), :);
        end

        full_path = [home_pos; ordered_points; home_pos];
        plot3(full_path(:,1), full_path(:,2), full_path(:,3), '-', ...
            'Color', colors(d_id, :), 'LineWidth', 2, 'DisplayName', sprintf('Agent %d Path', d_id));
    end
end

fprintf('TOTAL SYSTEM ASSIGNMENT COST: %.2f\n', total_assignment_cost);
fprintf('TOTAL SYSTEM FLIGHT DISTANCE: %.2f m\n', total_system_flight_dist);
legend('Location', 'eastoutside');
hold off;

%% TSP ROUTING, PATH STORAGE & SLOW ANIMATION
colors = lines(num_drones); 
total_system_flight_dist = 0;

% INITIALIZE the missing variable
all_agent_paths = cell(num_drones, 1);


for k = 1:size(assignments, 1)
    d_id = assignments(k, 1);
    c_id = assignments(k, 2); 
    home_pos = drones_pos(d_id, :);
    assigned_fois = final_fois(foi_cluster_idx == c_id, :);
    num_assigned = size(assigned_fois, 1);

    % TSP Calculation: J_total = d_ij + C_insp + d_RTL
    if num_assigned > 0
        all_perms = perms(1:num_assigned);
        min_route_dist = inf;
        best_route = [];

        for p = 1:size(all_perms, 1)
            current_order = all_perms(p, :);
            route_dist = 0;
            current_pos = home_pos;
            for step = 1:num_assigned
                target_foi = assigned_fois(current_order(step), :);
                route_dist = route_dist + norm(target_foi - current_pos);
                current_pos = target_foi;
            end
            route_dist = route_dist + norm(home_pos - current_pos);
            if route_dist < min_route_dist
                min_route_dist = route_dist;
                best_route = current_order;
            end
        end

        % POPULATE the variable: Home -> FOI sequence -> Home
        ordered_points = assigned_fois(best_route, :);
        all_agent_paths{d_id} = [home_pos; ordered_points; home_pos];
        total_system_flight_dist = total_system_flight_dist + min_route_dist;
    end
end

%% PARALLEL MISSION ANIMATION (WITH PRE-FLIGHT PAUSE)
figure('Name', 'Mission Simulation', 'Color', 'w'); hold on;

% DRAW STATIC ENVIRONMENT (Mesh, FOIs, and Voronoi)
patch('Faces', model.ConnectivityList, 'Vertices', shifted_points, ...
    'FaceColor', [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.2);
camlight('headlight'); lighting('gouraud');
xlim([0, 50]); ylim([0, 50]); zlim([0, 40]);
view(45, 30); grid on; axis equal;
xlabel('X (Hangar Length)'); ylabel('Y (Hangar Width)');
scatter3(final_fois(:,1), final_fois(:,2), final_fois(:,3), 40, 'r', 'filled', '^', 'HandleVisibility', 'off');

% INITIALIZE AGENT MARKERS AND PATH LINES
drone_markers = gobjects(num_drones, 1);
for d = 1:num_drones
    path = all_agent_paths{d};
    if ~isempty(path)
        drone_markers(d) = scatter3(path(1,1), path(1,2), path(1,3), 100, ...
            colors(d,:), 'o', 'filled', 'MarkerEdgeColor', 'k');
        plot3(path(:,1), path(:,2), path(:,3), ':', 'Color', colors(d,:), 'LineWidth', 1);
    end
end

title('Initializing Hangar & Task Clusters');
drawnow; 
pause(1.0);

% CALCULATE GLOBAL SIMULATION STEPS
num_frames_per_leg = 80; 
max_legs = 0;
for d = 1:num_drones
    if ~isempty(all_agent_paths{d})
        max_legs = max(max_legs, size(all_agent_paths{d}, 1) - 1);
    end
end

% SIMULTANEOUS ANIMATION LOOP
for leg = 1:max_legs
    for f = 1:num_frames_per_leg
        for d = 1:num_drones
            path = all_agent_paths{d};
            if ~isempty(path) && leg < size(path, 1)
                start_pt = path(leg, :);
                end_pt = path(leg+1, :);
                curr_pos = start_pt + (end_pt - start_pt) * (f / num_frames_per_leg);
                set(drone_markers(d), 'XData', curr_pos(1), 'YData', curr_pos(2), 'ZData', curr_pos(3));
            end
        end
        % Update title to show active mission
        title('Mission Active');
        drawnow limitrate;
        pause(0.01); 
    end
    pause(0.5); % Dwell at FOIs
end

title('Mission Success: All Agents Returned to Base');
hold off;

% STATIC HANGAR & FOI INITIALIZATION
figure('Name', 'Hangar & FOI Setup', 'Color', 'w');
hold on;

% Plot the Aircraft Mesh (Higher Opacity for Setup View)
patch('Faces', model.ConnectivityList, 'Vertices', shifted_points, ...
    'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.9);
camlight('headlight'); lighting('gouraud');

% Plot the 12 Optimized FOIs (Numbered for the Technical Report)
scatter3(final_fois(:,1), final_fois(:,2), final_fois(:,3), 100, 'r', 'filled', '^');
for i = 1:size(final_fois,1)
    text(final_fois(i,1), final_fois(i,2), final_fois(i,3)+1.5, num2str(i), ...
        'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
end

% Format the Environment
xlim([0, 50]); ylim([0, 50]); zlim([0, 40]);
view(45, 30); grid on; axis equal;
xlabel('X (Length)'); ylabel('Y (Width)'); zlabel('Z (Height)');
hold off;