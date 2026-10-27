function hangar_inspection_apf_cad()
% AIRCRAFT HANGAR INSPECTION: MULTI-AGENT PATH PLANNING & TASK ALLOCATION
% 
% This script simulates a multi-drone system for automated aircraft inspection
% inside a constrained hangar environment. The architecture is divided into:
% 1. Environment & Obstacle Generation (Aircraft CAD + Hangar Geometry)
% 2. Target Identification (Surface filtering for Features of Interest - FOIs)
% 3. Task Allocation (K-means clustering + Hungarian algorithm with battery/distance costs)
% 4. Path Planning (Traveling Salesperson Problem (TSP) for routing + 
%    Artificial Potential Fields (APF) for local collision avoidance)
% 5. Kinematic Simulation (3D Visualization with dynamic yaw/pitch/roll)

clc; clear; close all;

%% 1. CONFIGURABLE PARAMETERS & SYSTEM CONSTRAINTS
% Hangar bounds (meters) used to confine the operational airspace
hangar_length = 50; 
hangar_width = 50;  
hangar_height = 20;  

% Task Allocation Weights for the Cost Matrix
w_dist = 1.0;    % Weight penalizing distance from drone to cluster center
w_battery = 1.0; % Weight penalizing low battery levels

% Artificial Potential Field (APF) Parameters
params = struct();
params.attraction_gain = 1.2;      % Pull strength towards the current waypoint
params.repulsion_gain = 15.0;      % Push strength from obstacles
params.repulsion_distance = 2.0;   % Radius of influence for obstacle repulsion (m)
params.dt = 0.15;                  % Time step for integration
params.max_iterations = 1500;      % Timeout to prevent infinite loops in local minima
params.goal_threshold = 0.2;       % Distance (m) to consider a waypoint reached
params.escape_force = 0.5;         % Random perturbation magnitude to escape local minima
drone_radius = 0.75;               % Assumed physical radius of the drone (m)
min_waypoint_spacing = 0.5;        % Resolution for path simplification

%% 2. LOAD ENVIRONMENT CAD & GENERATE OBSTACLE MESHES
fprintf('Loading aircraft and generating APF obstacle field...\n');

% Load the primary inspection target (Aircraft CAD)
model = stlread('airplane.STL');
original_points = model.Points;
min_coords = min(original_points);
% Shift the aircraft to a central position within the hangar
shifted_points = original_points - min_coords + [10, 10, 0];
aircraft_faces = model.ConnectivityList;

% Generate bounding box obstacles for the APF local planner
% 1. Sample the aircraft surface to create discrete repulsive zones
obstacles_aircraft = generate_obstacle_field(shifted_points, aircraft_faces);
% 2. Generate static hangar infrastructure (pillars, equipment)
obstacles_hangar = generate_hangar_boxes();

% Concatenate all obstacles into a single environmental matrix
obstacles = [obstacles_aircraft; obstacles_hangar];

% Load drone CAD for visualization (fallback to markers if missing)
try
    drone_mesh = stlread('drone.stl');
    drone_scale = 0.1; 
    scaled_drone_pts = drone_mesh.Points * drone_scale;
    center_offset = mean(scaled_drone_pts);
    scaled_drone_pts = scaled_drone_pts - center_offset;
catch
    warning('drone.stl not found. Falling back to simple markers.');
    drone_mesh = [];
end

%% 3. SURFACE FILTERING & FOI REGISTRATION
% Map abstract Features of Interest (FOIs) to valid physical surfaces on the aircraft
TR = triangulation(model.ConnectivityList, shifted_points);
normals = faceNormal(TR); 
incenters = incenter(TR);

% Filter out the bottom of the aircraft (inaccessible/grounded areas)
z_threshold = 0.25 * max(shifted_points(:,3)); 
% Keep surfaces facing generally upwards/outwards and above the threshold
valid_mask = (normals(:,3) > 0.2) & (incenters(:,3) > z_threshold);
valid_points = incenters(valid_mask, :);

% User-defined global coordinates for inspection targets
user_defined_fois = [
    42.0, 25.0, 5.0;  45.0, 25.0, 4.0;  25.0, 25.0, 5.0;  
    25.0, 15.0, 2.5;  25.0, 10.0, 2.5;  27.5, 34.5, 2.5;
    27.5, 31.5, 2.5;  11.0, 28.0, 5.0;  15.0, 40.0, 11.0;
    11.0, 17.0, 5.0;  21.0, 18.0, 15.0; 13.0, 20.0, 3.0  
];

% Snap user-defined FOIs to the nearest valid mesh surface
final_fois = zeros(size(user_defined_fois, 1), 3);
for i = 1:size(user_defined_fois, 1)
    [~, min_idx] = min(sum((valid_points - user_defined_fois(i,:)).^2, 2));
    final_fois(i,:) = valid_points(min_idx, :);
end

%% 4. AGENT INITIALIZATION & HUNGARIAN TASK ALLOCATION
num_drones = 5;
% Initial home locations for the multi-agent system
drones_pos = [0, 0, 0; 50, 0, 0; 0, 50, 0; 50, 50, 0; 25, 0, 0];
drone_battery = [80; 80; 83; 85; 79]; % Simulated battery percentages
colors = lines(num_drones);

% Partition the FOIs into 'num_drones' operational clusters using K-means
[foi_cluster_idx, cluster_centers] = kmeans(final_fois, num_drones, 'Start', drones_pos);

% Formulate the Cost Matrix for the Linear Assignment Problem
CostMatrix = zeros(num_drones, num_drones);
for i = 1:num_drones
    for j = 1:num_drones 
        % Cost is a weighted sum of spatial distance and remaining battery capacity
        CostMatrix(i, j) = (w_dist * norm(drones_pos(i,:) - cluster_centers(j,:))) + ...
                           (w_battery * (100 - drone_battery(i)));
    end
end

% Solve task allocation using the Hungarian algorithm (matchpairs)
% Minimizes the global cost of assigning clusters to specific drones
assignments = matchpairs(CostMatrix, 1000);

%% 5. TSP ROUTING & APF LOCAL PATH PLANNING
fprintf('Calculating APF Collision-Free Direct Hops...\n');
all_agent_paths = cell(num_drones, 1);
transit_z_high = 8;     % Safe altitude for cross-hangar transit
hover_offset = 2;       % Standoff distance above the FOI for inspection

for k = 1:size(assignments, 1)
    d_id = assignments(k, 1); 
    c_id = assignments(k, 2); 
    home_pos = drones_pos(d_id, :);
    assigned_fois = final_fois(foi_cluster_idx == c_id, :);
    num_assigned = size(assigned_fois, 1);
    
    if num_assigned > 0
        % --- 1. Traveling Salesperson Problem (TSP) Optimization ---
        % Brute-force permutation to find the shortest visiting sequence
        % (Computationally feasible here due to low number of FOIs per agent)
        all_perms = perms(1:num_assigned);
        min_route_dist = inf; 
        best_route = [];
        
        for p = 1:size(all_perms, 1)
            current_order = all_perms(p, :); 
            route_dist = 0; 
            current_pos = home_pos;
            
            for step = 1:num_assigned
                route_dist = route_dist + norm(assigned_fois(current_order(step), :) - current_pos);
                current_pos = assigned_fois(current_order(step), :);
            end
            % Include return-to-home distance in the total route cost
            route_dist = route_dist + norm(home_pos - current_pos);
            
            if route_dist < min_route_dist
                min_route_dist = route_dist; 
                best_route = current_order;
            end
        end
        ordered_points = assigned_fois(best_route, :);
        
        % --- 2. Local Path Generation via Artificial Potential Fields ---
        full_path = home_pos;
        current = home_pos';
        
        % Ascend to safe transit altitude
        waypoint = [home_pos(1); home_pos(2); transit_z_high];
        segment = apf_segment(current, waypoint, obstacles, params, drone_radius, hangar_length, hangar_width, hangar_height);
        segment = simplify_path(segment, min_waypoint_spacing); 
        full_path = [full_path; segment]; 
        current = full_path(end, :)';
        
        % Navigate to each FOI in the optimized TSP sequence
        for step = 1:num_assigned
            target_foi = ordered_points(step, :);
            hover_z = target_foi(3) + hover_offset; 
            waypoint = [target_foi(1); target_foi(2); hover_z];
            
            segment = apf_segment(current, waypoint, obstacles, params, drone_radius, hangar_length, hangar_width, hangar_height);
            segment = simplify_path(segment, min_waypoint_spacing); 
            
            segment = [segment; waypoint'];
            full_path = [full_path; segment]; 
            current = waypoint;
            
            % Simulate inspection dwell time (hovering in place for 150 frames)
            dwell_frames = repmat(current', 150, 1);
            full_path = [full_path; dwell_frames];
        end
        
        % Return to home logic: Ascend to safe altitude, then descend
        waypoint = [home_pos(1); home_pos(2); transit_z_high];
        segment = apf_segment(current, waypoint, obstacles, params, drone_radius, hangar_length, hangar_width, hangar_height);
        segment = simplify_path(segment, min_waypoint_spacing);
        full_path = [full_path; segment]; 
        current = full_path(end, :)';
        
        waypoint = [home_pos(1); home_pos(2); home_pos(3)];
        segment = apf_segment(current, waypoint, obstacles, params, drone_radius, hangar_length, hangar_width, hangar_height);
        segment = simplify_path(segment, min_waypoint_spacing);
        full_path = [full_path; segment];
        
        all_agent_paths{d_id} = full_path;
    else
        all_agent_paths{d_id} = []; % Handle edge case where a drone is assigned 0 FOIs
    end
end

%% 6. 3D VISUALIZATION & KINEMATIC SIMULATION
figure('Name', 'Multi-Agent Hangar Inspection', 'Color', 'w', 'Position', [100, 100, 1000, 800]);
hold on;

% Render the aircraft CAD
patch('Faces', model.ConnectivityList, 'Vertices', shifted_points, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
camlight('headlight'); material('shiny'); lighting('gouraud');
xlim([0, hangar_length]); ylim([0, hangar_width]); zlim([0, hangar_height]);
daspect([1 1 1]); view(45, 30); grid on; rotate3d on;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (Altitude)');

% Render the static hangar constraints
plot_hangar_environment();

% Plot initial drone locations
for d = 1:num_drones
    scatter3(drones_pos(d,1), drones_pos(d,2), drones_pos(d,3), 150, colors(d,:), 's', 'filled');
end

% Plot mapped Features of Interest
scatter3(final_fois(:,1), final_fois(:,2), final_fois(:,3), 40, 'k', 'filled', '^');

% Initialize transformation objects for animated drone movement
drone_transforms = gobjects(num_drones, 1);
for d = 1:num_drones
    if ~isempty(all_agent_paths{d})
        path = all_agent_paths{d};
        % Plot the planned APF trajectory
        plot3(path(:,1), path(:,2), path(:,3), '-', 'Color', [colors(d,:) 0.5], 'LineWidth', 1);
        
        drone_transforms(d) = hgtransform('Parent', gca);
        if ~isempty(drone_mesh)
            patch('Faces', drone_mesh.ConnectivityList, 'Vertices', scaled_drone_pts, ...
                  'FaceColor', colors(d,:), 'EdgeColor', 'none', 'Parent', drone_transforms(d));
        else
            scatter3(0, 0, 0, 120, colors(d,:), 'o', 'filled', 'MarkerEdgeColor', 'k', 'Parent', drone_transforms(d));
        end
        set(drone_transforms(d), 'Matrix', makehgtform('translate', path(1,:)));
    end
end

title('Multi-Agent Hangar Inspection: APF Direct Hopping & Dynamic Kinematics');
drawnow; pause(2);

% --- SIMULATION LOOP: DENSE-PATH KINEMATICS ---
% Simulates pseudo-realistic UAV flight dynamics by coupling velocity 
% vectors to yaw, pitch, and roll attitudes.
max_steps = max(cellfun(@(x) size(x, 1), all_agent_paths));
drone_yaw = zeros(num_drones, 1);   
drone_pitch = zeros(num_drones, 1); 
drone_roll = zeros(num_drones, 1);  

for step = 2:max_steps
    for d = 1:num_drones
        path = all_agent_paths{d};
        if ~isempty(path) && step <= size(path, 1)
            curr_pos = path(step, :);
            prev_pos = path(step-1, :);
            vel = curr_pos - prev_pos; 
            
            % Kinematic response logic
            if norm(vel) < 1e-3
                target_pitch = 0; target_roll = 0; % Level out when hovering
            else
                if norm(vel(1:2)) > 1e-3
                    target_yaw = atan2(vel(2), vel(1)); % Point nose in direction of travel
                    target_pitch = 0.15;                % Pitch forward to simulate thrust
                else
                    target_yaw = drone_yaw(d); 
                    target_pitch = 0; 
                end
            end
            
            % Smooth attitude transitions using proportional error correction
            yaw_error = atan2(sin(target_yaw - drone_yaw(d)), cos(target_yaw - drone_yaw(d)));
            drone_yaw(d) = drone_yaw(d) + 0.20 * yaw_error;
            target_roll = -0.6 * yaw_error; % Induce roll during turning maneuvers
            
            drone_pitch(d) = drone_pitch(d) + 0.1 * (target_pitch - drone_pitch(d));
            drone_roll(d)  = drone_roll(d)  + 0.1 * (target_roll - drone_roll(d));
            
            % Apply translation and rotation matrices
            T = makehgtform('translate', curr_pos) * ...
                makehgtform('zrotate', drone_yaw(d)) * ...
                makehgtform('yrotate', drone_pitch(d)) * ...
                makehgtform('xrotate', drone_roll(d));
            set(drone_transforms(d), 'Matrix', T);
        end
    end
    drawnow; 
    pause(0.03); 
end
hold off;
end

%% ================= APF HELPER FUNCTIONS =================

function hangar_obs = generate_hangar_boxes()
    % Models static hangar environment features (pillars, scaffolding)
    % as bounding boxes for the APF repulsion logic.
    
    % Cylinders [x, y, radius, height]
    cyls = [
        6, 36, 1.8, 10;
        40, 45, 1.8, 20
    ];
    
    % Squares [x1, y1, x2, y2]
    sqs = [
        8, 10, 19, 15;
        41, 10, 46, 15
    ];

    hangar_obs = zeros(size(cyls,1) + size(sqs,1), 6);
    
    % Convert Cylinders to rectangular bounding boxes
    for i = 1:size(cyls, 1)
        cx = cyls(i, 1); cy = cyls(i, 2);
        r = cyls(i, 3);  h = cyls(i, 4);
        cz = h / 2;
        % Add slight padding (+0.2) to the radius as a safety buffer
        hangar_obs(i, :) = [cx, cy, cz, 2*r + 0.2, 2*r + 0.2, h];
    end
    
    % Convert Squares to 3D bounding boxes
    offset = size(cyls, 1);
    sq_h = 8;
    for i = 1:size(sqs, 1)
        x1 = sqs(i, 1); y1 = sqs(i, 2);
        x2 = sqs(i, 3); y2 = sqs(i, 4);
        
        w = (x2 - x1) + 0.2;
        d = (y2 - y1) + 0.2;
        cx = x1 + (x2 - x1)/2;
        cy = y1 + (y2 - y1)/2;
        cz = sq_h / 2;
        
        hangar_obs(offset + i, :) = [cx, cy, cz, w, d, sq_h];
    end
end

function plot_hangar_environment()
    % Renders the static environmental obstacles defined in generate_hangar_boxes
    cyls = [6, 36, 1.8, 10; 40, 45, 2.0, 14];
    sqs = [8, 10, 19, 15; 41, 10, 46, 15];
    
    for i = 1:size(cyls, 1)
        cx = cyls(i, 1); cy = cyls(i, 2);
        r  = cyls(i, 3); h  = cyls(i, 4);
        [Xc, Yc, Zc] = cylinder(r, 20);
        surf(Xc + cx, Yc + cy, Zc * h, 'FaceColor', [0.5 0.5 0.5], ...
            'EdgeColor', 'none', 'FaceAlpha', 0.7, 'HandleVisibility', 'off');
    end

    for i = 1:size(sqs, 1)
        x1 = sqs(i,1); y1 = sqs(i,2);
        x2 = sqs(i,3); y2 = sqs(i,4);
        h  = 8;
        vertices = [x1, y1, 0; x2, y1, 0; x2, y2, 0; x1, y2, 0;
                    x1, y1, h; x2, y1, h; x2, y2, h; x1, y2, h];
        faces = [1 2 3 4; 2 6 7 3; 6 5 8 7; 5 1 4 8; 1 5 6 2; 4 3 7 8];
        patch('Vertices', vertices, 'Faces', faces, ...
            'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'k', ...
            'FaceAlpha', 0.7, 'HandleVisibility', 'off');
    end
end

function obstacles = generate_obstacle_field(vertices, faces)
    % Uses Monte Carlo sampling to distribute point-cloud clusters over the 
    % aircraft CAD surface, converting them into repulsive bounding boxes.
    num_obstacles = 50; 
    sample_points = zeros(1500, 3); 
    
    for i = 1:1500
        face_idx = randi(size(faces, 1));
        face_verts = vertices(faces(face_idx, :), :);
        % Barycentric coordinates to sample uniformly within a triangular face
        a = rand; b = rand;
        if a + b > 1, a = 1 - a; b = 1 - b; end
        c = 1 - a - b;
        sample_points(i, :) = a * face_verts(1, :) + b * face_verts(2, :) + c * face_verts(3, :);
    end
    
    [cluster_idx, centers] = kmeans(sample_points, num_obstacles, 'MaxIter', 200);
    obstacles = zeros(num_obstacles, 6);
    for i = 1:num_obstacles
        cluster_pts = sample_points(cluster_idx == i, :);
        if size(cluster_pts, 1) > 1
            obstacles(i, 1:3) = centers(i, :);
            obstacles(i, 4) = max(cluster_pts(:,1)) - min(cluster_pts(:,1)) + 0.5;
            obstacles(i, 5) = max(cluster_pts(:,2)) - min(cluster_pts(:,2)) + 0.5;
            obstacles(i, 6) = max(cluster_pts(:,3)) - min(cluster_pts(:,3)) + 0.5;
        else
            obstacles(i, 1:3) = centers(i, :);
            obstacles(i, 4:6) = [1.0, 1.0, 1.0]; 
        end
    end
end

function [dist_box, nearest_pt, dir_vec] = point_to_box_distance(p, obs)
    % Calculates the Euclidean distance and directional gradient from a point
    % to the nearest edge of a 3D bounding box for APF force generation.
    cx = obs(1); cy = obs(2); cz = obs(3);
    w = obs(4); d = obs(5); h = obs(6);
    xmin = cx - w/2; xmax = cx + w/2; ymin = cy - d/2; ymax = cy + d/2; zmin = cz - h/2; zmax = cz + h/2;
    nearest_pt = [min(max(p(1), xmin), xmax), min(max(p(2), ymin), ymax), min(max(p(3), zmin), zmax)];
    diff = p - nearest_pt;
    dist_box = norm(diff);
    if dist_box < 1e-8
        % Point is inside or exactly on the box
        diff = p - [cx cy cz];
        if norm(diff) < 1e-8, diff = [1 0 0]; end
        dir_vec = diff / norm(diff);
        dist_box = 0;
    else
        dir_vec = diff / dist_box;
    end
end

function c = min_box_clearance(p, obstacles)
    % Returns the shortest distance to any obstacle in the environment.
    c = inf;
    for ii = 1:size(obstacles,1)
        [dist_box,~,~] = point_to_box_distance(p, obstacles(ii,:));
        c = min(c, dist_box);
    end
end

function p = keep_in_bounds(p, xmax, ymax, zmax)
    % Clamps positional coordinates to ensure drones do not breach hangar walls.
    p(1) = max(0, min(xmax, p(1))); 
    p(2) = max(0, min(ymax, p(2))); 
    p(3) = max(0, min(zmax, p(3)));
end

function segment = apf_segment(start, goal, obstacles, params, drone_radius, hangar_len, hangar_wid, hangar_ht)
    % Solves for a collision-free path segment using Artificial Potential Fields.
    % Forces act on the drone: Attractive to the goal, Repulsive from obstacles.
    segment = []; 
    current = start(:); 
    goal = goal(:);
    max_force = 3.5; % Saturation limit to prevent massive trajectory spikes
    
    for iter = 1:params.max_iterations
        % 1. Calculate Attractive Force (Linear mapping to distance)
        F_att = params.attraction_gain * (goal - current);
        
        % 2. Calculate Repulsive Force from all obstacles within influence radius
        F_rep = [0; 0; 0];
        min_dist = inf;
        dist_to_goal = norm(current - goal); 
        
        for i = 1:size(obstacles, 1)
            [dist_box, ~, dir_vec] = point_to_box_distance(current', obstacles(i,:));
            min_dist = min(min_dist, dist_box);
            clearance = dist_box - drone_radius;
            
            if clearance < params.repulsion_distance
                clearance = max(clearance, 0.05); % Prevent division by zero
                % Non-linear repulsive function based on proximity
                mag = params.repulsion_gain * (1/clearance - 1/params.repulsion_distance)^2;
                % Attenuate repulsion near the goal to allow approach
                mag = mag * min(1.0, (dist_to_goal / 2.0)^2); 
                F_rep = F_rep + mag * dir_vec(:);
            end
        end
        
        % 3. Calculate Total Force and update position
        F_total = F_att + F_rep;
        if norm(F_total) > max_force
            F_total = F_total / norm(F_total) * max_force; 
        end
        
        candidate = current + F_total * params.dt;
        candidate = keep_in_bounds(candidate, hangar_len, hangar_wid, hangar_ht);
        cand_clearance = min_box_clearance(candidate', obstacles);
        
        % Emergency collision handling: Override if pushing into a boundary
        if cand_clearance < drone_radius
            if norm(F_rep) > 1e-6
                emergency_dir = F_rep / norm(F_rep);
            else
                emergency_dir = randn(3,1); 
                emergency_dir = emergency_dir / norm(emergency_dir);
            end
            candidate = current + 0.5 * emergency_dir;
            candidate = keep_in_bounds(candidate, hangar_len, hangar_wid, hangar_ht);
        end
        
        current = candidate;
        segment = [segment; current'];
        
        % Check for convergence
        if norm(current - goal) < params.goal_threshold
            segment = [segment; goal'];
            break;
        end
        
        % Perturbation to escape local minima (e.g., getting stuck in a corner)
        if min_dist < drone_radius * 1.3 && params.escape_force > 0
            escape_dir = randn(3,1); 
            escape_dir = escape_dir / norm(escape_dir);
            current = current + escape_dir * params.escape_force * params.dt;
            current = keep_in_bounds(current, hangar_len, hangar_wid, hangar_ht);
        end
    end
end

function path_out = simplify_path(path_in, min_spacing)
    % Applies a Gaussian smoothing filter to the raw APF trajectory
    % and decimates waypoints to achieve a more uniform flight profile.
    if size(path_in,1) <= 2
        path_out = path_in; 
        return; 
    end
    
    window = min(15, size(path_in,1)); 
    smoothed = path_in;
    smoothed(:,1) = smoothdata(path_in(:,1), 'gaussian', window);
    smoothed(:,2) = smoothdata(path_in(:,2), 'gaussian', window);
    smoothed(:,3) = smoothdata(path_in(:,3), 'gaussian', window);
    
    % Enforce exact start and end points
    smoothed(1,:) = path_in(1,:);
    smoothed(end,:) = path_in(end,:);
    
    % Decimate based on spatial distance resolution
    keep_idx = 1; 
    last_kept = smoothed(1,:);
    for i = 2:size(smoothed,1)-1
        if norm(smoothed(i,:) - last_kept) >= min_spacing
            keep_idx(end+1) = i; 
            last_kept = smoothed(i,:);
        end
    end
    keep_idx(end+1) = size(smoothed,1);
    path_out = smoothed(keep_idx,:);
end