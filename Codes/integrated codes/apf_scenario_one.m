function hangar_inspection_apf_cad_with_pillars()
% AIRCRAFT HANGAR INSPECTION: DIRECT HOPPING APF WITH CAD KINEMATICS
% AND DIAGRAM-DEFINED PILLAR OBSTACLES (WITH TELEMETRY & PHYSICS PROOF)

clc; clear; close all;

%% 1. CONFIGURABLE PARAMETERS
hangar_length = 50; 
hangar_width = 50;  
hangar_height = 20;  

% Task Allocation Weights
w_dist = 1.0; 
w_battery = 1.0;

min_waypoint_spacing = 0.15;       

% APF Collision Avoidance Parameters
params = struct();
params.attraction_gain = 1.2;      
params.repulsion_gain = 75.0;      
params.repulsion_distance = 2.0;   
params.dt = 0.05;                  
params.max_iterations = 3000;      
params.goal_threshold = 0.2;       
params.escape_force = 1.0;         

drone_radius = 0.85;               
%% 2. LOAD AIRCRAFT, GENERATE OBSTACLES, & LOAD CAD DRONE
fprintf('Loading aircraft and generating APF obstacle field...\n');
model = stlread('airplane.STL');
original_points = model.Points;
min_coords = min(original_points);
shifted_points = original_points - min_coords + [10, 10, 0];
aircraft_faces = model.ConnectivityList;

aircraft_obstacles = generate_obstacle_field(shifted_points, aircraft_faces);

% fprintf('Loading stairs obstacle...\n');
% try
%     stairs_model = stlread('Stairs 1680mm FULL Assembly.stl');
%     stairs_pts = stairs_model.Points * 0.0017;
%     stairs_center = mean(stairs_pts);
%     stairs_pts = stairs_pts - stairs_center;
% 
%     theta_x = pi/2; Rx = [1 0 0; 0 cos(theta_x) -sin(theta_x); 0 sin(theta_x)  cos(theta_x)];
%     theta_z = -pi/2; Rz = [cos(theta_z) -sin(theta_z) 0; sin(theta_z)  cos(theta_z) 0; 0 0 1];
%     stairs_pts = ((Rz * Rx) * stairs_pts')';
%     stairs_offset = [42.5, 17.5, 0]; 
%     stairs_pts = stairs_pts - min(stairs_pts) + stairs_offset;
%     stairs_obstacles = generate_obstacle_field(stairs_pts, stairs_model.ConnectivityList);
% catch
%     warning('Stairs STL not found. Skipping stairs obstacle.');
%     stairs_model = []; stairs_obstacles = []; stairs_pts = [];
% end
% 
% pillar_positions = [5, 10, 5; 47.5, 10, 3];
% pillar_dims = [1.5, 1, 20; 1.5, 1, 20];
% num_pillars = size(pillar_positions, 1);
% new_pillar_obstacles = zeros(num_pillars, 6);
% for i = 1:num_pillars
%     new_pillar_obstacles(i, :) = [pillar_positions(i, :), pillar_dims(i, :)];
% end

obstacles = [aircraft_obstacles];   % Add stairs and pillar obstacle here, to this matrix if needed

try
    drone_mesh = stlread('drone.stl');
    scaled_drone_pts = drone_mesh.Points * 0.075;
    scaled_drone_pts = scaled_drone_pts - mean(scaled_drone_pts);
catch
    drone_mesh = [];
end
%% 3. SURFACE FILTERING & MANUAL FOI DEFINITION
TR = triangulation(model.ConnectivityList, shifted_points);
normals = faceNormal(TR); incenters = incenter(TR);
z_threshold = 0.25 * max(shifted_points(:,3)); 
valid_mask = (normals(:,3) > 0.2) & (incenters(:,3) > z_threshold);
valid_points = incenters(valid_mask, :);

user_defined_fois = [
    42.0, 25.0, 5.0;  45.0, 25.0, 4.0;  25.0, 25.0, 5.0;  
    25.0, 15.0, 2.5;  25.0, 10.0, 2.5;  27.5, 34.5, 2.5;
    27.5, 31.5, 2.5;  11.0, 28.0, 5.0;  15.0, 40.0, 11.0;
    11.0, 17.0, 5.0;  21.0, 18.0, 15.0; 13.0, 20.0, 3.0  
];

final_fois = zeros(size(user_defined_fois, 1), 3);
for i = 1:size(user_defined_fois, 1)
    [~, min_idx] = min(sum((valid_points - user_defined_fois(i,:)).^2, 2));
    final_fois(i,:) = valid_points(min_idx, :);
end
%% 4. DEFINE AGENTS & HUNGARIAN ALLOCATION
num_drones = 5;
drones_pos = [0, 0, 0; 50, 0, 0; 0, 50, 0; 50, 50, 0; 25, 0, 0];
drone_battery = [80; 85; 83; 85; 79]; 
colors = lines(num_drones);
[foi_cluster_idx, cluster_centers] = kmeans(final_fois, num_drones, 'Start', drones_pos);
CostMatrix = zeros(num_drones, num_drones);
for i = 1:num_drones
    for j = 1:num_drones 
        CostMatrix(i, j) = (w_dist * norm(drones_pos(i,:) - cluster_centers(j,:))) + (w_battery * (100 - drone_battery(i)));
    end
end
assignments = matchpairs(CostMatrix, 1000);

%% 5. TSP ROUTING WITH APF DIRECT HOPPING
fprintf('Calculating APF Collision-Free Direct Hops...\n');
all_agent_paths = cell(num_drones, 1);
transit_z_high = 6;  
hover_offset = 2.5;

% --- TELEMETRY PRE-ALLOCATION ---
assigned_fois_count = zeros(num_drones, 1);
path_distances = zeros(num_drones, 1);

for k = 1:size(assignments, 1)
    d_id = assignments(k, 1); c_id = assignments(k, 2); 
    home_pos = drones_pos(d_id, :);
    assigned_fois = final_fois(foi_cluster_idx == c_id, :);
    num_assigned = size(assigned_fois, 1);
    assigned_fois_count(d_id) = num_assigned; 
    
    if num_assigned > 0
        all_perms = perms(1:num_assigned);
        min_route_dist = inf; best_route = [];
        for p = 1:size(all_perms, 1)
            current_order = all_perms(p, :); route_dist = 0;
            current_pos = home_pos;
            for step = 1:num_assigned
                route_dist = route_dist + norm(assigned_fois(current_order(step), :) - current_pos);
                current_pos = assigned_fois(current_order(step), :);
            end
            route_dist = route_dist + norm(home_pos - current_pos);
            if route_dist < min_route_dist
                min_route_dist = route_dist; best_route = current_order;
            end
        end
        ordered_points = assigned_fois(best_route, :);
        
        full_path = home_pos;
        current = home_pos';
        
        waypoint = [home_pos(1); home_pos(2); transit_z_high];
        segment = apf_segment(current, waypoint, obstacles, params, drone_radius, hangar_length, hangar_width, hangar_height);
        segment = simplify_path(segment, min_waypoint_spacing); 
        full_path = [full_path; segment]; current = full_path(end, :)';
        
        for step = 1:num_assigned
            target_foi = ordered_points(step, :);
            hover_z = target_foi(3) + hover_offset; 
            waypoint = [target_foi(1); target_foi(2); hover_z];
            
            segment = apf_segment(current, waypoint, obstacles, params, drone_radius, hangar_length, hangar_width, hangar_height);
            segment = simplify_path(segment, min_waypoint_spacing); 
            
            segment = [segment; waypoint'];
            full_path = [full_path; segment]; 
            current = waypoint;
            
            dwell_frames = repmat(current', 150, 1);
            full_path = [full_path; dwell_frames];
        end
        
        waypoint = [home_pos(1); home_pos(2); transit_z_high];
        segment = apf_segment(current, waypoint, obstacles, params, drone_radius, hangar_length, hangar_width, hangar_height);
        segment = simplify_path(segment, min_waypoint_spacing);
        full_path = [full_path; segment]; current = full_path(end, :)';
        
        waypoint = [home_pos(1); home_pos(2); home_pos(3)];
        segment = apf_segment(current, waypoint, obstacles, params, drone_radius, hangar_length, hangar_width, hangar_height);
        segment = simplify_path(segment, min_waypoint_spacing);
        full_path = [full_path; segment];
        
        all_agent_paths{d_id} = full_path;
        
        % Log Distance
        dist = 0;
        for i = 2:size(full_path,1)
            dist = dist + norm(full_path(i,:) - full_path(i-1,:));
        end
        path_distances(d_id) = dist;
    else
        all_agent_paths{d_id} = [];
    end
end

%% 6. 3D CAD VISUALIZATION & DYNAMIC KINEMATICS
figure('Name', 'APF Direct Hopping Simulation with Pillars', 'Color', 'w', 'Position', [100, 100, 1000, 800]);
hold on;
patch('Faces', model.ConnectivityList, 'Vertices', shifted_points, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none', 'FaceAlpha', 1.0);
% if ~isempty(stairs_model)
%     patch('Faces', stairs_model.ConnectivityList, 'Vertices', stairs_pts, 'FaceColor', [0.4 0.5 0.6], 'EdgeColor', 'none', 'FaceAlpha', 0.9);
% end
% 
% plot_pillar_radii = pillar_dims(:,1)/2; 
% plot_pillar_heights = pillar_dims(:,3);
% plot_pillar_centers = pillar_positions;
% 
% for i = 1:num_pillars
%     [cX, cY, cZ] = cylinder(plot_pillar_radii(i), 20); 
%     surf(plot_pillar_centers(i, 1) + cX, plot_pillar_centers(i, 2) + cY, cZ * plot_pillar_heights(i), ...
%         'FaceColor', [0.8 0.1 0.1], 'EdgeColor', 'none', 'FaceAlpha', 0.8); 
% end

camlight('headlight'); material('shiny'); lighting('gouraud');
xlim([0, hangar_length]); ylim([0, hangar_width]); zlim([0, hangar_height]);
daspect([1 1 1]); view(45, 30); grid on; rotate3d on;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (Altitude)');

for d = 1:num_drones
    scatter3(drones_pos(d,1), drones_pos(d,2), drones_pos(d,3), 150, colors(d,:), 's', 'filled');
end
scatter3(final_fois(:,1), final_fois(:,2), final_fois(:,3), 40, 'k', 'filled', '^');

drone_transforms = gobjects(num_drones, 1);
for d = 1:num_drones
    if ~isempty(all_agent_paths{d})
        path = all_agent_paths{d};
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

drawnow; pause(2);

% --- INITIALIZE NEW PHYSICS TELEMETRY DATA STRUCTURE ---
telemetry = repmat(struct('clearance', [], 'U_rep', [], 'F_rep', []), num_drones, 1);

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
            
            if norm(vel) < 1e-3
                target_pitch = 0; target_roll = 0;
            else
                if norm(vel(1:2)) > 1e-3
                    target_yaw = atan2(vel(2), vel(1));
                    target_pitch = 0.15;
                else
                    target_yaw = drone_yaw(d); 
                    target_pitch = 0; 
                end
            end
            
            yaw_error = atan2(sin(target_yaw - drone_yaw(d)), cos(target_yaw - drone_yaw(d)));
            drone_yaw(d) = drone_yaw(d) + 0.20 * yaw_error;
            target_roll = -0.6 * yaw_error; 
            drone_pitch(d) = drone_pitch(d) + 0.1 * (target_pitch - drone_pitch(d));
            drone_roll(d)  = drone_roll(d)  + 0.1 * (target_roll - drone_roll(d));
            
            T = makehgtform('translate', curr_pos) * ...
                makehgtform('zrotate', drone_yaw(d)) * ...
                makehgtform('yrotate', drone_pitch(d)) * ...
                makehgtform('xrotate', drone_roll(d));
            set(drone_transforms(d), 'Matrix', T);
            
            % --- RECORD TELEMETRY (Safety & Force Physics) ---
            telemetry(d).clearance(end+1) = min_box_clearance(curr_pos, obstacles);
            
            % Calculate Theoretical Repulsive Potential & Force Magnitude at this step
            U_rep = 0;
            F_rep_vec = [0 0 0];
            for i = 1:size(obstacles, 1)
                [dist_box, ~, dir_vec] = point_to_box_distance(curr_pos', obstacles(i,:));
                c = dist_box - drone_radius;
                if c < params.repulsion_distance
                    c = max(c, 0.05);
                    % Potential Energy Equation
                    U_rep = U_rep + 0.5 * params.repulsion_gain * (1/c - 1/params.repulsion_distance)^2;
                    % Force Vector Equation
                    mag = params.repulsion_gain * (1/c - 1/params.repulsion_distance) * (1/c^2);
                    F_rep_vec = F_rep_vec + mag * dir_vec(:)';
                end
            end
            telemetry(d).U_rep(end+1) = U_rep;
            telemetry(d).F_rep(end+1) = norm(F_rep_vec);
        end
    end
    drawnow; 
    pause(0.08); 
end
% =========================================================================
% --- FINAL MISSION ANALYTICS (Cost, Time, Payoff) ---
% =========================================================================
fprintf('\n================ MISSION EVALUATION REPORT ================\n');

% 1. MISSION TIME: (Steps * dt) + (Dwells at FOIs)
% Your script uses 150 frames for each dwell. At 0.05s dt, that is 7.5s per FOI.
dwell_time_per_foi = 150 * params.dt; 

total_fleet_distance = sum(path_distances);
total_fleet_cost = sum(CostMatrix(sub2ind(size(CostMatrix), assignments(:,1), assignments(:,2))));

for k = 1:size(assignments, 1)
    d_id = assignments(k, 1);
    
    % Actual flight time from the recorded path
    flight_time = size(all_agent_paths{d_id}, 1) * params.dt;
    
    % Total time including FOI inspection stops
    total_time = flight_time + (assigned_fois_count(d_id) * dwell_time_per_foi);
    
    % Payoff Calculation (Efficiency Score)
    % Higher is better: reward FOI completion, penalize distance and time
    payoff = (assigned_fois_count(d_id) * 100) / (path_distances(d_id) + total_time);

    fprintf('Drone %d Performance:\n', d_id);
    fprintf('  - Total Mission Time:    %.2f seconds\n', total_time);
    fprintf('  - Actual Path Cost:      %.2f meters\n', path_distances(d_id));
    fprintf('  - Calculated Payoff:     %.2f\n', payoff);
end

fprintf('TOTAL FLEET PAYOFF:        %.2f\n', sum(assigned_fois_count)*10 / total_fleet_distance);
fprintf('TOTAL ASSIGNMENT COST:     %.2f\n', total_fleet_cost);
hold off;

% --- GENERATE ENGINEERING DASHBOARD ---
generate_telemetry_dashboard(telemetry, num_drones, drone_radius, assigned_fois_count, path_distances, colors);

fprintf('\n================ MISSION SUMMARY ================\n');
total_cost = 0;
for k = 1:size(assignments, 1)
    d_id = assignments(k, 1);
    c_id = assignments(k, 2);
    cost = CostMatrix(d_id, c_id);
    total_cost = total_cost + cost;
    
    % Time estimation
    avg_speed = 0.8;       % Safe indoor proximity flight speed (m/s)
    dwell_time = 25.0;     % 25s for stabilization, autofocus, and image capture
    accel_penalty = 1.5;   % Adds 1.5s per waypoint to account for braking/turning

num_waypoints = size(all_agent_paths{d_id}, 1);

time_taken = (path_distances(d_id) / avg_speed) + ...
             (assigned_fois_count(d_id) * dwell_time) + ...
             (num_waypoints * accel_penalty);
    
    fprintf('Drone %d:\n', d_id);
    fprintf('  - FOIs Assigned:     %d\n', assigned_fois_count(d_id));
    fprintf('  - Distance Traveled: %.2f m\n', path_distances(d_id));
    fprintf('  - Estimated Time:    %.1f seconds\n', time_taken);
    fprintf('  - Assignment Cost:   %.2f\n', cost);
end
fprintf('TOTAL MISSION COST:    %.2f\n', total_cost);
end

%% 
%% APF HELPER FUNCTIONS
%% 
function obstacles = generate_obstacle_field(vertices, faces)
    num_obstacles = 50;
    sample_points = zeros(1500, 3);
    for i = 1:1500
        face_idx = randi(size(faces, 1));
        face_verts = vertices(faces(face_idx, :), :);
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
    cx = obs(1); cy = obs(2); cz = obs(3);
    w = obs(4); d = obs(5); h = obs(6);
    xmin = cx - w/2; xmax = cx + w/2; ymin = cy - d/2; ymax = cy + d/2; zmin = cz - h/2; zmax = cz + h/2;
    nearest_pt = [min(max(p(1), xmin), xmax), min(max(p(2), ymin), ymax), min(max(p(3), zmin), zmax)];
    diff = p - nearest_pt;
    dist_box = norm(diff);
    if dist_box < 1e-8
        diff = p - [cx cy cz];
        if norm(diff) < 1e-8, diff = [1 0 0]; end
        dir_vec = diff / norm(diff);
        dist_box = 0;
    else
        dir_vec = diff / dist_box;
    end
end
function c = min_box_clearance(p, obstacles)
    c = inf;
    for ii = 1:size(obstacles,1)
        [dist_box,~,~] = point_to_box_distance(p, obstacles(ii,:));
        c = min(c, dist_box);
    end
end
function p = keep_in_bounds(p, xmax, ymax, zmax)
    p(1) = max(0, min(xmax, p(1))); p(2) = max(0, min(ymax, p(2))); p(3) = max(0, min(zmax, p(3)));
end
function segment = apf_segment(start, goal, obstacles, params, drone_radius, hangar_len, hangar_wid, hangar_ht)
    segment = []; current = start(:); goal = goal(:);
    max_force = 25.5;
    for iter = 1:params.max_iterations
        F_att = params.attraction_gain * (goal - current);
        F_rep = [0; 0; 0];
        min_dist = inf;
        dist_to_goal = norm(current - goal); 
        
        for i = 1:size(obstacles, 1)
            [dist_box, ~, dir_vec] = point_to_box_distance(current', obstacles(i,:));
            min_dist = min(min_dist, dist_box);
            clearance = dist_box - drone_radius;
            if clearance < params.repulsion_distance
                clearance = max(clearance, 0.05);
                mag = params.repulsion_gain * (1/clearance - 1/params.repulsion_distance)^2;
                mag = mag * min(1.0, (dist_to_goal / 2.0)^2); 
                F_rep = F_rep + mag * dir_vec(:);
            end
        end
        
        F_total = F_att + F_rep;
        if norm(F_total) > max_force, F_total = F_total / norm(F_total) * max_force; end
        
        candidate = current + F_total * params.dt;
        candidate = keep_in_bounds(candidate, hangar_len, hangar_wid, hangar_ht);
        cand_clearance = min_box_clearance(candidate', obstacles);
        
        if cand_clearance < drone_radius
            if norm(F_rep) > 1e-6
                emergency_dir = F_rep / norm(F_rep);
            else
                emergency_dir = randn(3,1); emergency_dir = emergency_dir / norm(emergency_dir);
            end
            candidate = current + 0.5 * emergency_dir;
            candidate = keep_in_bounds(candidate, hangar_len, hangar_wid, hangar_ht);
        end
        current = candidate;
        segment = [segment; current'];
        
        if norm(current - goal) < params.goal_threshold
            segment = [segment; goal'];
            break;
        end
        
        if min_dist < drone_radius * 1.3 && params.escape_force > 0
            escape_dir = randn(3,1); escape_dir = escape_dir / norm(escape_dir);
            current = current + escape_dir * params.escape_force * params.dt;
            current = keep_in_bounds(current, hangar_len, hangar_wid, hangar_ht);
        end
    end
end
function path_out = simplify_path(path_in, min_spacing)
    if size(path_in,1) <= 2, path_out = path_in; return; end
    window = min(15, size(path_in,1)); 
    smoothed = path_in;
    smoothed(:,1) = smoothdata(path_in(:,1), 'gaussian', window);
    smoothed(:,2) = smoothdata(path_in(:,2), 'gaussian', window);
    smoothed(:,3) = smoothdata(path_in(:,3), 'gaussian', window);
    smoothed(1,:) = path_in(1,:); smoothed(end,:) = path_in(end,:);
    
    keep_idx = 1; last_kept = smoothed(1,:);
    for i = 2:size(smoothed,1)-1
        if norm(smoothed(i,:) - last_kept) >= min_spacing
            keep_idx(end+1) = i; last_kept = smoothed(i,:);
        end
    end
    keep_idx(end+1) = size(smoothed,1);
    path_out = smoothed(keep_idx,:);
end

% --- TELEMETRY DASHBOARD FUNCTION ---
function generate_telemetry_dashboard(telemetry, num_drones, drone_radius, foi_counts, distances, colors)
    figure('Name', 'Engineering Telemetry Dashboard', 'Color', 'w', 'Position', [150, 150, 1200, 800]);
    
    % 1. Safety Clearance Plot
    figure(2)
    hold on; grid on;
    title('Safety Validation: Minimum Clearance');
    xlabel('Simulation Step'); ylabel('Distance to Obstacle (m)');
    for d = 1:num_drones
        if ~isempty(telemetry(d).clearance)
            plot(telemetry(d).clearance, 'Color', colors(d,:), 'LineWidth', 1.5, 'DisplayName', sprintf('Drone %d', d));
        end
    end
    yline(drone_radius, 'r--', 'Collision Threshold', 'LineWidth', 2, 'LabelHorizontalAlignment', 'left');
    legend('Location', 'best');
    
    % % 2. Forces and Potentials Plot (Replacing Kinematics)
    % figure(2)
    % yyaxis left; hold on; grid on;
    % plot(telemetry(1).F_rep, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Repulsive Force (F_{rep})');
    % ylabel('Force Magnitude');
    % 
    % yyaxis right;
    % plot(telemetry(1).U_rep, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Repulsive Potential (U_{rep})');
    % ylabel('Potential Energy');
    % 
    % title('Obstacle Field Physics (Drone 1)');
    % xlabel('Simulation Step');
    % legend('Location', 'northwest');
    
    % 3. Fleet Workload Distribution (Sorted)
    figure(3)
    subplot(2, 1, 1);
    [sorted_fois, sort_idx1] = sort(foi_counts, 'descend');
    b = bar(1:num_drones, sorted_fois);
    b.FaceColor = 'flat';
    for k = 1:num_drones
        b.CData(k,:) = colors(sort_idx1(k),:);
    end
    title('Workload Balancing: Assigned FOIs');
    xlabel('Drone ID'); ylabel('Number of FOIs');
    set(gca, 'XTick', 1:num_drones, 'XTickLabel', arrayfun(@(x) sprintf('Drone %d', x), sort_idx1, 'UniformOutput', false));
    grid on;
    
    % 4. Flight Efficiency (Sorted)
    subplot(2, 1, 2);
    [sorted_dists, sort_idx2] = sort(distances, 'ascend');
    b2 = bar(1:num_drones, sorted_dists);
    b2.FaceColor = 'flat';
    for k = 1:num_drones
        b2.CData(k,:) = colors(sort_idx2(k),:);
    end
    title('Flight Efficiency: Distance Traveled');
    xlabel('Drone ID'); ylabel('Total Distance (m)');
    set(gca, 'XTick', 1:num_drones, 'XTickLabel', arrayfun(@(x) sprintf('Drone %d', x), sort_idx2, 'UniformOutput', false));
    grid on;
end
