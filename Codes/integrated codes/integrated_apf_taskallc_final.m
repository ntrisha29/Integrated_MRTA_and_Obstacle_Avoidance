function hangar_inspection_apf_cad_with_pillars()
% AIRCRAFT HANGAR INSPECTION: DIRECT HOPPING APF WITH CAD KINEMATICS
% AND DIAGRAM-DEFINED PILLAR OBSTACLES (SLOW FLIGHT) + TELEMETRY PLOTS
clc; clear; close all;

%% 1. CONFIGURABLE PARAMETERS
hangar_length = 50; 
hangar_width = 50;  
hangar_height = 20;  

% Task Allocation Weights
w_dist = 1.0; 
w_battery = 1.0;

% --- PRESERVED FROM PRIOR TURN: SLOW FLIGHT PARAMETERS ---
min_waypoint_spacing = 0.15;       

% APF Collision Avoidance Parameters
params = struct();
params.attraction_gain = 6.2;      
params.repulsion_gain = 70.0;
params.repulsion_distance = 2.5;
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

fprintf('Loading stairs obstacle...\n');
try
    stairs_model = stlread('Stairs 1680mm FULL Assembly.stl');
    stairs_pts = stairs_model.Points * 0.0017; 
    stairs_center = mean(stairs_pts);
    stairs_pts = stairs_pts - stairs_center;
    
    theta_x = pi/2;
    Rx = [1 0 0; 0 cos(theta_x) -sin(theta_x); 0 sin(theta_x)  cos(theta_x)];
    theta_z = -pi/2;
    Rz = [cos(theta_z) -sin(theta_z) 0; sin(theta_z)  cos(theta_z) 0; 0 0 1];
    stairs_pts = ((Rz * Rx) * stairs_pts')';
    stairs_offset = [42.5, 17.5, 0]; 
    stairs_pts = stairs_pts - min(stairs_pts) + stairs_offset;
    stairs_obstacles = generate_obstacle_field(stairs_pts, stairs_model.ConnectivityList);
catch
    stairs_model = []; stairs_obstacles = []; stairs_pts = [];
end

pillar_positions = [5, 10, 5; 47.5, 10, 3];
pillar_dims = [1.5, 1, 20; 1.5, 1, 20]; 
num_pillars = size(pillar_positions, 1);
new_pillar_obstacles = zeros(num_pillars, 6);
for i = 1:num_pillars
    new_pillar_obstacles(i, :) = [pillar_positions(i, :), pillar_dims(i, :)];
end
obstacles = [aircraft_obstacles; stairs_obstacles; new_pillar_obstacles];

try
    drone_mesh = stlread('drone.stl');
    scaled_drone_pts = drone_mesh.Points * 0.075;
    center_offset = mean(scaled_drone_pts);
    scaled_drone_pts = scaled_drone_pts - center_offset;
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
    42.0, 25.0, 5.0;   45.0, 25.0, 4.0;  
    25.0, 10.0, 2.5;   27.5, 17.5, 2.5;
    25.0, 42.0, 2.5;   27.5, 31.5, 2.5;
    11.0, 28.0, 5.0;   15.0, 40.0, 11.0;
    11.0, 17.0, 5.0;   13.0, 20.0, 15.0; 
];
final_fois = zeros(size(user_defined_fois, 1), 3);
for i = 1:size(user_defined_fois, 1)
    [~, min_idx] = min(sum((valid_points - user_defined_fois(i,:)).^2, 2));
    final_fois(i,:) = valid_points(min_idx, :);
end

%% 4. DEFINE AGENTS & HUNGARIAN ALLOCATION
num_drones = 2;
drones_pos = [0, 0, 0; 50, 0, 0];
drone_battery = [80; 85]; 
colors = lines(num_drones);

[foi_cluster_idx, cluster_centers] = kmeans(final_fois, num_drones, 'Start', drones_pos);
CostMatrix = zeros(num_drones, num_drones);
for i = 1:num_drones
    for j = 1:num_drones 
        CostMatrix(i, j) = (w_dist * norm(drones_pos(i,:) - cluster_centers(j,:))) + (w_battery * (100 - drone_battery(i)));
    end
end
assignments = matchpairs(CostMatrix, 1000);

%% 5. TSP ROUTING WITH APF DIRECT HOPPING & EXACT DWELLS
fprintf('Calculating APF Collision-Free Direct Hops...\n');
all_agent_paths = cell(num_drones, 1);
all_agent_goals = cell(num_drones, 1); % --- NEW: Store active goals for Convergence Plot ---
transit_z_high = 6;  
hover_offset = 2.5;
d1_forces = [];
d1_comp_times = [];

for k = 1:size(assignments, 1)
    d_id = assignments(k, 1); c_id = assignments(k, 2); 
    home_pos = drones_pos(d_id, :);
    assigned_fois = final_fois(foi_cluster_idx == c_id, :);
    num_assigned = size(assigned_fois, 1);
    
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
        
        waypoint = [home_pos(1); home_pos(2); transit_z_high];
        current_goals = waypoint'; % Track initial goal
        current = home_pos';
        
        [segment, f_log, t_log] = apf_segment(current, waypoint, obstacles, params, drone_radius, hangar_length, hangar_width, hangar_height);
        if d_id == 1, d1_forces = [d1_forces; f_log]; d1_comp_times = [d1_comp_times; t_log]; end
        segment = simplify_path(segment, min_waypoint_spacing); 
        full_path = [full_path; segment]; 
        current_goals = [current_goals; repmat(waypoint', size(segment, 1), 1)];
        current = full_path(end, :)';
        
        for step = 1:num_assigned
            target_foi = ordered_points(step, :);
            hover_z = target_foi(3) + hover_offset; 
            waypoint = [target_foi(1); target_foi(2); hover_z];
            
            [segment, f_log, t_log] = apf_segment(current, waypoint, obstacles, params, drone_radius, hangar_length, hangar_width, hangar_height);
            if d_id == 1, d1_forces = [d1_forces; f_log]; d1_comp_times = [d1_comp_times; t_log]; end
            segment = simplify_path(segment, min_waypoint_spacing); 
            
            segment = [segment; waypoint'];
            full_path = [full_path; segment]; 
            current_goals = [current_goals; repmat(waypoint', size(segment, 1), 1)];
            current = waypoint;
            
            dwell_frames = repmat(current', 150, 1);
            full_path = [full_path; dwell_frames];
            current_goals = [current_goals; repmat(waypoint', size(dwell_frames, 1), 1)];
        end
        
        waypoint = [home_pos(1); home_pos(2); transit_z_high];
        [segment, f_log, t_log] = apf_segment(current, waypoint, obstacles, params, drone_radius, hangar_length, hangar_width, hangar_height);
        if d_id == 1, d1_forces = [d1_forces; f_log]; d1_comp_times = [d1_comp_times; t_log]; end
        segment = simplify_path(segment, min_waypoint_spacing);
        full_path = [full_path; segment]; 
        current_goals = [current_goals; repmat(waypoint', size(segment, 1), 1)];
        current = full_path(end, :)';
        
        waypoint = [home_pos(1); home_pos(2); home_pos(3)];
        [segment, f_log, t_log] = apf_segment(current, waypoint, obstacles, params, drone_radius, hangar_length, hangar_width, hangar_height);
        if d_id == 1, d1_forces = [d1_forces; f_log]; d1_comp_times = [d1_comp_times; t_log]; end
        segment = simplify_path(segment, min_waypoint_spacing);
        full_path = [full_path; segment];
        current_goals = [current_goals; repmat(waypoint', size(segment, 1), 1)];
        
        all_agent_paths{d_id} = full_path;
        all_agent_goals{d_id} = current_goals;
    else
        all_agent_paths{d_id} = [];
        all_agent_goals{d_id} = [];
    end
end

%% 6. 3D CAD VISUALIZATION & DYNAMIC KINEMATICS
figure('Name', 'APF Direct Hopping Simulation with Pillars', 'Color', 'w', 'Position', [100, 100, 1000, 800]);
hold on;
patch('Faces', model.ConnectivityList, 'Vertices', shifted_points, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none', 'FaceAlpha', 1.0);
if ~isempty(stairs_model)
    patch('Faces', stairs_model.ConnectivityList, 'Vertices', stairs_pts, 'FaceColor', [0.4 0.5 0.6], 'EdgeColor', 'none', 'FaceAlpha', 0.9);
end

plot_pillar_radii = pillar_dims(:,1)/2; 
plot_pillar_heights = pillar_dims(:,3);
plot_pillar_centers = pillar_positions;
for i = 1:num_pillars
    [cX, cY, cZ] = cylinder(plot_pillar_radii(i), 20); 
    cZ = cZ * plot_pillar_heights(i); 
    pX = plot_pillar_centers(i, 1) + cX;
    pY = plot_pillar_centers(i, 2) + cY;
    surf(pX, pY, cZ, 'FaceColor', [0.8 0.1 0.1], 'EdgeColor', 'none', 'FaceAlpha', 0.8); 
end
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
            patch('Faces', drone_mesh.ConnectivityList, 'Vertices', scaled_drone_pts, 'FaceColor', colors(d,:), 'EdgeColor', 'none', 'Parent', drone_transforms(d));
        else
            scatter3(0, 0, 0, 120, colors(d,:), 'o', 'filled', 'MarkerEdgeColor', 'k', 'Parent', drone_transforms(d));
        end
        set(drone_transforms(d), 'Matrix', makehgtform('translate', path(1,:)));
    end
end
title('APF Active Collision Avoidance with Pillar Redirection');
drawnow; pause(2);

max_steps = max(cellfun(@(x) size(x, 1), all_agent_paths));
drone_yaw = zeros(num_drones, 1);   
drone_pitch = zeros(num_drones, 1); 
drone_roll = zeros(num_drones, 1);  

for step = 2:max_steps
    for d = 1:num_drones
        path = all_agent_paths{d};
        if ~isempty(path) && step <= size(path, 1)
            curr_pos = path(step, :); prev_pos = path(step-1, :); vel = curr_pos - prev_pos; 
            
            if norm(vel) < 1e-3
                target_pitch = 0; target_roll = 0;
            else
                if norm(vel(1:2)) > 1e-3
                    target_yaw = atan2(vel(2), vel(1)); target_pitch = 0.15;
                else
                    target_yaw = drone_yaw(d); target_pitch = 0; 
                end
            end
            
            yaw_error = atan2(sin(target_yaw - drone_yaw(d)), cos(target_yaw - drone_yaw(d)));
            drone_yaw(d) = drone_yaw(d) + 0.20 * yaw_error;
            target_roll = -0.6 * yaw_error; 
            drone_pitch(d) = drone_pitch(d) + 0.1 * (target_pitch - drone_pitch(d));
            drone_roll(d)  = drone_roll(d)  + 0.1 * (target_roll - drone_roll(d));
            
            T = makehgtform('translate', curr_pos) * makehgtform('zrotate', drone_yaw(d)) * makehgtform('yrotate', drone_pitch(d)) * makehgtform('xrotate', drone_roll(d));
            set(drone_transforms(d), 'Matrix', T);
        end
    end
    drawnow; pause(0.05); 
end
hold off;

% =========================================================================
% --- CALCULATE SUB-SYSTEM METRICS FOR ACADEMIC TABLE ---
% =========================================================================
min_clearance = inf(num_drones, 1);
detour_ratio = zeros(num_drones, 1);
rmse_tracking = zeros(num_drones, 1);
total_cost_val = zeros(num_drones, 1);
payoff_val = zeros(num_drones, 1);

clearance_history = cell(num_drones, 1);

for d = 1:num_drones
    path = all_agent_paths{d};
    if isempty(path), continue; end
    
    % 1. Minimum Obstacle Clearance & History Tracking
    temp_dist = zeros(size(path, 1), 1);
    for p_idx = 1:size(path, 1)
        dist_box = min_box_clearance(path(p_idx,:), obstacles);
        temp_dist(p_idx) = dist_box; 
        c = dist_box - drone_radius;
        min_clearance(d) = min(min_clearance(d), c);
    end
    clearance_history{d} = temp_dist;
    
    % 2. Path Detour Ratio
    actual_dist = 0;
    for p_idx = 2:size(path, 1)
        actual_dist = actual_dist + norm(path(p_idx,:) - path(p_idx-1,:));
    end
    
    c_id = assignments(assignments(:,1)==d, 2);
    assigned_fois = final_fois(foi_cluster_idx == c_id, :);
    num_assigned = size(assigned_fois, 1);
    sl_dist = 0;
    
    if num_assigned > 0
        all_perms = perms(1:num_assigned);
        min_route_dist = inf;
        for p = 1:size(all_perms, 1)
            current_order = all_perms(p, :); route_dist = 0; current_pos = drones_pos(d,:);
            for step = 1:num_assigned
                route_dist = route_dist + norm(assigned_fois(current_order(step), :) - current_pos);
                current_pos = assigned_fois(current_order(step), :);
            end
            route_dist = route_dist + norm(drones_pos(d,:) - current_pos);
            if route_dist < min_route_dist
                min_route_dist = route_dist; 
            end
        end
        sl_dist = min_route_dist;
    end
    if sl_dist > 0
        detour_ratio(d) = actual_dist / sl_dist;
    else
        detour_ratio(d) = 1.0;
    end
    
    % 3. Tracking Error (RMSE of Kinematic Variation)
    step_sizes = vecnorm(diff(path, 1, 1), 2, 2);
    if ~isempty(step_sizes)
        rmse_tracking(d) = sqrt(mean((step_sizes - mean(step_sizes)).^2));
    end
    
    % 4. Task Allocation
    if ~isempty(c_id)
        cost = CostMatrix(d, c_id);
        total_cost_val(d) = cost;
        payoff_val(d) = max(0, 100 - (cost / max(1, num_assigned)));
    end
end

% --- PRINT THE CONSOLE TABLE ---
fprintf('\n\n==================================================================\n');
fprintf('%-25s %-32s %-8s %-8s\n', 'Subsystem', 'Metric', 'Agent 1', 'Agent 2');
fprintf('------------------------------------------------------------------\n');
fprintf('%-25s %-32s %-8.3f %-8.3f\n', 'Obstacle Avoidance', 'Minimum Obstacle Clearance', min_clearance(1), min_clearance(2));
fprintf('%-25s %-32s %-8.3f %-8.3f\n', '', 'Path Detour Ratio', detour_ratio(1), detour_ratio(2));
fprintf('%-25s %-32s %-8.3f %-8.3f\n', '', 'Tracking Error (RMSE)', rmse_tracking(1), rmse_tracking(2));
fprintf('------------------------------------------------------------------\n');
fprintf('%-25s %-32s %-8.2f %-8.2f\n', 'Task Allocation', 'Total Cost', total_cost_val(1), total_cost_val(2));
fprintf('%-25s %-32s %-8.2f %-8.2f\n', '', 'Payoff', payoff_val(1), payoff_val(2));
fprintf('==================================================================\n\n');

% =========================================================================
% --- GENERATE ACADEMIC TELEMETRY GRAPHS ---
% =========================================================================
time_vector = (0:(size(d1_forces, 1)-1)) * params.dt;
% --- NEW: Reconfigured figure layout for a 2x2 Grid ---
figure('Name', 'APF Engineering Telemetry', 'Color', 'w', 'Position', [100, 100, 1200, 800]);

% 1. Force Analysis Plot
subplot(2, 2, 1); hold on; grid on;
plot(time_vector, d1_forces(:, 1), 'g', 'LineWidth', 2, 'DisplayName', 'F_{att}');
plot(time_vector, d1_forces(:, 2), 'r', 'LineWidth', 2, 'DisplayName', 'F_{rep}');
plot(time_vector, d1_forces(:, 3), 'k', 'LineWidth', 2, 'DisplayName', 'F_{total}');
title('Force Analysis', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time (s)', 'FontSize', 12);
ylabel('Force Magnitude', 'FontSize', 12);
legend('Location', 'northeast', 'FontSize', 11);
set(gca, 'FontSize', 11);

% 2. Computation Time Plot
subplot(2, 2, 2); hold on; grid on;
iterations_to_show = min(150, length(d1_comp_times)); 
plot(1:iterations_to_show, d1_comp_times(1:iterations_to_show), 'k', 'LineWidth', 2.5);
title('Computation Time per Iteration', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Iteration', 'FontSize', 12);
ylabel('Time (ms)', 'FontSize', 12);
set(gca, 'FontSize', 11);

% 3. Safety Validation: Minimum Clearance Plot
subplot(2, 2, 3); hold on; grid on;
for d = 1:num_drones
    if ~isempty(clearance_history{d})
        plot(1:length(clearance_history{d}), clearance_history{d}, 'LineWidth', 2, 'DisplayName', sprintf('Drone %d', d));
    end
end
max_plot_steps = max(cellfun(@length, clearance_history));
if max_plot_steps > 0
    plot([1, max_plot_steps], [drone_radius, drone_radius], '--r', 'LineWidth', 2, 'DisplayName', 'Collision Threshold');
    text(5, drone_radius + 0.5, 'Collision Threshold', 'Color', 'r', 'FontSize', 11, 'FontWeight', 'bold');
end
title('Safety Validation: Minimum Clearance', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Simulation Step', 'FontSize', 12);
ylabel('Distance to Obstacle (m)', 'FontSize', 12);
legend('Location', 'northeast', 'FontSize', 11);
set(gca, 'FontSize', 11);

% 4. Convergence Analysis Plot (NEW)
subplot(2, 2, 4); hold on; grid on;
plot_colors = {'b', 'r', 'g', 'm'};
max_time = 0;
for d = 1:num_drones
    if ~isempty(all_agent_paths{d}) && ~isempty(all_agent_goals{d})
        path = all_agent_paths{d};
        goals = all_agent_goals{d};
        % Calculate euclidean distance between current position and active goal
        dist_to_goal = vecnorm(path - goals, 2, 2);
        % Map steps to estimated Time (s)
        time_vec = (0:length(dist_to_goal)-1) * params.dt;
        max_time = max(max_time, max(time_vec));
        plot(time_vec, dist_to_goal, 'Color', plot_colors{d}, 'LineWidth', 2.5, 'DisplayName', sprintf('Drone %d Distance to Goal', d));
    end
end
if max_time > 0
    plot([0, max_time], [params.goal_threshold, params.goal_threshold], '--g', 'LineWidth', 2, 'DisplayName', 'Goal Threshold');
end
title('Convergence Analysis', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time (s)', 'FontSize', 12);
ylabel('Distance to Goal (m)', 'FontSize', 12);
legend('Location', 'northeast', 'FontSize', 11);
set(gca, 'FontSize', 11);

end

%% =========================================================
%% APF HELPER FUNCTIONS
%% =========================================================
function obstacles = generate_obstacle_field(vertices, faces)
    num_obstacles = 50; sample_points = zeros(1500, 3);
    for i = 1:1500
        face_idx = randi(size(faces, 1)); face_verts = vertices(faces(face_idx, :), :);
        a = rand; b = rand; if a + b > 1, a = 1 - a; b = 1 - b; end; c = 1 - a - b;
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
            obstacles(i, 1:3) = centers(i, :); obstacles(i, 4:6) = [1.0, 1.0, 1.0];
        end
    end
end

function [dist_box, nearest_pt, dir_vec] = point_to_box_distance(p, obs)
    cx = obs(1); cy = obs(2); cz = obs(3); w = obs(4); d = obs(5); h = obs(6);
    xmin = cx - w/2; xmax = cx + w/2; ymin = cy - d/2; ymax = cy + d/2; zmin = cz - h/2; zmax = cz + h/2;
    nearest_pt = [min(max(p(1), xmin), xmax), min(max(p(2), ymin), ymax), min(max(p(3), zmin), zmax)];
    diff = p - nearest_pt; dist_box = norm(diff);
    if dist_box < 1e-8, diff = p - [cx cy cz]; if norm(diff) < 1e-8, diff = [1 0 0]; end; dir_vec = diff / norm(diff); dist_box = 0;
    else, dir_vec = diff / dist_box; end
end

function [segment, force_log, comp_times] = apf_segment(start, goal, obstacles, params, drone_radius, hangar_len, hangar_wid, hangar_ht)
    segment = []; current = start(:); goal = goal(:);
    force_log = []; comp_times = [];
    max_force = 25.5;
    for iter = 1:params.max_iterations
        t_start = tic; 
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
                mag = mag * max(0.20, min(1.0, (dist_to_goal / 2.0)^2));
                F_rep = F_rep + mag * dir_vec(:);
            end
        end
        F_total = F_att + F_rep;
        if norm(F_total) > max_force, F_total = F_total / norm(F_total) * max_force; end
        
        force_log(end+1, :) = [norm(F_att), norm(F_rep), norm(F_total)];
        comp_times(end+1, 1) = toc(t_start) * 1000; 
        
        candidate = current + F_total * params.dt;
        candidate(1) = max(0, min(hangar_len, candidate(1)));
        candidate(2) = max(0, min(hangar_wid, candidate(2)));
        candidate(3) = max(0, min(hangar_ht, candidate(3)));
        cand_clearance = inf;
        for ii = 1:size(obstacles,1)
            [d_box,~,~] = point_to_box_distance(candidate', obstacles(ii,:));
            cand_clearance = min(cand_clearance, d_box);
        end
        safety_margin = 0.35; 
        if cand_clearance < (drone_radius + safety_margin)
            if norm(F_rep) > 1e-6, emergency_dir = F_rep / norm(F_rep); else, emergency_dir = randn(3,1); emergency_dir = emergency_dir / norm(emergency_dir); end
            candidate = current + 1.5 * emergency_dir;
            candidate(1) = max(0, min(hangar_len, candidate(1))); candidate(2) = max(0, min(hangar_wid, candidate(2))); candidate(3) = max(0, min(hangar_ht, candidate(3)));
        end
        current = candidate; segment = [segment; current'];
        if norm(current - goal) < params.goal_threshold
            segment = [segment; goal']; break;
        end
        if min_dist < drone_radius * 1.3 && params.escape_force > 0
            escape_dir = randn(3,1); escape_dir = escape_dir / norm(escape_dir);
            current = current + escape_dir * params.escape_force * params.dt;
            current(1) = max(0, min(hangar_len, current(1))); current(2) = max(0, min(hangar_wid, current(2))); current(3) = max(0, min(hangar_ht, current(3)));
        end
    end
end

function path_out = simplify_path(path_in, min_spacing)
    if size(path_in,1) <= 2, path_out = path_in; return; end
    window = min(35, size(path_in,1)); smoothed = path_in;
    smoothed(:,1) = smoothdata(path_in(:,1), 'gaussian', window); smoothed(:,2) = smoothdata(path_in(:,2), 'gaussian', window); smoothed(:,3) = smoothdata(path_in(:,3), 'gaussian', window);
    smoothed(1,:) = path_in(1,:); smoothed(end,:) = path_in(end,:);
    keep_idx = 1; last_kept = smoothed(1,:);
    for i = 2:size(smoothed,1)-1
        if norm(smoothed(i,:) - last_kept) >= min_spacing, keep_idx(end+1) = i; last_kept = smoothed(i,:); end
    end
    keep_idx(end+1) = size(smoothed,1); path_out = smoothed(keep_idx,:);
end

function c = min_box_clearance(p, obstacles)
    c = inf;
    for ii = 1:size(obstacles,1)
        [dist_box,~,~] = point_to_box_distance(p, obstacles(ii,:));
        c = min(c, dist_box);
    end
end