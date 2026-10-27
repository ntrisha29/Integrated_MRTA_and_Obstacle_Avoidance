function hangar_inspection_apf_cad()
% AIRCRAFT HANGAR INSPECTION: DIRECT HOPPING APF WITH CAD KINEMATICS
clc; clear; close all;

%% 1. CONFIGURABLE PARAMETERS
hangar_length = 50; 
hangar_width = 50;  
hangar_height = 20;  

% Task Allocation Weights
w_dist = 1.0; 
w_battery = 1.0;

% APF Collision Avoidance Parameters
params = struct();
params.attraction_gain = 1.2;      % Slightly stronger pull to target
params.repulsion_gain = 15.0;      % Softer push from aircraft
params.repulsion_distance = 1.0;   % Smaller forcefield (was 6.0)
params.dt = 0.15;                  % Time step 
params.max_iterations = 1500;      % Max steps per segment
params.goal_threshold = 0.2;       % Tighter waypoint arrival tolerance
params.escape_force = 0.5;         % Reduced panic force
drone_radius = 0.75;               
min_waypoint_spacing = 0.5;        % Path simplification

%% 2. LOAD AIRCRAFT, GENERATE OBSTACLES, & LOAD CAD DRONE
fprintf('Loading aircraft and generating APF obstacle field...\n');
model = stlread('airplane.STL');
original_points = model.Points;
min_coords = min(original_points);
shifted_points = original_points - min_coords + [10, 10, 0];
aircraft_faces = model.ConnectivityList;

% Generate the APF bounding boxes representing the aircraft
obstacles = generate_obstacle_field(shifted_points, aircraft_faces);

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

%% 3. SURFACE FILTERING & MANUAL FOI DEFINITION
TR = triangulation(model.ConnectivityList, shifted_points);
normals = faceNormal(TR); incenters = incenter(TR);
z_threshold = 0.25 * max(shifted_points(:,3)); 
valid_mask = (normals(:,3) > 0.2) & (incenters(:,3) > z_threshold);
valid_points = incenters(valid_mask, :);

% custom 13 FOIs
user_defined_fois = [
    42.0, 25.0, 5.0;   45.0, 25.0, 4.0;   25.0, 25.0, 5.0;  
    25.0, 15.0, 2.5;   25.0, 10.0, 2.5;   27.5, 17.5, 2.5;
    25.0, 42.0, 2.5;   27.5, 34.5, 2.5;   27.5, 31.5, 2.5;
    11.0, 28.0, 5.0;   15.0, 40.0, 11.0;  11.0, 17.0, 5.0;   
    21.0, 18.0, 15.0;  13.0, 20.0, 3.0  
];

final_fois = zeros(size(user_defined_fois, 1), 3);
for i = 1:size(user_defined_fois, 1)
    [~, min_idx] = min(sum((valid_points - user_defined_fois(i,:)).^2, 2));
    final_fois(i,:) = valid_points(min_idx, :);
end

%% 4. DEFINE AGENTS & HUNGARIAN ALLOCATION
num_drones = 5;
drones_pos = [0, 0, 0; 50, 0, 0; 0, 50, 0; 50, 50, 0; 25, 0, 0];
drone_battery = [80; 80; 83; 85; 79]; 
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
transit_z_high = 8;  
hover_offset = 2;

for k = 1:size(assignments, 1)
    d_id = assignments(k, 1); c_id = assignments(k, 2); 
    home_pos = drones_pos(d_id, :);
    assigned_fois = final_fois(foi_cluster_idx == c_id, :);
    num_assigned = size(assigned_fois, 1);
    
    if num_assigned > 0
        % 1. TSP Optimization
        all_perms = perms(1:num_assigned);
        min_route_dist = inf; best_route = [];
        for p = 1:size(all_perms, 1)
            current_order = all_perms(p, :); route_dist = 0; current_pos = home_pos;
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
        
        % 2. APF Path Generation
        full_path = home_pos;
        current = home_pos';
        
        % Takeoff to safe ceiling
        waypoint = [home_pos(1); home_pos(2); transit_z_high];
        segment = apf_segment(current, waypoint, obstacles, params, drone_radius, hangar_length, hangar_width, hangar_height);
        segment = simplify_path(segment, min_waypoint_spacing); % Simplify BEFORE adding dwells
        full_path = [full_path; segment]; current = full_path(end, :)';
        
        % Hop directly between FOIs
        for step = 1:num_assigned
            target_foi = ordered_points(step, :);
            hover_z = target_foi(3) + hover_offset; 
            waypoint = [target_foi(1); target_foi(2); hover_z];
            
            % Physics-based flight to target
            segment = apf_segment(current, waypoint, obstacles, params, drone_radius, hangar_length, hangar_width, hangar_height);
            segment = simplify_path(segment, min_waypoint_spacing); % Simplify transit
            
            % FORCE EXACT ARRIVAL: Append the exact 1.0m coordinate to fix APF gap
            segment = [segment; waypoint'];
            full_path = [full_path; segment]; 
            current = waypoint;
            
            % DWELL: Guaranteed 5-Second Inspection Hold
            % (150 simulation frames * 0.033 sec visual pause = ~5 real seconds)
            dwell_frames = repmat(current', 150, 1);
            full_path = [full_path; dwell_frames];
        end
        
        % Return Home
        waypoint = [home_pos(1); home_pos(2); transit_z_high];
        segment = apf_segment(current, waypoint, obstacles, params, drone_radius, hangar_length, hangar_width, hangar_height);
        segment = simplify_path(segment, min_waypoint_spacing);
        full_path = [full_path; segment]; current = full_path(end, :)';
        
        waypoint = [home_pos(1); home_pos(2); home_pos(3)];
        segment = apf_segment(current, waypoint, obstacles, params, drone_radius, hangar_length, hangar_width, hangar_height);
        segment = simplify_path(segment, min_waypoint_spacing);
        full_path = [full_path; segment];
        
        all_agent_paths{d_id} = full_path;
    else
        all_agent_paths{d_id} = [];
    end
end
%% 6. 3D CAD VISUALIZATION & DYNAMIC KINEMATICS
figure('Name', 'APF Direct Hopping Simulation', 'Color', 'w', 'Position', [100, 100, 1000, 800]);
hold on;
patch('Faces', model.ConnectivityList, 'Vertices', shifted_points, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
camlight('headlight'); material('shiny'); lighting('gouraud');
xlim([0, hangar_length]); ylim([0, hangar_width]); zlim([0, hangar_height]);
daspect([1 1 1]); view(45, 30); grid on; rotate3d on;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (Altitude)');

% % Plot the invisible APF bounding boxes (Uncomment to see them)
% for i = 1:size(obstacles, 1)
%     plot_obstacle_cube(obstacles(i, :), [0.8, 0.1, 0.1], 0.1);
% end

for d = 1:num_drones
    scatter3(drones_pos(d,1), drones_pos(d,2), drones_pos(d,3), 150, colors(d,:), 's', 'filled');
end

% Plot FOI markers
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

title('APF Active Collision Avoidance');
drawnow; pause(2);

% --- APF DENSE-PATH KINEMATIC LOOP ---
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
            vel = curr_pos - prev_pos; % Velocity vector generated by APF
            
            if norm(vel) < 1e-3
                % Dwelling
                target_pitch = 0; target_roll = 0;
            else
                if norm(vel(1:2)) > 1e-3
                    target_yaw = atan2(vel(2), vel(1));
                    target_pitch = 0.15; % Tilt forward based on movement
                else
                    % Vertical movement
                    target_yaw = drone_yaw(d); 
                    target_pitch = 0; 
                end
            end
            
            % Smooth Angular Math
            yaw_error = atan2(sin(target_yaw - drone_yaw(d)), cos(target_yaw - drone_yaw(d)));
            drone_yaw(d) = drone_yaw(d) + 0.20 * yaw_error;
            target_roll = -0.6 * yaw_error; 
            drone_pitch(d) = drone_pitch(d) + 0.1 * (target_pitch - drone_pitch(d));
            drone_roll(d)  = drone_roll(d)  + 0.1 * (target_roll - drone_roll(d));
            
            % Apply Transform
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

%% APF HELPER FUNCTIONS
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
            
            % REDUCED PADDING: Changed + 2.0 to + 0.5 for much tighter boxes
            obstacles(i, 4) = max(cluster_pts(:,1)) - min(cluster_pts(:,1)) + 0.5;
            obstacles(i, 5) = max(cluster_pts(:,2)) - min(cluster_pts(:,2)) + 0.5;
            obstacles(i, 6) = max(cluster_pts(:,3)) - min(cluster_pts(:,3)) + 0.5;
        else
            obstacles(i, 1:3) = centers(i, :);
            obstacles(i, 4:6) = [1.0, 1.0, 1.0]; % Reduced fallback box size
        end
    end
end

function plot_obstacle_cube(obs, color, alpha)
    x = obs(1); y = obs(2); z = obs(3);
    w = obs(4); d = obs(5); h = obs(6);
    verts = [x-w/2, y-d/2, z-h/2; x+w/2, y-d/2, z-h/2; x+w/2, y+d/2, z-h/2; x-w/2, y+d/2, z-h/2;
             x-w/2, y-d/2, z+h/2; x+w/2, y-d/2, z+h/2; x+w/2, y+d/2, z+h/2; x-w/2, y+d/2, z+h/2];
    faces = [1,2,6,5; 2,3,7,6; 3,4,8,7; 4,1,5,8; 1,2,3,4; 5,6,7,8];
    patch('Vertices', verts, 'Faces', faces, 'FaceColor', color, 'EdgeColor', 'none', 'FaceAlpha', alpha);
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
    max_force = 3.5;
    for iter = 1:params.max_iterations
        F_att = params.attraction_gain * (goal - current);
        F_rep = [0; 0; 0];
        min_dist = inf;
        dist_to_goal = norm(current - goal); % NEW: Track distance to goal
        
        for i = 1:size(obstacles, 1)
            [dist_box, ~, dir_vec] = point_to_box_distance(current', obstacles(i,:));
            min_dist = min(min_dist, dist_box);
            clearance = dist_box - drone_radius;
            if clearance < params.repulsion_distance
                clearance = max(clearance, 0.05);
                mag = params.repulsion_gain * (1/clearance - 1/params.repulsion_distance)^2;
                
                % NEW GNRON FIX: Weaken repulsion as we get very close to the goal
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
    if size(path_in,1) <= 2
        path_out = path_in; 
        return; 
    end
    
    % 1. GAUSSIAN SMOOTHING (The "Aerodynamic" Fix)
    % Averages out the APF micro-vibrations for beautiful sweeping curves
    window = min(15, size(path_in,1)); 
    smoothed = path_in;
    smoothed(:,1) = smoothdata(path_in(:,1), 'gaussian', window);
    smoothed(:,2) = smoothdata(path_in(:,2), 'gaussian', window);
    smoothed(:,3) = smoothdata(path_in(:,3), 'gaussian', window);
    
    % Lock the exact start and end points so we don't miss the FOI
    smoothed(1,:) = path_in(1,:);
    smoothed(end,:) = path_in(end,:);
    
    % 2. SPACING SIMPLIFICATION
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