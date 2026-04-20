function hangar_inspection_sim()
% AIRCRAFT HANGAR INSPECTION: SAFE 3D WAYPOINT & DWA AVOIDANCE
clc; clear; close all;

%% 1. CONFIGURABLE HANGAR & WEIGHTS
hangar_length = 50; 
hangar_width = 50;  
hangar_height = 20;  
w_dist = 1.0; w_battery = 1.0;

%% 2. LOAD AND SHIFT AIRCRAFT
model = stlread('airplane.STL');
original_points = model.Points;
min_coords = min(original_points);
shifted_points = original_points - min_coords + [10, 10, 0];

%% 3. SURFACE FILTERING & MANUAL FOI DEFINITION
TR = triangulation(model.ConnectivityList, shifted_points);
normals = faceNormal(TR); incenters = incenter(TR);
z_threshold = 0.25 * max(shifted_points(:,3)); 
valid_mask = (normals(:,3) > 0.2) & (incenters(:,3) > z_threshold);
valid_points = incenters(find(valid_mask), :);

user_defined_fois = [
    42.0, 25.0, 12.0;  % Nose
    45.0, 25.0, 4.0;
    25.0, 25.0, 16.0;  % Mid-Fuselage
    25.0, 15.0, 2.5;   % Right-Wing
    25.0, 10.0, 2.5;
    27.5, 17.5, 2.5;
    25.0, 42.0, 2.5;   % Left-Wing
    27.5, 34.5, 2.5;
    11.0, 28.0, 5.0;
    15.0, 40.0, 11.0;  % Vertical Stabilizer
    11.0, 17.0, 5.0;
    13.0, 30.0, 15.0;  % Left Horizontal Stabilizer
    13.0, 18.0, 15.0   % Right Horizontal Stabilizer
];

final_fois = zeros(size(user_defined_fois, 1), 3);
for i = 1:size(user_defined_fois, 1)
    [~, min_idx] = min(sum((valid_points - user_defined_fois(i,:)).^2, 2));
    final_fois(i,:) = valid_points(min_idx, :);
end
fprintf('Successfully loaded %d manual FOIs.\n', size(final_fois, 1));

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

%% 5. TSP ROUTING WITH SAFE ALTITUDE & INSPECTION DWELL
all_agent_paths = cell(num_drones, 1);
transit_z_high = 3; % Safe ceiling for the (0,0,0) drone
hover_offset = 1.5;    % Hover 1.5 meters ABOVE the FOI

for k = 1:size(assignments, 1)
    d_id = assignments(k, 1); c_id = assignments(k, 2); 
    home_pos = drones_pos(d_id, :);
    assigned_fois = final_fois(foi_cluster_idx == c_id, :);
    num_assigned = size(assigned_fois, 1);
    
    if num_assigned > 0
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
        safe_path = [home_pos]; 
        
        if isequal(home_pos, [0, 0, 0])
            safe_path = [safe_path; home_pos(1), home_pos(2), transit_z_high]; 
            for step = 1:num_assigned
                target_foi = ordered_points(step, :);
                hover_z = target_foi(3) + hover_offset; 
                safe_path = [safe_path; target_foi(1), target_foi(2), transit_z_high]; 
                safe_path = [safe_path; target_foi(1), target_foi(2), hover_z];        
                safe_path = [safe_path; target_foi(1), target_foi(2), hover_z]; % DWELL       
                safe_path = [safe_path; target_foi(1), target_foi(2), transit_z_high]; 
            end
            safe_path = [safe_path; home_pos(1), home_pos(2), transit_z_high]; 
        else
            for step = 1:num_assigned
                target_foi = ordered_points(step, :);
                hover_z = target_foi(3) + hover_offset; 
                safe_path = [safe_path; safe_path(end, 1), safe_path(end, 2), hover_z]; 
                safe_path = [safe_path; target_foi(1), target_foi(2), hover_z];         
                safe_path = [safe_path; target_foi(1), target_foi(2), hover_z]; % DWELL      
            end
            safe_path = [safe_path; home_pos(1), home_pos(2), safe_path(end, 3)]; 
        end
        safe_path = [safe_path; home_pos]; 
        all_agent_paths{d_id} = safe_path;
    else
        all_agent_paths{d_id} = [];
    end
end

%% 6. 2.5D DWA OBSTACLE AVOIDANCE & KINEMATIC SIMULATION
fprintf('\n--- INITIALIZING 2.5D DWA LOCAL PLANNER ---\n');

% DWA Parameters (Tuned for Hangar)
cfg.inflate = 1.0;      
cfg.goal_tol = 0.5;     
cfg.dt = 0.1;           
cfg.sim_dt = 0.1;
cfg.v_min = 0.0;  cfg.v_max = 5.0; 
cfg.w_max = 3.0;
cfg.a_max = 4.0;  cfg.alpha_max = 4.0;
cfg.L_open = 2.0; cfg.L_tight = 1.0;
cfg.pred_time_open = 2.0; cfg.pred_time_tight = 1.0;
cfg.Nv_base = 15; cfg.Nw_base = 31;

% Define Fuselage Obstacles (Blocks direct paths through the cabin/tail)
obs_x = linspace(10, 45, 15)'; 
obs_y = ones(15, 1) * 25;      
obs_r = ones(15, 1) * 2.5;     
obs_2d = [obs_x, obs_y, obs_r];

% Initialize Drone States for DWA
agent_states = repmat(struct('p', [], 'z', 0, 'psi', 0, 'v', 0, 'w', 0, ...
    'path_2d', [], 'path_3d', [], 'path_s', [], 'target_idx', 2, ...
    'done', false, 'dwell_timer', 0), num_drones, 1);

for d = 1:num_drones
    if ~isempty(all_agent_paths{d})
        path_3d = all_agent_paths{d};
        path_2d = path_3d(:, 1:2);
        agent_states(d).path_3d = path_3d;
        agent_states(d).path_2d = path_2d;
        agent_states(d).path_s = [0; cumsum(vecnorm(diff(path_2d, 1, 1), 2, 2))];
        agent_states(d).p = path_2d(1, :);
        agent_states(d).z = path_3d(1, 3);
        if size(path_2d, 1) > 1
            agent_states(d).psi = atan2(path_2d(2,2)-path_2d(1,2), path_2d(2,1)-path_2d(1,1));
        end
    else
        agent_states(d).done = true;
    end
end

% Visualization Setup
figure('Name', 'SIL DWA Simulation (Decoupled 2.5D)', 'Color', 'w', 'Position', [100, 100, 1000, 800]); 
hold on;
patch('Faces', model.ConnectivityList, 'Vertices', shifted_points, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
camlight('headlight'); material('shiny'); lighting('gouraud');
xlim([0, hangar_length]); ylim([0, hangar_width]); zlim([0, hangar_height]);
daspect([1 1 1]); view(45, 30); grid on; rotate3d on;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (Altitude)');

% Draw Obstacles (Rings mapping the physical fuselage keep-out zone)
% th = linspace(0,2*pi,30);
% for i=1:size(obs_2d,1)
%     % Lowered the Z-coordinate to 5 meters to wrap the actual cabin
%     plot3(obs_2d(i,1)+obs_2d(i,3)*cos(th), obs_2d(i,2)+obs_2d(i,3)*sin(th), ones(1,30)*5, 'r-', 'LineWidth', 1.5);
% end

hDrones = gobjects(num_drones, 1);
for d = 1:num_drones
    if ~agent_states(d).done
        plot3(agent_states(d).path_3d(:,1), agent_states(d).path_3d(:,2), agent_states(d).path_3d(:,3), ...
            '-', 'Color', [colors(d,:) 0.4], 'LineWidth', 1.5);
        hDrones(d) = scatter3(agent_states(d).p(1), agent_states(d).p(2), agent_states(d).z, ...
            150, colors(d,:), 'filled', 'o', 'MarkerEdgeColor', 'k');
    end
end
title('Initiating DWA Local Obstacle Avoidance...');
drawnow; pause(1);

% --- ACTIVE KINEMATIC LOOP ---
max_sim_steps = 3000;
all_done = false;

for step = 1:max_sim_steps
    if all_done, break; end
    all_done = true;
    any_dwelling = false;
    
    for d = 1:num_drones
        if agent_states(d).done, continue; end
        all_done = false; 
        
        st = agent_states(d);
        local_goal_3d = st.path_3d(st.target_idx, :);
        local_goal_2d = local_goal_3d(1:2);
        
        dist_2d = norm(st.p - local_goal_2d);
        dist_z  = abs(st.z - local_goal_3d(3));
        
        % Detect Zero-Distance Dwell Waypoint
        is_dwelling = false;
        if st.target_idx > 1 && isequal(local_goal_3d, st.path_3d(st.target_idx - 1, :))
            is_dwelling = true;
        end
        
        % Check if waypoint is reached
        if dist_2d <= cfg.goal_tol && dist_z <= 0.2
            if is_dwelling
                if st.dwell_timer < 5.0 % 5-second simulated hold
                    st.dwell_timer = st.dwell_timer + cfg.sim_dt;
                    any_dwelling = true;
                    set(hDrones(d), 'MarkerFaceColor', 'y', 'SizeData', 250);
                else
                    st.target_idx = st.target_idx + 1;
                    st.dwell_timer = 0;
                    set(hDrones(d), 'MarkerFaceColor', colors(d,:), 'SizeData', 150);
                end
            else
                st.target_idx = st.target_idx + 1;
            end
            if st.target_idx > size(st.path_3d, 1)
                st.done = true; st.v = 0; st.w = 0;
            end
        end
        
        % Movement execution
        if ~st.done
            if ~is_dwelling
                % X/Y DWA Avoidance
                if dist_2d > cfg.goal_tol
                    [v_cmd, w_cmd, ~, ~, ~] = computeCmdDWA(st, obs_2d, st.path_2d, st.path_s, local_goal_2d, cfg);
                    st.v = clamp(st.v + clamp(v_cmd - st.v, -cfg.a_max*cfg.dt, cfg.a_max*cfg.dt), cfg.v_min, cfg.v_max);
                    st.w = clamp(st.w + clamp(w_cmd - st.w, -cfg.alpha_max*cfg.dt, cfg.alpha_max*cfg.dt), -cfg.w_max, cfg.w_max);
                    st.psi = wrapToPi(st.psi + st.w * cfg.dt);
                    st.p = st.p + [st.v * cos(st.psi), st.v * sin(st.psi)] * cfg.dt;
                else
                    st.v = 0; st.w = 0;
                end
                
                % Z-Axis Independent Tracking
                if dist_z > 0.1
                    vz = 3.0; 
                    st.z = st.z + sign(local_goal_3d(3) - st.z) * min(vz * cfg.dt, dist_z);
                end
            end
        end
        
        agent_states(d) = st;
        set(hDrones(d), 'XData', st.p(1), 'YData', st.p(2), 'ZData', st.z);
    end
    
    % Render every 4th step for smooth speed
    if mod(step, 15) == 0
        if any_dwelling
            title(sprintf('DWA Active: [CAMERA ACTIVE - 5s HOLD] | Step %d', step));
        else
            title(sprintf('DWA Active: Decoupled Target Tracking & Obstacle Avoidance | Step %d', step));
        end
        drawnow;
    end
end
title('Mission Success: Safe Navigation Complete');
disp('Mission Complete. All agents successfully navigated the hangar.');

end % CLOSES MAIN FUNCTION

%% ======================================================================
%% DWA HELPER FUNCTIONS
%% ======================================================================

function [v_cmd, w_cmd, carrot, best_traj, dbg] = computeCmdDWA(pose, obstacles, globalPath, path_s, goal, cfg)
    p = pose.p; psi = pose.psi; v = pose.v; w = pose.w;
    clear_now = minClearancePointCircles(p, obstacles, cfg.inflate);
    dbg.clear_now = clear_now;

    if clear_now < 0.9
        L = cfg.L_tight; predT = cfg.pred_time_tight;
        Nv = max(cfg.Nv_base, 15); Nw = max(cfg.Nw_base, 35);
    else
        L = cfg.L_open;  predT = cfg.pred_time_open;
        Nv = cfg.Nv_base; Nw = cfg.Nw_base;
    end

    s_now = pathProgressArc(p, globalPath, path_s);
    carrot = pointAtArc(globalPath, path_s, s_now + L);

    v_lo = max(cfg.v_min, v - cfg.a_max*cfg.dt);
    v_hi = min(cfg.v_max, v + cfg.a_max*cfg.dt);
    w_lo = max(-cfg.w_max, w - cfg.alpha_max*cfg.dt);
    w_hi = min( cfg.w_max, w + cfg.alpha_max*cfg.dt);
    
    v_hi = max(v_hi, 0.25); v_lo = min(v_lo, 0.10);

    v_s = linspace(v_lo, v_hi, Nv); w_s = linspace(w_lo, w_hi, Nw);
    bestScore = -Inf; v_best = 0.2; w_best = 0.0; best_traj = [];

    for vi = v_s
        for wi = w_s
            [traj, minclr, p_end, psi_end] = rolloutCircles(p, psi, vi, wi, predT, cfg.sim_dt, obstacles, cfg.inflate);
            if minclr <= 0, continue; end

            s_clear = min(minclr/1.5, 1.0);
            angC = atan2(carrot(2)-p_end(2), carrot(1)-p_end(1));
            s_head = 1 - abs(wrapToPi(angC - psi_end))/pi;
            s_dist = 1/(1+norm(carrot - p_end));
            s_speed = (vi - cfg.v_min)/(cfg.v_max - cfg.v_min + 1e-6);
            s_goal  = 1/(1+norm(goal - p_end));
            s_prog = clamp((pathProgressArc(p_end, globalPath, path_s) - s_now)/0.35, -1, 1);
            rot_pen = abs(wi)/cfg.w_max;

            w_clear = (clear_now < 0.9) * 5.0 + (clear_now >= 0.9) * 2.0;
            w_speed = (clear_now < 0.9) * 0.15 + (clear_now >= 0.9) * 1.0;

            score = 2.0*s_head + 1.6*s_dist + w_clear*s_clear + w_speed*s_speed + 1.2*s_goal + 4.0*s_prog - 0.25*rot_pen;
            if vi < 0.08, score = score - 1.0; end

            if score > bestScore
                bestScore = score; v_best = vi; w_best = wi; best_traj = traj;
            end
        end
    end

    if ~isfinite(bestScore)
        [v_best, w_best] = fallbackTurnAndCreep(p, psi, carrot, obstacles, cfg.inflate, cfg.w_max);
    end
    v_cmd = v_best; w_cmd = w_best;
end

function [traj, minclr, p_end, psi_end] = rolloutCircles(p0, psi0, v, w, T, dt, obs, inflate)
    N = max(1, round(T/dt)); traj = zeros(N,2); p = p0; psi = psi0; minclr = Inf;
    for k=1:N
        psi = wrapToPi(psi + w*dt);
        p = p + [v*cos(psi), v*sin(psi)]*dt;
        traj(k,:) = p;
        for i=1:size(obs,1)
            minclr = min(minclr, norm(p-obs(i,1:2)) - (obs(i,3)+inflate));
        end
        if minclr <= 0, break; end
    end
    p_end = p; psi_end = psi;
end

function d = minClearancePointCircles(p, obs, inflate)
    d = Inf;
    for i=1:size(obs,1), d = min(d, norm(p-obs(i,1:2)) - (obs(i,3)+inflate)); end
end

function [v_cmd, w_cmd] = fallbackTurnAndCreep(p, psi, carrot, obs, inflate, w_max)
    e = wrapToPi(atan2(carrot(2)-p(2), carrot(1)-p(1)) - psi);
    w_cmd = clamp(1.8*e, -w_max, w_max); v_cmd = 0.12;
    if minClearancePointCircles(p, obs, inflate) < 0.25
        v_cmd = 0.0; w_cmd = 0.6*sign(e + 1e-6);
    end
end

function s = pathProgressArc(p, path, path_dists)
    best_d2 = Inf; best_s = 0;
    for i=1:size(path,1)-1
        a = path(i,:); b = path(i+1,:); ab = b-a; L2 = dot(ab,ab);
        if L2 < 1e-12, continue; end
        t = clamp(dot(p-a, ab)/L2, 0, 1);
        d2 = sum((p-(a + t*ab)).^2);
        if d2 < best_d2, best_d2 = d2; best_s = path_dists(i) + t*sqrt(L2); end
    end
    s = best_s;
end

function pt = pointAtArc(path, path_dists, s_query)
    s_query = clamp(s_query, 0, path_dists(end));
    for i=1:length(path_dists)-1
        if path_dists(i) <= s_query && path_dists(i+1) >= s_query
            seg = (s_query - path_dists(i)) / (path_dists(i+1)-path_dists(i)+1e-12);
            pt = path(i,:) + seg*(path(i+1,:)-path(i,:)); return;
        end
    end
    pt = path(end,:);
end

function y = clamp(x, lo, hi), y = min(max(x, lo), hi); end
function a = wrapToPi(a), a = mod(a+pi, 2*pi) - pi; end