function demo_dwa()

clear; clc; close all;

%% ==================== CONFIG ====================
cfg.inflate = 0.50;      % SAFETY RADIUS
cfg.goal_tol = 0.18;

cfg.dt = 0.10;
cfg.maxSteps = 5000;

cfg.v_min = 0.0;  cfg.v_max = 1.2;
cfg.w_max = 2.2;
cfg.a_max = 2.5;
cfg.alpha_max = 4.0;

cfg.sim_dt = 0.06;
cfg.pred_time_open  = 1.6;
cfg.pred_time_tight = 1.0;

cfg.Nv_base = 13;
cfg.Nw_base = 31;

% Lookahead
cfg.L_open = 1.6;
cfg.L_tight = 0.8;

% Room bounds
sim.room = [0 10 0 6];

% Start/Goal
sim.p0   = [1.0 1.0];
sim.goal = [8.5 5.0];

% Obstacles [cx cy r]
sim.obs = [ ...
    2.8, 4.3, 0.35;
    3.6, 2.2, 0.35;
    5.6, 4.2, 0.40;
    6.6, 2.0, 0.40;
    7.8, 3.6, 0.35];

fprintf('DWA obstacle avoidance core. inflate=%.2f\n', cfg.inflate);

%% ==================== GLOBAL PATH (Sim A*) ====================
sim.globalPath = getGlobalPathSim(sim, cfg);

% Precompute arc-length for carrot/progress
sim.path_s = [0; cumsum(vecnorm(diff(sim.globalPath,1,1),2,2))];

%% ==================== VISUALIZATION ====================
figure(1); clf; hold on; axis equal; grid on;
xlim([sim.room(1) sim.room(2)]); 
ylim([sim.room(3) sim.room(4)]);
title('Obstacle Avoidance Subsystem: Dynamic Window Approach (DWA)');
xlabel('X (m)'); 
ylabel('Y (m)');

plot([sim.room(1) sim.room(2) sim.room(2) sim.room(1) sim.room(1)], ...
     [sim.room(3) sim.room(3) sim.room(4) sim.room(4) sim.room(3)], ...
     'k-', 'LineWidth', 2);

th = linspace(0,2*pi,120);

% Dummy handles for legend entries
hObstacle = plot(nan, nan, 'r-', 'LineWidth', 1.8);
hSafety   = plot(nan, nan, 'r-', 'LineWidth', 0.9);

% Plot obstacles and safety buffers
for i = 1:size(sim.obs,1)
    cx = sim.obs(i,1);
    cy = sim.obs(i,2);
    r_obs = sim.obs(i,3);
    r_inf = sim.obs(i,3) + cfg.inflate;

    % Outer safety/inflated circle
    plot(cx + r_inf*cos(th), cy + r_inf*sin(th), ...
        'r-', 'LineWidth', 0.9);

    % Physical obstacle circle
    plot(cx + r_obs*cos(th), cy + r_obs*sin(th), ...
        'r-', 'LineWidth', 1.8);
end

% Main plotted elements
hGlobal = plot(sim.globalPath(:,1), sim.globalPath(:,2), 'g--', 'LineWidth', 2);
hStart  = plot(sim.p0(1), sim.p0(2), 'bo', 'MarkerFaceColor','b', 'MarkerSize',10);
hGoal   = plot(sim.goal(1), sim.goal(2), 'go', 'MarkerFaceColor','g', 'MarkerSize',10);

hDrone  = plot(sim.p0(1), sim.p0(2), 'b.', 'MarkerSize',40);
hPath   = plot(sim.p0(1), sim.p0(2), 'b-', 'LineWidth',2);
hCarrot = plot(sim.p0(1), sim.p0(2), 'mo', 'MarkerFaceColor','m', 'MarkerSize',8);

text(sim.p0(1)+0.12, sim.p0(2)-0.18, 'Start', ...
    'FontSize', 9, 'Color', 'b', 'FontWeight', 'bold');

text(sim.goal(1)+0.12, sim.goal(2), 'Goal', ...
    'FontSize', 9, 'Color', [0 0.5 0], 'FontWeight', 'bold');

legend([hStart, hGoal, hGlobal, hPath, hCarrot, hObstacle, hSafety], ...
    {'Start', 'Goal', 'Global A* Path', 'UAV Path', 'Carrot Point', ...
     'Obstacle', 'Inflated Safety Buffer'}, ...
    'Location', 'northeastoutside');

%% ==================== STATE ====================
st.p = sim.p0;
st.psi = atan2(sim.goal(2)-sim.p0(2), sim.goal(1)-sim.p0(1));
st.v = 0;
st.w = 0;

path_hist = st.p;

stall = 0;
last_goal_dist = norm(st.p - sim.goal);

%% ==================== MAIN LOOP ====================
for step = 1:cfg.maxSteps

    % -------- Sense --------
    pose = getPoseSim(st);
    obstacles = getObstaclesSim(sim);
    globalPath = sim.globalPath;
    path_s = sim.path_s;
    goal = sim.goal;

    % -------- Stop if reached goal --------
    goal_dist = norm(pose.p - goal);
    if goal_dist < cfg.goal_tol
        fprintf('Reached goal in %d steps (%.1f s)\n', step, step*cfg.dt);
        publishCmdSim(0, 0);
        break;
    end

    % -------- Plan --------
    [v_cmd, w_cmd, carrot, ~, debug] = computeCmdDWA( ...
        pose, obstacles, globalPath, path_s, goal, cfg);

    % -------- Act --------
    publishCmdSim(v_cmd, w_cmd);

    % -------- Update sim state --------
    st = applyCmdSim(st, v_cmd, w_cmd, cfg, sim);

    % -------- Visualization --------
    path_hist(end+1,:) = st.p; %#ok<AGROW>
    set(hDrone, 'XData', st.p(1), 'YData', st.p(2));
    set(hPath,  'XData', path_hist(:,1), 'YData', path_hist(:,2));
    set(hCarrot,'XData', carrot(1), 'YData', carrot(2));

    drawnow limitrate;

    % -------- Stall update --------
    if goal_dist < last_goal_dist - 0.01
        stall = max(0, stall-2);
    else
        stall = stall + 1;
    end
    last_goal_dist = goal_dist;

    if stall > 140
        stall = 120;
    end

    if mod(step,80)==0
        fprintf('step=%d dist=%.2f v=%.2f clear=%.2f\n', ...
            step, goal_dist, st.v, debug.clear_now);
    end
end

end

%% ======================================================================
%% =================== CORE: COMPUTE CMD ================================
function [v_cmd, w_cmd, carrot, best_traj, dbg] = computeCmdDWA(pose, obstacles, globalPath, path_s, goal, cfg)

p = pose.p;
psi = pose.psi;
v = pose.v;
w = pose.w;

% Clearance at current pose
clear_now = minClearancePointCircles(p, obstacles, cfg.inflate);
dbg.clear_now = clear_now;

% Adaptive horizon + sampling + lookahead
if clear_now < 0.9
    L = cfg.L_tight;
    predT = cfg.pred_time_tight;
    Nv = max(cfg.Nv_base, 15);
    Nw = max(cfg.Nw_base, 35);
else
    L = cfg.L_open;
    predT = cfg.pred_time_open;
    Nv = cfg.Nv_base;
    Nw = cfg.Nw_base;
end

% Carrot from global path
s_now = pathProgressArc(p, globalPath, path_s);
carrot = pointAtArc(globalPath, path_s, s_now + L);

% Dynamic window
v_lo = max(cfg.v_min, v - cfg.a_max*cfg.dt);
v_hi = min(cfg.v_max, v + cfg.a_max*cfg.dt);
w_lo = max(-cfg.w_max, w - cfg.alpha_max*cfg.dt);
w_hi = min( cfg.w_max, w + cfg.alpha_max*cfg.dt);

% Always allow some forward motion
v_hi = max(v_hi, 0.25);
v_lo = min(v_lo, 0.10);

v_s = linspace(v_lo, v_hi, Nv);
w_s = linspace(w_lo, w_hi, Nw);

bestScore = -Inf;
v_best = 0.2;
w_best = 0.0;
best_traj = [];

for vi = v_s
    for wi = w_s
        [traj, minclr, p_end, psi_end] = rolloutCircles(p, psi, vi, wi, ...
            predT, cfg.sim_dt, obstacles, cfg.inflate);
        if minclr <= 0
            continue;
        end

        % Scores
        s_clear = min(minclr/1.5, 1.0);

        angC = atan2(carrot(2)-p_end(2), carrot(1)-p_end(1));
        e_head = abs(wrapToPi(angC - psi_end));
        s_head = 1 - e_head/pi;

        dC = norm(carrot - p_end);
        s_dist = 1/(1+dC);

        s_speed = (vi - cfg.v_min)/(cfg.v_max - cfg.v_min + 1e-6);
        s_goal  = 1/(1+norm(goal - p_end));

        s_end = pathProgressArc(p_end, globalPath, path_s);
        adv = s_end - s_now;
        s_prog = clamp(adv/0.35, -1, 1);

        rot_pen = abs(wi)/cfg.w_max;

        if clear_now < 0.9
            w_clear = 5.0;
            w_speed = 0.15;
        else
            w_clear = 2.0;
            w_speed = 1.0;
        end

        score = 2.0*s_head + 1.6*s_dist + w_clear*s_clear + ...
                w_speed*s_speed + 1.2*s_goal + 4.0*s_prog - 0.25*rot_pen;

        if vi < 0.08
            score = score - 1.0;
        end

        if score > bestScore
            bestScore = score;
            v_best = vi;
            w_best = wi;
            best_traj = traj;
        end
    end
end

if ~isfinite(bestScore)
    [v_best, w_best] = fallbackTurnAndCreep(p, psi, carrot, obstacles, ...
        cfg.inflate, cfg.w_max);
    best_traj = [];
end

v_cmd = v_best;
w_cmd = w_best;

end

%% ======================================================================
%% ========================== SIM SENSORS ===============================
function pose = getPoseSim(st)
pose.p = st.p;
pose.psi = st.psi;
pose.v = st.v;
pose.w = st.w;
end

function obs = getObstaclesSim(sim)
obs = sim.obs;
end

function globalPath = getGlobalPathSim(sim, cfg)
gridRes = 0.10;
[occ, xg, yg] = buildOccupancyGrid(sim.room, sim.obs, cfg.inflate, gridRes);
globalPath = astarPlan(occ, xg, yg, sim.p0, sim.goal);

if isempty(globalPath)
    globalPath = [sim.p0; 2.2 1.3; 4.8 3.0; 6.7 4.6; sim.goal];
else
    globalPath = smoothPath(globalPath, sim.obs, sim.room, cfg.inflate);
end
end

function publishCmdSim(v_cmd, w_cmd) %#ok<INUSD>
% Placeholder for publishing
end

function st = applyCmdSim(st, v_cmd, w_cmd, cfg, sim)
st.v = clamp(st.v + clamp(v_cmd - st.v, -cfg.a_max*cfg.dt, cfg.a_max*cfg.dt), ...
    cfg.v_min, cfg.v_max);

st.w = clamp(st.w + clamp(w_cmd - st.w, -cfg.alpha_max*cfg.dt, cfg.alpha_max*cfg.dt), ...
    -cfg.w_max, cfg.w_max);

st.psi = wrapToPi(st.psi + st.w*cfg.dt);
st.p = st.p + [st.v*cos(st.psi), st.v*sin(st.psi)]*cfg.dt;

st.p(1) = clamp(st.p(1), sim.room(1)+cfg.inflate+0.05, sim.room(2)-cfg.inflate-0.05);
st.p(2) = clamp(st.p(2), sim.room(3)+cfg.inflate+0.05, sim.room(4)-cfg.inflate-0.05);
end

%% ======================================================================
%% ===================== ROLLOUT + CLEARANCE ============================
function [traj, minclr, p_end, psi_end] = rolloutCircles(p0, psi0, v, w, T, dt, obs, inflate)
N = max(1, round(T/dt));
traj = zeros(N,2);
p = p0;
psi = psi0;
minclr = Inf;

for k = 1:N
    psi = wrapToPi(psi + w*dt);
    p = p + [v*cos(psi), v*sin(psi)]*dt;
    traj(k,:) = p;

    for i = 1:size(obs,1)
        d = norm(p - obs(i,1:2)) - (obs(i,3) + inflate);
        minclr = min(minclr, d);
    end

    if minclr <= 0
        break;
    end
end

p_end = p;
psi_end = psi;
end

function d = minClearancePointCircles(p, obs, inflate)
d = Inf;
for i = 1:size(obs,1)
    d = min(d, norm(p - obs(i,1:2)) - (obs(i,3) + inflate));
end
end

function [v_cmd, w_cmd] = fallbackTurnAndCreep(p, psi, carrot, obs, inflate, w_max)
angC = atan2(carrot(2)-p(2), carrot(1)-p(1));
e = wrapToPi(angC - psi);
w_cmd = clamp(1.8*e, -w_max, w_max);
v_cmd = 0.12;

if minClearancePointCircles(p, obs, inflate) < 0.25
    v_cmd = 0.0;
    w_cmd = 0.6*sign(e + 1e-6);
end
end

%% ======================================================================
%% ========================= PATH UTILITIES =============================
function s = pathProgressArc(p, path, path_dists)
best_d2 = Inf;
best_s = 0;

for i = 1:size(path,1)-1
    a = path(i,:);
    b = path(i+1,:);
    ab = b-a;
    L2 = dot(ab,ab);
    if L2 < 1e-12
        continue;
    end

    t = dot(p-a, ab)/L2;
    t = clamp(t, 0, 1);
    proj = a + t*ab;
    d2 = sum((p-proj).^2);

    if d2 < best_d2
        best_d2 = d2;
        best_s = path_dists(i) + t*sqrt(L2);
    end
end

s = best_s;
end

function pt = pointAtArc(path, path_dists, s_query)
s_query = clamp(s_query, 0, path_dists(end));

for i = 1:length(path_dists)-1
    if path_dists(i) <= s_query && path_dists(i+1) >= s_query
        seg = (s_query - path_dists(i)) / (path_dists(i+1)-path_dists(i)+1e-12);
        pt = path(i,:) + seg*(path(i+1,:)-path(i,:));
        return;
    end
end

pt = path(end,:);
end

%% ======================================================================
%% ======================== GLOBAL A* (SIM) =============================
function [occ, xg, yg] = buildOccupancyGrid(room, obs, inflate, res)
xg = room(1):res:room(2);
yg = room(3):res:room(4);
[X,Y] = meshgrid(xg, yg);
occ = false(size(X));

occ((X-room(1))<inflate | (room(2)-X)<inflate | ...
    (Y-room(3))<inflate | (room(4)-Y)<inflate) = true;

for i = 1:size(obs,1)
    r = obs(i,3) + inflate;
    occ(hypot(X-obs(i,1), Y-obs(i,2)) <= r) = true;
end
end

function path = astarPlan(occ, xg, yg, start, goal)
[sx, sy] = world2grid(xg, yg, start);
[gx, gy] = world2grid(xg, yg, goal);

if occ(sy,sx) || occ(gy,gx)
    path = [];
    return;
end

[ny,nx] = size(occ);
open = [sx sy 0 heuristic(sx,sy,gx,gy)];
closed = false(ny,nx);
came = zeros(ny,nx,2);
gScore = inf(ny,nx);
gScore(sy,sx) = 0;

nbr = [1 0; -1 0; 0 1; 0 -1; 1 1; 1 -1; -1 1; -1 -1];

while ~isempty(open)
    [~,i] = min(open(:,4));
    cx = open(i,1);
    cy = open(i,2);
    open(i,:) = [];

    if closed(cy,cx)
        continue;
    end
    closed(cy,cx) = true;

    if cx == gx && cy == gy
        path = reconstruct(came, xg, yg, sx, sy, gx, gy);
        return;
    end

    for k = 1:size(nbr,1)
        nx2 = cx + nbr(k,1);
        ny2 = cy + nbr(k,2);

        if nx2<1 || nx2>nx || ny2<1 || ny2>ny
            continue;
        end
        if occ(ny2,nx2) || closed(ny2,nx2)
            continue;
        end

        cost = hypot(nbr(k,1), nbr(k,2));
        tent = gScore(cy,cx) + cost;

        if tent < gScore(ny2,nx2)
            gScore(ny2,nx2) = tent;
            came(ny2,nx2,:) = [cx cy];
            f = tent + heuristic(nx2,ny2,gx,gy);
            open(end+1,:) = [nx2 ny2 tent f]; %#ok<AGROW>
        end
    end
end

path = [];
end

function h = heuristic(x,y,gx,gy)
h = hypot(gx-x, gy-y);
end

function [ix,iy] = world2grid(xg, yg, xy)
[~,ix] = min(abs(xg-xy(1)));
[~,iy] = min(abs(yg-xy(2)));
end

function path = reconstruct(came, xg, yg, sx, sy, gx, gy)
cells = [gx gy];
cx = gx;
cy = gy;

while ~(cx==sx && cy==sy)
    px = came(cy,cx,1);
    py = came(cy,cx,2);
    if px==0 && py==0
        path = [];
        return;
    end
    cx = px;
    cy = py;
    cells = [cx cy; cells]; %#ok<AGROW>
end

path = [xg(cells(:,1))' yg(cells(:,2))'];
end

function path2 = smoothPath(path, obs, room, inflate) %#ok<INUSD>
path2 = path;
i = 1;

while i < size(path2,1)-1
    j = i+2;
    while j <= size(path2,1)
        if lineClear(path2(i,:), path2(j,:), obs, inflate)
            path2(i+1:j-1,:) = [];
            j = i+2;
        else
            j = j+1;
        end
    end
    i = i+1;
end
end

function ok = lineClear(a,b,obs,inflate)
ok = true;
for t = linspace(0,1,40)
    p = a*(1-t) + b*t;
    if minClearancePointCircles(p, obs, inflate) <= 0
        ok = false;
        return;
    end
end
end

%% ==================== UTIL ====================
function y = clamp(x, lo, hi)
y = min(max(x, lo), hi);
end

function a = wrapToPi(a)
a = mod(a+pi, 2*pi) - pi;
end