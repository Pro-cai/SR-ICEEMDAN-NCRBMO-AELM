function [BestF, BestX, curve] = NCRBMO(N, T, lb, ub, dim, fobj)
% Input: N - Population size, T - Maximum iterations, lb - Lower bound, ub - Upper bound, dim - Dimension, fobj - Objective function
% Output: BestF - Best fitness value, BestX - Best position, curve - Convergence curve

PopPos = zeros(N, dim);
PopFit = zeros(N, 1);

%% Population initialization with multiphase mapping and inverse generation strategy
PopPos = multi_chaotic_op(N, dim, ub, lb, fobj);
for i = 1:N
    PopFit(i) = fobj(PopPos(i, :));
end
BestF = inf;
BestX = [];

% Initialize global worst solution
WorstF = -inf;
WorstX = [];

for i = 1:N
    if PopFit(i) <= BestF
        BestF = PopFit(i);
        BestX = PopPos(i, :);
    end
    if PopFit(i) >= WorstF
        WorstF = PopFit(i);
        WorstX = PopPos(i, :);
    end
end

curve = zeros(T, 1);
Epsilon = 0.5; % Default parameter from RMBO algorithm

%% ------------------- Begin Iteration ----------------------------%
for It = 1:T
    for i = 1:N
        theta1 = (1 - It / T);
        B = 2 * log(1 / rand) * theta1;  %% Switching coefficient
        
        %% ------------------- 1. Global Exploration Phase -------------------%
        if B > 0.5
            %% ------------------- 1.1 Group Search Behavior -------------------%
            % Randomly select 2-5 individuals
            p = randi([2, 5]); % Default parameter from RMBO algorithm
            selected_index_p = randperm(N, p);
            Xp = PopPos(selected_index_p, :);
            Xpmean = mean(Xp);
            
            % Randomly select 10 to N individuals
            q = randi([10, N]); % Default parameter from RMBO algorithm
            selected_index_q = randperm(N, q);
            Xq = PopPos(selected_index_q, :);
            Xqmean = mean(Xq);
            
            % Randomly select an individual R1
            A = randperm(N);
            R1 = A(1);
            
            % Use RMBO search strategy
            if rand < Epsilon
                Y = PopPos(i, :) + (Xpmean - PopPos(R1, :)) .* rand;
            else
                Y = PopPos(i, :) + (Xqmean - PopPos(R1, :)) .* rand;
            end
            
            %% ------------------- 1.2 Foraging Behavior Integrated with Normal Cloud Model -------------------%
            % Generate random number to decide which strategy to use
            Q = rand(1);
            if Q < 0.5
                % Original foraging behavior strategy
                Z = (Y - BestX) .* Levy(dim) + rand(1) * mean(Y) * (1 - It/T)^(2 * It/T);
            else
                %% Normal cloud model
                Ex = BestX;                 % Normal cloud expectation
                En = exp(It/T);             % Normal cloud entropy
                He = En / 10^(-3);          % Normal cloud hyper-entropy
                
                % Generate normally distributed random numbers
                E_n = normrnd(En, He, 1, dim);
                
                % Generate normal random numbers
                ra = normrnd(Ex, abs(E_n), 1, dim);
                
                % Calculate membership function
                Z = exp(-(ra - Ex).^2 ./ (2 * E_n.^2));
                
                % Map membership values back to original search space
                Z = lb + Z .* (ub - lb);
            end
            
            % Boundary handling
            Y = SpaceBound(Y, ub, lb);
            Z = SpaceBound(Z, ub, lb);
            
            % Select better solution
            NewPop = [Y; Z];
            NewPopfit = [fobj(Y); fobj(Z)];
            [~, sorted_indexes] = sort(NewPopfit);
            newPopPos = NewPop(sorted_indexes(1), :);
            
        else
            %% ------------------- 2. Local Exploitation Phase -------------------%
            F = 0.5;
            K = [1:i - 1, i + 1:N];
            f = (0.1 * (rand - 1) * (T - It)) / T;
            
            while true
                RandInd = K(randi([1 N-1], 1, 3));
                step1 = PopPos(RandInd(2), :) - PopPos(RandInd(3), :);
                if norm(step1) ~= 0 && RandInd(2) ~= RandInd(3)
                    break;
                end
            end
            
            %% ------------------- 2.1 Attacking Prey Behavior -------------------%
            % Calculate convergence factor CF
            CF = (1 - It/T)^(2 * It/T);
            
            % Randomly select 2-5 individuals
            p = randi([2, 5]); % Default parameter from RMBO algorithm
            selected_index_p = randperm(N, p);
            Xp = PopPos(selected_index_p, :);
            Xpmean = mean(Xp);

            % Randomly select 10 to N individuals
            q = randi([10, N]); % Default parameter from RMBO algorithm
            selected_index_q = randperm(N, q);
            Xq = PopPos(selected_index_q, :);
            Xqmean = mean(Xq);

            
            if rand() < Epsilon
                W_R = BestX + CF * (Xpmean - PopPos(i, :)) .* randn(1, dim); % Based on small group mean
            else
                W_R = BestX + CF * (Xqmean - PopPos(i, :)) .* randn(1, dim); % Based on large group mean
            end
            
            %% ------------------- 2.2 Expansion Search Behavior -------------------%
            if rand < 0.5
                W_I = BestX + F .* step1;
            else
                W_I = BestX + F .* Levy(dim) .* step1;
            end
            Y = (1 + f) * W_I;
            
            %% ------------------- 2.3 Benefit-Seeking and Risk-Avoidance Behavior -------------------%
           
            Z = PopPos(i, :) + rand * (BestX - abs(PopPos(i, :))) - rand * (WorstX - abs(PopPos(i, :)));
            
            % Boundary handling
            W_R = SpaceBound(W_R, ub, lb);
            Y = SpaceBound(Y, ub, lb);
            Z = SpaceBound(Z, ub, lb);
            
            % Select better solution
            NewPop = [W_R; Y; Z];
            NewPopfit = [fobj(W_R); fobj(Y); fobj(Z)];
            [~, sorted_indexes] = sort(NewPopfit);
            newPopPos = NewPop(sorted_indexes(1), :);
        end
        
        % Boundary handling
        newPopPos = SpaceBound(newPopPos, ub, lb);
        newPopFit = fobj(newPopPos);
        
        % Greedy selection
        if newPopFit < PopFit(i)
            PopFit(i) = newPopFit;
            PopPos(i, :) = newPopPos;
        end
        
    end
    
    % Elite preservation mechanism
    for i = 1:N
        if PopFit(i) < BestF
            BestF = PopFit(i);
            BestX = PopPos(i, :);
        end
        if PopFit(i) > WorstF
            WorstF = PopFit(i);
            WorstX = PopPos(i, :);
        end
    end
    
    curve(It) = BestF;
end
end

%% Auxiliary Functions
function o = Levy(Dim)
beta = 1.5;
sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
u = randn(1, Dim) * sigma;
v = randn(1, Dim);
step = u ./ abs(v).^(1 / beta);
o = step;
end

function X = SpaceBound(X, Up, Low)
Dim = length(X);
S = (X > Up) + (X < Low);
X = (rand(1, Dim) .* (Up - Low) + Low) .* S + X .* (~S);
end

function pop = multi_chaotic_op(N, dim, ub, lb, fobj)
%% Multiphase chaotic mapping with inverse generation initialization strategy
% Parameter Description:
% m: Boundary dimension. Tent mapping is used for dimensions less than m,
%    Chebyshev mapping is used for dimensions greater than or equal to m.
% n: Order of Chebyshev polynomial, which determines different function forms.

% Set boundary dimension m (default: half of the total dimension)
m = ceil(dim/2);

% Initialize chaotic sequence
Chaotic = rand(N, dim);
SearchAgents_no = N;

% Selection of Chebyshev mapping order (commonly used orders: 2, 3, 4)
% Different values of n produce different chaotic characteristics:
% n=2: Produces unimodal distribution
% n=3: Produces bimodal distribution
% n=4: Produces multimodal distribution
% n=5: Produces more complex distribution forms
% Here we randomly select the order to increase diversity
n_values = [2, 3, 4, 5];

for i = 1:SearchAgents_no
    % Randomly select an order for the current individual
    n = n_values(randi(length(n_values)));
    
    for j = 2:dim
        if j < m
            % Low dimensions: Use Tent mapping
            alpha = 0.499; % Tent mapping parameter
            if Chaotic(i, j-1) < alpha
                Chaotic(i, j) = Chaotic(i, j-1) / alpha;
            else
                Chaotic(i, j) = (1 - Chaotic(i, j-1)) / (1 - alpha);
            end
        else
            % High dimensions: Use Chebyshev mapping
            % Map [0,1] to [-1,1]
            x_normalized = 2 * Chaotic(i, j-1) - 1;
            
            % Calculate Chebyshev polynomial
            % Formula: T_n(x) = cos(n * acos(x))
            chebyshev_value = cos(n * acos(x_normalized));
            
            % Map [-1,1] back to [0,1]
            Chaotic(i, j) = (chebyshev_value + 1) / 2;
        end
    end
end

X = Chaotic;
Positions = zeros(N, dim);
OP_Positions = zeros(N, dim);
Fit = zeros(1, N);
OP_Fit = zeros(1, N);

for i = 1:SearchAgents_no
    % Original population
    Positions(i, :) = lb + X(i, :) .* (ub - lb);
    
    % Inverse generation strategy
    % Use random opposition-based learning to increase diversity
    OP_Positions(i, :) = lb + ub - Positions(i, :);
    
    % Add random perturbation to enhance exploration capability
    random_factor = 0.1 * rand(1, dim);
    OP_Positions(i, :) = OP_Positions(i, :) + random_factor .* (ub - lb);
    
    % Calculate fitness
    Fit(i) = feval(fobj, Positions(i, :));
    
    % Boundary handling
    OP_Positions(i, :) = max(OP_Positions(i, :), lb);
    OP_Positions(i, :) = min(OP_Positions(i, :), ub);
    
    OP_Fit(i) = feval(fobj, OP_Positions(i, :));
end

% Merge original and inverse-generated populations, select the best N individuals
All_Fit = [Fit, OP_Fit];
All_Positions = [Positions; OP_Positions];

% Sort by fitness
[~, index] = sort(All_Fit);

% Select the first N best individuals
pop = All_Positions(index(1:N), :);
end